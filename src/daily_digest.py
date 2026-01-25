import argparse
import csv
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from config import DATA_DIR, LOG_DIR
from pipeline.filter import score_and_filter
from pipeline.ingest import ingest
from pipeline.llm import enrich_tenders_with_llm
from pipeline.notify_email import render_digest_html, send_digest_email

STATE_PATH = DATA_DIR / "state.json"
EXPORT_DIR = DATA_DIR / "exports"
DEFAULT_SEEN_RETENTION_DAYS = 180
DEFAULT_WATCHLIST_MAX = 15
DEFAULT_WATCHLIST_BLOCKLIST = {
    "exclude_supply_arrangement",
    "exclude_staffing",
    "exclude_negative_title",
    "exclude_non_it_unspsc",
    "exclude_goods_or_hardware",
    "exclude_no_erp_domain",
    "exclude_far_future_close",
}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return list(default)
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _now_local() -> datetime:
    try:
        return datetime.now(ZoneInfo("America/Toronto"))
    except Exception:
        return datetime.now()


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now_local = _now_local()
    log_path = LOG_DIR / f"run_{now_local:%Y%m%d}.log"

    logger = logging.getLogger("sap_tender_bot")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    logger.info("Logging to %s", log_path)
    return logger


def load_state() -> dict:
    if STATE_PATH.exists():
        content = STATE_PATH.read_text(encoding="utf-8").strip()
        if content:
            state = json.loads(content)
        else:
            state = {}
    else:
        state = {}

    if "seen" not in state:
        seen_ids = state.get("seen_ids", []) or []
        last_seen = state.get("last_run_iso") or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        state["seen"] = {uid: last_seen for uid in seen_ids if uid}

    state.setdefault("last_run_iso", None)
    return state


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    state.pop("seen_ids", None)
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_iso_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _prune_seen(seen: dict, retention_days: int) -> dict:
    if retention_days <= 0:
        return seen
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    pruned = {}
    for uid, last_seen in seen.items():
        last_dt = _parse_iso_dt(str(last_seen))
        if last_dt is None or last_dt >= cutoff:
            pruned[uid] = last_seen
    return pruned


def _export_csv(tenders: Iterable[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "uid",
        "id",
        "amendment_number",
        "title",
        "org",
        "publish_date",
        "amendment_date",
        "updated_at",
        "close_date",
        "score",
        "url",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for t in tenders:
            writer.writerow(t)
    return path


def _export_near_miss(rejected: Iterable[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "uid",
        "id",
        "amendment_number",
        "title",
        "org",
        "updated_at",
        "close_date",
        "url",
        "_reject_reason",
        "_hits",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for t in rejected:
            reasons = t.get("_reject_reasons")
            if isinstance(reasons, list) and len(reasons) != 1:
                continue
            row = dict(t)
            if "_hits" in row:
                row["_hits"] = json.dumps(row.get("_hits", {}), ensure_ascii=False)
            writer.writerow(row)
    return path


def _export_reject_report(reasons: dict, totals: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": now_utc_iso(),
        "totals": totals,
        "reasons": dict(reasons),
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _default_export_path() -> Path:
    now_local = _now_local()
    return EXPORT_DIR / f"flagged_{now_local:%Y%m%d}.csv"


def _default_near_miss_path() -> Path:
    now_local = _now_local()
    return EXPORT_DIR / f"near_miss_{now_local:%Y%m%d}.csv"


def _default_report_path() -> Path:
    now_local = _now_local()
    return EXPORT_DIR / f"reject_report_{now_local:%Y%m%d_%H%M%S}.json"


def main() -> int:
    load_dotenv()
    env_min_score = _env_int("SAP_TENDER_MIN_SCORE_NON_SAP", 70)
    env_max_close_days = _env_int("SAP_TENDER_MAX_NON_SAP_CLOSE_DAYS", 365)
    env_include_staffing = _env_bool("SAP_TENDER_INCLUDE_STAFFING", False)
    env_include_supply = _env_bool("SAP_TENDER_INCLUDE_SUPPLY_ARRANGEMENTS", False)
    env_store_hits = _env_bool("SAP_TENDER_STORE_HITS", True)
    env_seen_retention = _env_int("SAP_TENDER_SEEN_RETENTION_DAYS", DEFAULT_SEEN_RETENTION_DAYS)
    env_watchlist_max = _env_int("SAP_TENDER_WATCHLIST_MAX", DEFAULT_WATCHLIST_MAX)
    env_watchlist_blocklist = _env_list(
        "SAP_TENDER_WATCHLIST_BLOCKLIST",
        sorted(DEFAULT_WATCHLIST_BLOCKLIST),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't send email or update state.")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="First-run mode: seed state without sending email.",
    )
    parser.add_argument(
        "--since-iso",
        help="Override last_run_iso for backfills (e.g. 2025-12-01T00:00:00Z).",
    )
    parser.add_argument("--top-n", type=int, default=25, help="Max tenders to email.")
    parser.add_argument(
        "--min-score-non-sap",
        type=int,
        default=env_min_score,
        help="Minimum score to keep non-SAP-direct tenders.",
    )
    parser.add_argument(
        "--max-non-sap-close-days",
        type=int,
        default=env_max_close_days,
        help="Exclude non-SAP tenders closing further out than this many days.",
    )
    parser.add_argument(
        "--include-staffing",
        action=argparse.BooleanOptionalAction,
        default=env_include_staffing,
        help="Include staffing/resource-augmentation tenders.",
    )
    parser.add_argument(
        "--include-supply-arrangements",
        action=argparse.BooleanOptionalAction,
        default=env_include_supply,
        help="Include supply arrangements / standing offers.",
    )
    parser.add_argument(
        "--store-hits",
        action=argparse.BooleanOptionalAction,
        default=env_store_hits,
        help="Store explainability hits on each tender.",
    )
    parser.add_argument(
        "--seen-retention-days",
        type=int,
        default=env_seen_retention,
        help="Retention window for seen_ids state.",
    )
    parser.add_argument(
        "--watchlist-max",
        type=int,
        default=env_watchlist_max,
        help="Maximum number of watchlist items to include.",
    )
    parser.add_argument(
        "--watchlist-blocklist",
        default=None,
        help="Comma-separated reject reasons to exclude from watchlist.",
    )
    parser.add_argument(
        "--export-csv",
        nargs="?",
        const="__default__",
        help="Write full flagged list to CSV (optional path).",
    )
    parser.add_argument(
        "--export-near-miss",
        nargs="?",
        const="__default__",
        help="Write rejected items with reasons to CSV (optional path).",
    )
    parser.add_argument(
        "--export-report",
        nargs="?",
        const="__default__",
        help="Write rejection summary report to JSON (optional path).",
    )
    parser.add_argument("--no-llm", action="store_true", help="Skip Cohere enrichment.")
    args = parser.parse_args()

    logger = _setup_logging()
    state = load_state()
    last_run_iso = state.get("last_run_iso")
    seen_map = state.get("seen", {}) or {}
    pruned_seen = _prune_seen(seen_map, args.seen_retention_days)
    if len(pruned_seen) != len(seen_map):
        logger.info("Pruned seen_ids: %s -> %s", len(seen_map), len(pruned_seen))
    seen_map = pruned_seen
    state["seen"] = seen_map
    seen = set(seen_map)
    since_iso = args.since_iso or last_run_iso

    logger.info("State last_run_iso=%s seen=%s", last_run_iso, len(seen))
    logger.info("Fetching CanadaBuys tenders since=%s", since_iso)

    try:
        tenders = ingest(since_iso=since_iso)
    except Exception:
        logger.exception("Ingest failed")
        return 1

    new_items = []
    for t in tenders:
        uid = t.get("uid")
        if uid and uid not in seen:
            new_items.append(t)

    if args.dry_run and args.export_near_miss is None:
        args.export_near_miss = "__default__"
    if args.dry_run and args.export_report is None:
        args.export_report = "__default__"

    filter_config = {
        "include_supply_arrangements": args.include_supply_arrangements,
        "include_staffing": args.include_staffing,
        "max_non_sap_close_days": args.max_non_sap_close_days,
        "min_score_non_sap": args.min_score_non_sap,
        "store_hits": args.store_hits,
    }
    flagged, reasons, rejected = score_and_filter(new_items, return_rejected=True, config=filter_config)
    for k, v in reasons.most_common(15):
        logger.info("Reject: %s=%s", k, v)

    logger.info("New=%s Flagged=%s", len(new_items), len(flagged))

    watchlist_blocklist = env_watchlist_blocklist
    if args.watchlist_blocklist is not None:
        watchlist_blocklist = [
            v.strip() for v in str(args.watchlist_blocklist).split(",") if v.strip()
        ]
    watchlist_blocklist_set = set(watchlist_blocklist)

    watchlist = []
    for t in rejected:
        reasons_list = t.get("_reject_reasons")
        if isinstance(reasons_list, list) and len(reasons_list) == 1:
            reason = reasons_list[0]
            if reason not in watchlist_blocklist_set:
                watchlist.append(t)
    if len(watchlist) > args.watchlist_max:
        watchlist = watchlist[: args.watchlist_max]

    if not args.no_llm and flagged:
        try:
            enrich_tenders_with_llm(flagged, logger=logger)
        except Exception:
            logger.exception("LLM enrichment failed; continuing without LLM output.")

    csv_path = None
    if args.export_csv is not None:
        csv_path = _default_export_path() if args.export_csv == "__default__" else Path(args.export_csv)
        try:
            _export_csv(flagged, csv_path)
            logger.info("Exported CSV to %s", csv_path)
        except Exception:
            logger.exception("CSV export failed")

    near_miss_path = None
    if args.export_near_miss is not None:
        near_miss_path = (
            _default_near_miss_path()
            if args.export_near_miss == "__default__"
            else Path(args.export_near_miss)
        )
        try:
            _export_near_miss(rejected, near_miss_path)
            logger.info("Exported near-miss CSV to %s", near_miss_path)
        except Exception:
            logger.exception("Near-miss export failed")

    report_path = None
    if args.export_report is not None:
        report_path = (
            _default_report_path()
            if args.export_report == "__default__"
            else Path(args.export_report)
        )
        try:
            totals = {
                "new": len(new_items),
                "flagged": len(flagged),
                "rejected": len(rejected),
            }
            _export_reject_report(reasons, totals, report_path)
            logger.info("Exported reject report to %s", report_path)
        except Exception:
            logger.exception("Reject report export failed")

    flagged_top = flagged[: max(0, args.top_n)]

    if args.dry_run:
        html_path = EXPORT_DIR / f"digest_{datetime.now():%Y%m%d_%H%M%S}.html"
        html = render_digest_html(flagged_top, flagged, csv_path=csv_path, watchlist=watchlist)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html, encoding="utf-8")
        logger.info("Dry-run: wrote HTML to %s", html_path)
    elif args.bootstrap:
        logger.info("Bootstrap mode: skipping email send.")
    else:
        if flagged_top:
            try:
                send_digest_email(flagged_top, all_flagged=flagged, csv_path=csv_path, watchlist=watchlist)
                logger.info("Email sent: %s tenders", len(flagged_top))
            except Exception:
                logger.exception("Email send failed")
                return 1
        else:
            logger.info("No flagged tenders to send.")

    if args.dry_run:
        logger.info("Dry-run: state not updated.")
        return 0

    if args.bootstrap and last_run_iso is not None:
        logger.info("Bootstrap requested but state already has last_run_iso; updating anyway.")

    run_iso = now_utc_iso()
    for t in new_items:
        uid = t.get("uid")
        if uid:
            seen_map[uid] = run_iso

    state["seen"] = seen_map
    state["last_run_iso"] = run_iso
    save_state(state)
    logger.info("State updated: last_run_iso=%s seen=%s", state["last_run_iso"], len(seen_map))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
