import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from config import DATA_DIR, LOG_DIR
from pipeline.filter import score_and_filter
from pipeline.ingest import ingest
from pipeline.llm import enrich_tenders_with_llm
from pipeline.notify_email import render_digest_html, send_digest_email

STATE_PATH = DATA_DIR / "state.json"
EXPORT_DIR = DATA_DIR / "exports"


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
            return json.loads(content)
    return {"last_run_iso": None, "seen_ids": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _default_export_path() -> Path:
    now_local = _now_local()
    return EXPORT_DIR / f"flagged_{now_local:%Y%m%d}.csv"


def main() -> int:
    load_dotenv()
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
        "--export-csv",
        nargs="?",
        const="__default__",
        help="Write full flagged list to CSV (optional path).",
    )
    parser.add_argument("--no-llm", action="store_true", help="Skip Cohere enrichment.")
    args = parser.parse_args()

    logger = _setup_logging()
    state = load_state()
    last_run_iso = state.get("last_run_iso")
    seen = set(state.get("seen_ids", []))
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

    flagged, reasons = score_and_filter(new_items)
    for k, v in reasons.most_common(15):
        logger.info("Reject: %s=%s", k, v)

    logger.info("New=%s Flagged=%s", len(new_items), len(flagged))

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

    flagged_top = flagged[: max(0, args.top_n)]

    if args.dry_run:
        html_path = EXPORT_DIR / f"digest_{datetime.now():%Y%m%d_%H%M%S}.html"
        html = render_digest_html(flagged_top, flagged, csv_path=csv_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html, encoding="utf-8")
        logger.info("Dry-run: wrote HTML to %s", html_path)
    elif args.bootstrap:
        logger.info("Bootstrap mode: skipping email send.")
    else:
        if flagged_top:
            try:
                send_digest_email(flagged_top, all_flagged=flagged, csv_path=csv_path)
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

    for t in new_items:
        if t.get("uid"):
            seen.add(t["uid"])

    state["seen_ids"] = list(seen)
    state["last_run_iso"] = now_utc_iso()
    save_state(state)
    logger.info("State updated: last_run_iso=%s seen=%s", state["last_run_iso"], len(seen))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
