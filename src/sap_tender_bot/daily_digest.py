from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from sap_tender_bot.config import AppConfig, load_config
from sap_tender_bot.pipeline.attachments import (
    enrich_tenders_with_attachments,
    merge_attachment_requirements,
)
from sap_tender_bot.pipeline.filter import score_and_filter
from sap_tender_bot.pipeline.ingest import ingest
from sap_tender_bot.pipeline.llm import enrich_tenders_with_llm
from sap_tender_bot.pipeline.notify_email import render_digest_html, send_digest_email
from sap_tender_bot.store import RunRecord, TenderStore


def _now_local(tz_name: str) -> datetime:
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return datetime.now()


def _setup_logging(log_dir: Path, tz_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    now_local = _now_local(tz_name)
    log_path = log_dir / f"run_{now_local:%Y%m%d}.log"

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


EVENT_LABELS = {
    "new": "New tender",
    "amendment": "Amendment posted",
    "close_date_changed": "Close date changed",
    "new_attachment": "New attachment",
}


def _event_label_list(events: Iterable[str]) -> list[str]:
    labels = []
    for e in events:
        label = EVENT_LABELS.get(e, str(e))
        if label:
            labels.append(label)
    return labels


def _tender_key_from_uid(uid: Optional[str]) -> Optional[str]:
    if not uid:
        return None
    uid = str(uid)
    if ":am" in uid:
        return uid.rsplit(":am", 1)[0]
    return uid


def _infer_tender_key(t: dict) -> str:
    tender_key = t.get("tender_key")
    if tender_key:
        return str(tender_key)
    uid_key = _tender_key_from_uid(t.get("uid"))
    if uid_key:
        return uid_key
    source = str(t.get("source") or "unknown")
    base = t.get("id") or t.get("solicitation_number") or t.get("url") or ""
    if base:
        return f"{source}:{base}"
    blob = json.dumps(t, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"


def _attachment_hash(attachments: Iterable[str]) -> str:
    cleaned = sorted({str(a).strip() for a in attachments if a and str(a).strip()})
    if not cleaned:
        return ""
    payload = json.dumps(cleaned, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_amendment_number(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _latest_amendment(prev: Optional[str], new: Optional[str]) -> Optional[str]:
    prev_i = _parse_amendment_number(prev)
    new_i = _parse_amendment_number(new)
    if prev_i is None:
        return str(new) if new is not None else None
    if new_i is None:
        return str(prev)
    return str(new) if new_i >= prev_i else str(prev)


def _detect_events(t: dict, existing: Optional[dict]) -> list[str]:
    if not existing:
        return ["new"]

    events: list[str] = []
    new_amend = _parse_amendment_number(t.get("amendment_number"))
    old_amend = _parse_amendment_number(existing.get("last_amendment_seen"))
    if new_amend is not None and old_amend is not None and new_amend > old_amend:
        events.append("amendment")

    new_close = t.get("close_date")
    old_close = existing.get("last_close_date_seen")
    if new_close and old_close and str(new_close) != str(old_close):
        events.append("close_date_changed")

    new_hash = t.get("_attachment_hash") or ""
    old_hash = existing.get("last_attachment_hash") or ""
    if new_hash and new_hash != old_hash:
        events.append("new_attachment")

    return events


def _upcoming_closes(
    tenders: Iterable[dict],
    *,
    days: int = 7,
    max_items: int = 15,
) -> list[dict]:
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=days)
    upcoming: list[tuple[datetime, dict]] = []
    for t in tenders:
        close_dt = _parse_iso_dt(t.get("close_date"))
        if not close_dt:
            continue
        if now <= close_dt <= horizon:
            upcoming.append((close_dt, t))
    upcoming.sort(key=lambda pair: pair[0])
    return [t for _, t in upcoming[:max_items]]


def _tender_sort_key(t: dict) -> tuple[int, datetime]:
    amend = _parse_amendment_number(t.get("amendment_number")) or -1
    updated = _parse_iso_dt(t.get("updated_at")) or datetime.min.replace(tzinfo=timezone.utc)
    return amend, updated


def _dedupe_tenders(tenders: Iterable[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    for t in tenders:
        key = _infer_tender_key(t)
        t["tender_key"] = key
        current = latest.get(key)
        if current is None or _tender_sort_key(t) > _tender_sort_key(current):
            latest[key] = t
    return list(latest.values())


def _score_for_llm(t: dict) -> int:
    for key in ("score", "semantic_score", "_rule_score"):
        value = t.get(key)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


def _export_csv(tenders: Iterable[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tender_key",
        "uid",
        "id",
        "solicitation_number",
        "amendment_number",
        "title",
        "org",
        "publish_date",
        "amendment_date",
        "updated_at",
        "close_date",
        "first_seen",
        "last_seen",
        "last_amendment_seen",
        "last_close_date_seen",
        "events",
        "score",
        "semantic_bucket",
        "semantic_similarity",
        "semantic_score",
        "rationale",
        "url",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for t in tenders:
            row = dict(t)
            events = row.get("_events") or row.get("events") or []
            if isinstance(events, list):
                row["events"] = "; ".join(_event_label_list(events))
            writer.writerow(row)
    return path


def _export_near_miss(rejected: Iterable[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tender_key",
        "uid",
        "id",
        "amendment_number",
        "title",
        "org",
        "updated_at",
        "close_date",
        "events",
        "url",
        "_reject_reason",
        "semantic_bucket",
        "semantic_similarity",
        "semantic_score",
        "rationale",
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
            events = row.get("_events") or row.get("events") or []
            if isinstance(events, list):
                row["events"] = "; ".join(_event_label_list(events))
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


def _default_export_path(export_dir: Path, tz_name: str) -> Path:
    now_local = _now_local(tz_name)
    return export_dir / f"flagged_{now_local:%Y%m%d}.csv"


def _default_near_miss_path(export_dir: Path, tz_name: str) -> Path:
    now_local = _now_local(tz_name)
    return export_dir / f"near_miss_{now_local:%Y%m%d}.csv"


def _default_report_path(export_dir: Path, tz_name: str) -> Path:
    now_local = _now_local(tz_name)
    return export_dir / f"reject_report_{now_local:%Y%m%d_%H%M%S}.json"


def _load_settings(config_path: Optional[str]) -> AppConfig:
    path = Path(config_path).expanduser().resolve() if config_path else None
    return load_config(path)


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", help="Path to YAML config.")
    pre_args, _ = pre_parser.parse_known_args()

    settings = _load_settings(pre_args.config)
    tz_name = settings.timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config.")
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
        default=settings.rules.min_score_non_sap,
        help="Minimum score to keep non-SAP-direct tenders.",
    )
    parser.add_argument(
        "--max-non-sap-close-days",
        type=int,
        default=settings.rules.max_non_sap_close_days,
        help="Exclude non-SAP tenders closing further out than this many days.",
    )
    parser.add_argument(
        "--include-staffing",
        action=argparse.BooleanOptionalAction,
        default=settings.rules.include_staffing,
        help="Include staffing/resource-augmentation tenders.",
    )
    parser.add_argument(
        "--include-supply-arrangements",
        action=argparse.BooleanOptionalAction,
        default=settings.rules.include_supply_arrangements,
        help="Include supply arrangements / standing offers.",
    )
    parser.add_argument(
        "--store-hits",
        action=argparse.BooleanOptionalAction,
        default=settings.rules.store_hits,
        help="Store explainability hits on each tender.",
    )
    parser.add_argument(
        "--seen-retention-days",
        type=int,
        default=settings.digest.seen_retention_days,
        help="Retention window for seen_ids state.",
    )
    parser.add_argument(
        "--watchlist-max",
        type=int,
        default=settings.digest.watchlist_max,
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
    parser.add_argument(
        "--attachments",
        action=argparse.BooleanOptionalAction,
        default=settings.attachments.enabled,
        help="Enable attachment enrichment.",
    )
    args = parser.parse_args()

    if args.config and args.config != pre_args.config:
        settings = _load_settings(args.config)
        tz_name = settings.timezone

    paths = settings.paths
    logger = _setup_logging(paths.log_dir, tz_name)
    store = TenderStore(paths.db_path, logger=logger).open()
    run_id = None
    run_started = now_utc_iso()
    last_run_iso = store.get_last_run_iso()
    since_iso = args.since_iso or last_run_iso

    if last_run_iso is None and args.since_iso is None:
        logger.info("No prior run found; consider --bootstrap to seed state.")

    logger.info("SQLite store: %s", paths.db_path)
    logger.info("Last successful run: %s", last_run_iso)
    logger.info("Fetching CanadaBuys tenders since=%s", since_iso)

    try:
        tenders = ingest(since_iso=since_iso, config=settings)
    except Exception:
        logger.exception("Ingest failed")
        if run_id is not None:
            store.finish_run(run_id, finished_at=now_utc_iso(), status="error", error="ingest_failed")
        store.close()
        return 1
    tenders = _dedupe_tenders(tenders)

    if not args.dry_run:
        run_id = store.start_run(
            RunRecord(
                started_at=run_started,
                since_iso=since_iso,
                bootstrap=args.bootstrap,
                dry_run=args.dry_run,
            )
        )

    event_candidates: list[dict] = []
    new_count = 0
    updated_count = 0

    for t in tenders:
        tender_key = t.get("tender_key") or _infer_tender_key(t)
        t["tender_key"] = tender_key

        attachments = t.get("attachments") or []
        if isinstance(attachments, str):
            attachments = [attachments]
        t["attachments"] = attachments
        t["_attachment_hash"] = _attachment_hash(attachments)

        existing = store.fetch_tender(tender_key)
        events = _detect_events(t, existing)
        t["_events"] = events

        if events:
            event_candidates.append(t)
            if "new" in events:
                new_count += 1
            else:
                updated_count += 1
        first_seen = (
            existing.get("first_seen")
            if existing and existing.get("first_seen")
            else run_started
        )
        last_seen = run_started
        last_amendment_seen = _latest_amendment(
            existing.get("last_amendment_seen") if existing else None,
            t.get("amendment_number"),
        )
        last_close_date_seen = t.get("close_date") or (
            existing.get("last_close_date_seen") if existing else None
        )
        last_attachment_hash = t.get("_attachment_hash") or (
            existing.get("last_attachment_hash") if existing else None
        )

        t["first_seen"] = first_seen
        t["last_seen"] = last_seen
        t["last_amendment_seen"] = last_amendment_seen
        t["last_close_date_seen"] = last_close_date_seen

        if args.dry_run:
            continue

        store.upsert_tender(
            tender_key=tender_key,
            source=str(t.get("source") or ""),
            uid=str(t.get("uid") or ""),
            ref_id=str(t.get("id") or ""),
            solicitation_number=str(t.get("solicitation_number") or ""),
            title=str(t.get("title") or ""),
            org=str(t.get("org") or ""),
            url=str(t.get("url") or ""),
            first_seen=first_seen,
            last_seen=last_seen,
            last_amendment_seen=last_amendment_seen,
            last_close_date_seen=last_close_date_seen,
            last_attachment_hash=last_attachment_hash,
            payload=t,
        )

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
    flagged, reasons, rejected = score_and_filter(
        event_candidates,
        return_rejected=True,
        config=filter_config,
        semantic_config=settings.embeddings,
        cache_dir=paths.cache_dir,
        logger=logger,
    )
    for k, v in reasons.most_common(15):
        logger.info("Reject: %s=%s", k, v)

    logger.info(
        "Events: new=%s updated=%s candidates=%s flagged=%s",
        new_count,
        updated_count,
        len(event_candidates),
        len(flagged),
    )

    if run_id is not None and not args.dry_run:
        for t in flagged:
            tender_key = t.get("tender_key")
            if tender_key:
                store.update_tender_payload(tender_key=str(tender_key), payload=t)

    watchlist_blocklist = list(settings.digest.watchlist_blocklist)
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

    if run_id is not None:
        decision_rows: list[dict] = []
        created_at = now_utc_iso()
        for t in flagged:
            decision_rows.append(
                {
                    "run_id": run_id,
                    "tender_key": t.get("tender_key"),
                    "uid": t.get("uid"),
                    "decision": "kept",
                    "score": t.get("score"),
                    "reject_reason": None,
                    "events": json.dumps(t.get("_events") or []),
                    "created_at": created_at,
                }
            )
        for t in rejected:
            decision_rows.append(
                {
                    "run_id": run_id,
                    "tender_key": t.get("tender_key"),
                    "uid": t.get("uid"),
                    "decision": "rejected",
                    "score": t.get("score"),
                    "reject_reason": t.get("_reject_reason"),
                    "events": json.dumps(t.get("_events") or []),
                    "created_at": created_at,
                }
            )
        store.record_decisions(decision_rows)

    use_llm = (not args.no_llm) and settings.llm.enabled
    flagged_top = flagged[: max(0, args.top_n)]
    llm_slice = max(0, args.top_n + int(settings.llm.borderline_max))
    llm_candidates = flagged[:llm_slice]

    watchlist_min_score = int(getattr(settings.llm, "watchlist_min_score", 0) or 0)
    watchlist_llm = [
        t for t in watchlist if _score_for_llm(t) >= watchlist_min_score
    ]
    if watchlist_llm:
        merged: dict[str, dict] = {}
        for t in llm_candidates + watchlist_llm:
            key = str(t.get("tender_key") or t.get("uid") or "")
            if not key:
                key = str(id(t))
            merged.setdefault(key, t)
        llm_candidates = list(merged.values())

    per_run_max = int(getattr(settings.llm, "per_run_max", 0) or 0)
    if per_run_max > 0:
        llm_candidates = llm_candidates[:per_run_max]

    if use_llm and llm_candidates:
        try:
            stats = enrich_tenders_with_llm(
                llm_candidates,
                config=settings.llm,
                cache_dir=paths.cache_dir,
                logger=logger,
                max_requests=per_run_max if per_run_max > 0 else None,
            )
            if logger:
                logger.info(
                    "LLM: requested=%s cached=%s skipped=%s daily_remaining=%s req_with=%s conf_hi=%s conf_med=%s conf_low=%s",
                    stats.get("requested"),
                    stats.get("cached"),
                    stats.get("skipped"),
                    stats.get("daily_remaining"),
                    stats.get("with_requirements"),
                    stats.get("confidence_high"),
                    stats.get("confidence_medium"),
                    stats.get("confidence_low"),
                )
            if run_id is not None and not args.dry_run:
                for t in llm_candidates:
                    tender_key = t.get("tender_key")
                    if tender_key:
                        store.update_tender_payload(tender_key=str(tender_key), payload=t)
        except Exception:
            logger.exception("LLM enrichment failed; continuing without LLM output.")

    attachments_enabled = bool(args.attachments)
    settings.attachments.enabled = attachments_enabled
    attachment_candidates: list[dict] = []
    if attachments_enabled:
        merged: dict[str, dict] = {}
        for t in flagged:
            key = str(t.get("tender_key") or t.get("uid") or "")
            if not key:
                key = str(id(t))
            merged[key] = t
        for t in watchlist_llm:
            key = str(t.get("tender_key") or t.get("uid") or "")
            if not key:
                key = str(id(t))
            merged.setdefault(key, t)
        attachment_candidates = list(merged.values())

    if attachment_candidates and attachments_enabled:
        try:
            att_stats = enrich_tenders_with_attachments(
                attachment_candidates,
                config=settings.attachments,
                store=store,
                cache_dir=paths.cache_dir,
                logger=logger,
            )
            logger.info(
                "Attachments: total=%s processed=%s skipped=%s downloaded=%s cached=%s llm_req=%s llm_cached=%s llm_skipped=%s ocr_used=%s empty=%s bytes=%s chars=%s",
                att_stats.get("attachments_total"),
                att_stats.get("attachments_processed"),
                att_stats.get("attachments_skipped"),
                att_stats.get("downloaded"),
                att_stats.get("cached"),
                att_stats.get("llm_requested"),
                att_stats.get("llm_cached"),
                att_stats.get("llm_skipped"),
                att_stats.get("ocr_used"),
                att_stats.get("empty_text"),
                att_stats.get("bytes"),
                att_stats.get("chars"),
            )
        except Exception:
            logger.exception("Attachment enrichment failed; continuing without attachment output.")

    for t in attachment_candidates:
        merge_attachment_requirements(t)
        if run_id is not None and not args.dry_run:
            tender_key = t.get("tender_key")
            if tender_key:
                store.update_tender_payload(tender_key=str(tender_key), payload=t)

    csv_path = None
    if args.export_csv is not None:
        csv_path = (
            _default_export_path(paths.exports_dir, tz_name)
            if args.export_csv == "__default__"
            else Path(args.export_csv)
        )
        try:
            _export_csv(flagged, csv_path)
            logger.info("Exported CSV to %s", csv_path)
        except Exception:
            logger.exception("CSV export failed")

    near_miss_path = None
    if args.export_near_miss is not None:
        near_miss_path = (
            _default_near_miss_path(paths.exports_dir, tz_name)
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
            _default_report_path(paths.exports_dir, tz_name)
            if args.export_report == "__default__"
            else Path(args.export_report)
        )
        try:
            totals = {
                "new": new_count,
                "updated": updated_count,
                "flagged": len(flagged),
                "rejected": len(rejected),
            }
            _export_reject_report(reasons, totals, report_path)
            logger.info("Exported reject report to %s", report_path)
        except Exception:
            logger.exception("Reject report export failed")

    flagged_new = [t for t in flagged_top if "new" in (t.get("_events") or [])]
    flagged_updated = [t for t in flagged_top if "new" not in (t.get("_events") or [])]
    upcoming = _upcoming_closes(flagged_top, days=7, max_items=15)
    attachment_with_reqs = sum(
        1 for t in flagged if (t.get("attachment_requirements") or [])
    )
    attachment_coverage = {
        "with_requirements": attachment_with_reqs,
        "total": len(flagged),
    }

    if args.dry_run:
        html_path = paths.exports_dir / f"digest_{_now_local(tz_name):%Y%m%d_%H%M%S}.html"
        html = render_digest_html(
            flagged_new,
            flagged_updated,
            all_flagged=flagged,
            csv_path=csv_path,
            watchlist=watchlist,
            upcoming=upcoming,
            attachment_coverage=attachment_coverage,
        )
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html, encoding="utf-8")
        logger.info("Dry-run: wrote HTML to %s", html_path)
    elif args.bootstrap:
        logger.info("Bootstrap mode: skipping email send.")
    else:
        if flagged_top:
            try:
                send_digest_email(
                    flagged_new,
                    flagged_updated,
                    all_flagged=flagged,
                    csv_path=csv_path,
                    watchlist=watchlist,
                    upcoming=upcoming,
                    attachment_coverage=attachment_coverage,
                    notifications=settings.notifications,
                )
                logger.info("Email sent: %s tenders", len(flagged_top))
            except Exception:
                logger.exception("Email send failed")
                if run_id is not None:
                    store.finish_run(
                        run_id,
                        finished_at=now_utc_iso(),
                        status="error",
                        error="email_failed",
                        new_count=new_count,
                        updated_count=updated_count,
                        flagged_count=len(flagged),
                        rejected_count=len(rejected),
                        watchlist_count=len(watchlist),
                    )
                store.close()
                return 1
        else:
            logger.info("No flagged tenders to send.")

    if args.dry_run:
        logger.info("Dry-run: state not updated.")
        store.close()
        return 0

    if args.bootstrap and last_run_iso is not None:
        logger.info("Bootstrap requested but state already has last_run_iso; updating anyway.")

    if run_id is not None:
        store.finish_run(
            run_id,
            finished_at=now_utc_iso(),
            status="ok",
            new_count=new_count,
            updated_count=updated_count,
            flagged_count=len(flagged),
            rejected_count=len(rejected),
            watchlist_count=len(watchlist),
        )

    store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
