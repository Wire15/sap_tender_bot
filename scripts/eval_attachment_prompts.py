from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sap_tender_bot.config import load_config
from sap_tender_bot.pipeline import attachments as att
from sap_tender_bot.store import TenderStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _summarize(items: list[dict]) -> dict:
    totals = {"count": len(items), "with_signal": 0, "avg_reqs": 0.0, "avg_dates": 0.0}
    if not items:
        return totals
    reqs = [int(i.get("req_count", 0)) for i in items]
    dates = [int(i.get("date_count", 0)) for i in items]
    totals["with_signal"] = sum(1 for i in items if i.get("has_signal"))
    totals["avg_reqs"] = round(sum(reqs) / len(reqs), 2)
    totals["avg_dates"] = round(sum(dates) / len(dates), 2)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate attachment prompt variants.")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--min-chars", type=int, default=800)
    parser.add_argument("--prompt-a", default="v1")
    parser.add_argument("--prompt-b", default="v2")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    config = load_config()
    if not config.attachments.llm_api_key:
        raise SystemExit("Attachments LLM API key missing. Set SAP_TENDER_ATTACHMENTS_LLM_API_KEY.")

    store = TenderStore(config.paths.db_path).open()
    try:
        samples = store.fetch_attachment_cache_samples(
            limit=args.limit, min_text_chars=args.min_chars
        )
    finally:
        store.close()

    if not samples:
        raise SystemExit("No attachment cache samples found.")

    results = []
    for entry in samples:
        text_path = entry.get("text_path")
        if not text_path or not Path(text_path).exists():
            continue
        try:
            text, _meta = att._extract_text(
                Path(text_path),
                mime_type=str(entry.get("mime_type") or ""),
                config=config.attachments,
            )
        except Exception:
            continue
        if not text:
            continue
        excerpt, _sections = att._sectionize(
            text,
            max_chars=config.attachments.section_char_budget,
            max_sections=config.attachments.max_sections,
        )

        row = {
            "fingerprint": entry.get("fingerprint"),
            "url": entry.get("url"),
            "attachment_name": att._attachment_name(str(entry.get("url") or "")),
            "text_chars": _safe_int(entry.get("text_chars")),
            "created_at": entry.get("fetched_at"),
            "excerpt_chars": len(excerpt),
        }
        for label in ("a", "b"):
            version = getattr(args, f"prompt_{label}")
            prompt = att._build_prompt(
                row["attachment_name"],
                excerpt,
                language=str(config.attachments.language_default or "en"),
                prompt_version=str(version),
            )
            raw = att._cohere_chat(prompt, config.attachments)
            normalized = att._normalize_summary(raw, evidence_text=excerpt)
            has_signal = att._summary_has_signal(normalized, None)
            row[f"{label}_version"] = version
            row[f"{label}_raw"] = raw
            row[f"{label}_summary"] = normalized
            row[f"{label}_req_count"] = len(normalized.get("requirements") or [])
            row[f"{label}_date_count"] = len(normalized.get("dates") or [])
            row[f"{label}_risk_count"] = len(normalized.get("key_risks") or [])
            row[f"{label}_has_signal"] = has_signal
        results.append(row)

    summary_a = _summarize(
        [
            {
                "req_count": r.get("a_req_count"),
                "date_count": r.get("a_date_count"),
                "has_signal": r.get("a_has_signal"),
            }
            for r in results
        ]
    )
    summary_b = _summarize(
        [
            {
                "req_count": r.get("b_req_count"),
                "date_count": r.get("b_date_count"),
                "has_signal": r.get("b_has_signal"),
            }
            for r in results
        ]
    )

    payload = {
        "generated_at": _now_iso(),
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "summary_a": summary_a,
        "summary_b": summary_b,
        "results": results,
    }

    exports_dir = config.paths.exports_dir
    exports_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else exports_dir / "attachment_prompt_eval.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("Summary A:", summary_a)
    print("Summary B:", summary_b)


if __name__ == "__main__":
    main()
