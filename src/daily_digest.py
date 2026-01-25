import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .pipeline.filter import score_and_filter
from .pipeline.ingest import ingest
from .pipeline.notify_email import send_digest_email

STATE_PATH = Path("data/state.json")


def load_state():
    if STATE_PATH.exists():
        content = STATE_PATH.read_text(encoding="utf-8").strip()
        if content:
            return json.loads(content)
    return {"last_run_iso": None, "seen_ids": []}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don’t send email; print results.")
    args = parser.parse_args()

    state = load_state()
    last_run_iso = state.get("last_run_iso")
    seen = set(state.get("seen_ids", []))

    tenders = ingest(since_iso=last_run_iso)

    new_items = []
    for t in tenders:
        uid = t.get("uid")
        if uid and uid not in seen:
            new_items.append(t)

    flagged, reasons = score_and_filter(tenders)
    print("Rejection reasons (top):")
    for k, v in reasons.most_common(15):
        print(f"  {k}: {v}")


    if args.dry_run:
        print(f"Last run: {last_run_iso}")
        print(f"New: {len(new_items)} | Flagged: {len(flagged)}")
        for t in flagged[:25]:
            print(f"- [{t.get('score',0)}] {t.get('title')} | closes: {t.get('close_date')} | {t.get('url')}")
    else:
        if flagged:
            send_digest_email(flagged)

    # mark all fetched items as seen so you don’t re-alert tomorrow
    for t in new_items:
        if t.get("uid"):
            seen.add(t["uid"])

    state["seen_ids"] = list(seen)
    state["last_run_iso"] = now_utc_iso()
    save_state(state)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
