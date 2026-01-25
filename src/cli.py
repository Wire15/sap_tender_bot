import argparse

from pipeline.filter import score_and_filter
from pipeline.ingest import ingest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't send emails")
    args = parser.parse_args()

    tenders = ingest()
    picked, _ = score_and_filter(tenders)

    print(f"Ingested: {len(tenders)} | Selected: {len(picked)}")
    if args.dry_run:
        for t in picked[:10]:
            print(f"- {t['title']} ({t.get('close_date', 'n/a')}) -> {t['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
