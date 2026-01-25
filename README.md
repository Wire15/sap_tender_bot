# SAP Tender Digest Bot

A daily digest bot that pulls open CanadaBuys tender notices, filters for SAP/ERP relevance, and (optionally) enriches with Cohere Command R before emailing a summary.

## Current status (as of 2026-01-25)
- **Ingest:** CanadaBuys “Open tender notices” via CKAN; normalized fields include EN/FR titles/descriptions, dates, orgs, UNSPSC/GSIN, and URLs.
- **Delta semantics:** Uses `updated_at = max(publicationDate, amendmentDate)` in the connector; daily digest compares to `data/state.json`.
- **Filtering:** Heavily tuned to remove supply arrangements/standing offers, staffing, goods/hardware, non‑IT services, and portal boilerplate. Accent/apostrophe normalization is in place. Explainability is stored on each tender (`_hits`).
- **LLM:** Cohere Command R integration with retries and cache (`data/cache/llm/`).
- **Email:** SMTP HTML digest with optional CSV attachment. Dry‑run writes HTML to `data/exports/`.
- **Logging:** File logging at `logs/run_YYYYMMDD.log`.
- **Result snapshot:** With the current (tight) filters, the latest dry runs found **0** relevant open tenders. This is expected when the dataset has no SAP/ERP opportunities.

## Repo layout
```
src/
  connectors/
    canadabuys_tender_notices.py   # CKAN download + normalization
    open_canada_ckan.py            # CKAN client
  pipeline/
    ingest.py                      # connector wrapper
    filter.py                      # scoring + filters
    llm.py                         # Cohere Command R integration + cache
    notify_email.py                # SMTP HTML email
  daily_digest.py                  # main runner
sap_tender_bot/
  daily_digest.py                  # CLI wrapper for module execution
  __init__.py

config.py                          # repo-root paths
```

## Usage
Dry run (no email, no state update):
```powershell
python -m sap_tender_bot.daily_digest --dry-run --export-csv --no-llm
```

Production run:
```powershell
python -m sap_tender_bot.daily_digest --export-csv
```

Bootstrap (seed state, no email):
```powershell
python -m sap_tender_bot.daily_digest --bootstrap
```

Backfill:
```powershell
python -m sap_tender_bot.daily_digest --since-iso 2025-12-01T00:00:00Z --export-csv
```

## Configuration (.env)
Required for email:
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `EMAIL_TO`, `EMAIL_FROM`

Optional for LLM:
- `COHERE_API_KEY`, `COHERE_MODEL`, `COHERE_API_BASE`, `COHERE_TIMEOUT`

## State and outputs
- State: `data/state.json` (`last_run_iso`, `seen_ids`)
- Exports: `data/exports/flagged_YYYYMMDD.csv`, `data/exports/digest_YYYYMMDD_HHMMSS.html`
- Logs: `logs/run_YYYYMMDD.log`

## Scheduling (Windows Task Scheduler)
```powershell
schtasks /Create /SC DAILY /ST 08:35 /TN "SAP Tender Digest" /TR "cmd /c \"cd /d C:\\dev\\sap-tender-bot && .venv\\Scripts\\python.exe -m sap_tender_bot.daily_digest --export-csv\""
```

## Suggested improvements
Short‑term:
- Add a **near‑miss export** (items rejected by a single rule) for ongoing filter tuning.
- Add **unit tests** for new negative filters and portal‑boilerplate handling.
- Add a **report section** in the digest showing top rejection reasons.

Medium‑term:
- Cache the CanadaBuys ZIP/CSV with an ETag/Last‑Modified check to avoid re‑download.
- Persist LLM outputs to include structured summaries in the HTML digest.
- Add a **watchlist mode**: send a second section for “ERP‑adjacent” items below the main threshold.

Long‑term:
- Add additional sources beyond CanadaBuys (provincial, municipal, Crown corp feeds).
- Add a small admin UI or notebook for reviewing filter hits and training a classifier.
- Add a dedupe key across sources using normalized title + org + close date.
