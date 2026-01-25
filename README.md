# SAP Tender Digest Bot

A daily digest bot that pulls open CanadaBuys tender notices, filters for SAP/ERP relevance, and (optionally) enriches with Cohere Command R before emailing a summary.

## Current status (as of 2026-01-25)
- **Ingest:** CanadaBuys “Open tender notices” via CKAN; normalized fields include EN/FR titles/descriptions, dates, orgs, UNSPSC/GSIN, and URLs.
- **Caching:** Conditional GET with ETag/Last‑Modified to avoid re‑downloading unchanged CKAN resources.
- **Delta semantics:** Uses `updated_at = max(publicationDate, amendmentDate)` in the connector; daily digest compares to `data/state.json`.
- **Filtering:** Heavily tuned to remove supply arrangements/standing offers, staffing, goods/hardware, non‑IT services, and portal boilerplate. Accent/apostrophe normalization is in place. Explainability is stored on each tender (`_hits`).
- **LLM:** Cohere Command R integration with retries and cache (`data/cache/llm/`).
- **Email:** SMTP HTML digest with optional CSV attachment. Dry‑run writes HTML to `data/exports/`.
- **Watchlist:** Optional near‑miss section in the email for single‑reason rejects.
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

Dev exports (near-miss + reject report). Near-miss includes only single-reason rejects:
```powershell
python -m sap_tender_bot.daily_digest --dry-run --export-csv --export-near-miss --export-report --no-llm
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

## Admin UI (Streamlit)
Launch the multi-page admin UI:
```powershell
streamlit run streamlit_app.py
```

Notes:
- The UI reads from `data/exports/`, `data/state.json`, and `logs/`.
- Run a dry run export first if the UI is empty:
  ```powershell
  python -m sap_tender_bot.daily_digest --dry-run --export-csv --export-near-miss --export-report --no-llm
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

## v1.0 release plan

Goal: ship a reliable daily digest with stable interfaces, reproducible runs, and clear operator ergonomics.

Release gates:
- Daily run can complete without manual intervention for 14 consecutive days.
- Zero crashes across dry run, production run, bootstrap, and backfill paths.
- Docs cover setup, operations, and troubleshooting for a new operator.

Milestones:
1. Reliability and data hygiene
   - Add deterministic dedupe key across sources (normalized title + org + close date).
   - Enforce schema validation on normalized tenders (required fields, date formats).
   - Persist state/version metadata for reproducibility and audits.

2. Digest quality
   - Persist LLM outputs and include structured summaries in HTML digest.
   - Add precision/recall tracking report for filter changes.
   - Expand near-miss explanations to help tune rules.

3. Operations and observability
   - Add health summary to the email (counts, duration, cache hits, errors).
   - Add a minimal admin UI view for recent runs and flagged items.
   - Add alerting on failures (email to ops on exceptions).

4. Source expansion
   - Add at least one additional public source (provincial/municipal/Crown corp).
   - Add per-source flags for enable/disable and source tagging in outputs.

5. Release hardening
   - Add automated tests for connectors, filters, and state transitions.
   - Add CI job for linting and tests.
   - Add a changelog and versioning policy (SemVer).
