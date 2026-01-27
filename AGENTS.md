# v0.5 Implementation Tasks

## Structured pursuit brief schema
- [x] Replace "relevance + bullets" with a structured pursuit brief JSON.
- [x] Define fields: `relevance`, `opportunity_category`, `SAP_solution_fit`, `key_requirements`, `stakeholder_entities`, `timeline`, `competitive_signals`, `recommended_next_step`.

## Enrichment workflow
- [x] Update LLM prompt to produce the structured brief.
- [x] Store the brief alongside tender data for downstream use.
- [x] Add a concise, human-readable summary for email/Slack output.

## Cost controls
- [x] Enrich only high-confidence items plus watchlist above threshold.
- [x] Add daily cap and per-run limits.
- [x] Cache by `tender_key` + amendment number.

## Acceptance checks
- [ ] Each item reads like a mini pursuit memo (not a generic summary).

# v0.4 Implementation Tasks

## Multi-stage filtering pipeline
- [x] Define stages: rules/meta -> semantic ranking -> LLM on top slice.
- [x] Wire stage gating so only top-N/borderline reach LLM.

## Embeddings-based scoring
- [x] Create an intent library of SAP/ERP buckets (ERP modernization, procurement, HR/payroll, integration/data, AMS).
- [x] Compute embeddings for tender text and bucket prototypes.
- [x] Score by max similarity + weighted evidence (UNSPSC IT, ERP hits, transformation signals).

## Explainability
- [x] Persist top bucket hit and evidence signals for each flagged tender.
- [x] Add a human-readable rationale to outputs (rules + semantic bucket hit).

## Acceptance checks
- [ ] Daily output remains comprehensive without missing relevant tenders; high-confidence + watchlist structure still applies.
- [ ] Each flag is explainable (rules + semantic bucket hit).

# v0.3 Implementation Tasks

## Carryover from v0.2 (uncompleted)
- [ ] Fresh machine: `pip install -e .` then `sap-tender-digest --dry-run` succeeds.
- [ ] CI runs green.

## Data store and schema
- [x] Introduce SQLite store with tables: `runs`, `tenders`, `decisions`, `feedback`.
- [x] Define migrations or a bootstrap path for initial DB creation.

## Tender identity + dedupe
- [x] Add stable `tender_key` that ignores amendment number for grouping.
- [x] Track per-tender fields: `first_seen`, `last_seen`, `last_amendment_seen`, `last_close_date_seen`.

## Event detection
- [x] Emit explicit events: New tender, Amendment posted, Close date changed, New attachment.
- [x] Ensure repeated runs only notify on material changes.

## Digest formatting
- [x] Split output into: High confidence (New), High confidence (Updated/Amended), Watchlist.
- [x] Optional section: Upcoming closes (next 7 days).

## Run modes
- [x] Keep state without wiping it; support `--since-iso` and `--bootstrap` flows.

## Completed (v0.2 foundation)
- [x] Remove `sys.path` bootstrapping and ensure the package runs via module entrypoints.
- [x] Convert to a proper installable package (editable install works).
- [x] Add console scripts: `sap-tender-digest` and `sap-tender-ui`.
- [x] Decide on dependency source of truth (`pyproject.toml` or `requirements.txt`) and normalize encoding.
- [x] Define a single config model (YAML with env overrides).
- [x] Update CLI to load config from the shared model.
- [x] Update Streamlit to read/write the same config model.
- [x] Ensure scheduler usage consumes the same config inputs.
- [x] Add `.env.example` with required and optional keys.
- [x] Document setup steps in README (fresh machine flow).
- [x] Add a PowerShell "make run" equivalent for local development.
- [x] Add GitHub Actions workflow for ruff and pytest.
- [x] Add minimal test scaffolding if missing (smoke test for `sap-tender-digest --dry-run`).
