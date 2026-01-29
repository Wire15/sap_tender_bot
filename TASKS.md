# Attachment Enrichment Improvements Tasks

Last updated: 2026-01-28

## 0) Scoping and guardrails
- [x] Confirm attachment pipeline entry points (download, extract, summarize, persist, digest). Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/daily_digest.py`
- [x] Identify current size/cost caps and where to extend them safely for new formats. Refs: `src/sap_tender_bot/config.py`, `config.yaml`, `src/sap_tender_bot/pipeline/attachments.py`
- [x] Add feature flags for: spreadsheets, chunked retrieval, structured summaries, language handling. Refs: `src/sap_tender_bot/config.py`, `config.yaml`, `src/sap_tender_bot/daily_digest.py`
- [x] Define per-run and per-tender attachment processing caps for new stages. Refs: `src/sap_tender_bot/config.py`, `config.yaml`, `src/sap_tender_bot/pipeline/attachments.py`
- [x] Document new config knobs (caps, chunk sizes, thresholds, language routing). Refs: `README.md`, `.env.example`, `config.yaml`

## 1) Quick wins

### 1.1 Spreadsheet support (.xlsx/.xls)
- [x] Add `.xlsx`/`.xls` to `SUPPORTED_MIME` and `SUPPORTED_EXT`. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Implement spreadsheet extraction via pandas + openpyxl. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `pyproject.toml`
- [x] Enforce max bytes for spreadsheet ingest. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Enforce row and column limits to prevent runaway memory. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/config.py`, `config.yaml`
- [x] Convert sheets to text with stable, csv-like row formatting. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Ensure multi-sheet handling (sheet headers + per-sheet truncation). Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Add extraction tests for small and large spreadsheets. Refs: `tests/test_attachments.py`

### 1.2 Expand heading detection
- [x] Add French headings for procurement markers (submission, evaluation, requirements). Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Add procurement-specific headings (e.g., "submission requirements", "evaluation criteria"). Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Reduce "sections=0" cases via fallback heading heuristics. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Add unit tests for heading detection with English + French samples. Refs: `tests/test_attachments.py`

### 1.3 Memo formatting in digest
- [x] Add attachment name + provenance hint to memo lines. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/daily_digest.py`, `src/sap_tender_bot/pipeline/notify_email.py`
- [x] Add "Attachment signal" line (counts + date/risk found flags). Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/daily_digest.py`, `src/sap_tender_bot/pipeline/notify_email.py`
- [x] Ensure digest HTML/email formatting remains stable with new memo lines. Refs: `src/sap_tender_bot/pipeline/notify_email.py`, `src/sap_tender_bot/daily_digest.py`

## 2) Targeted extraction upgrade

### 2.1 Chunked retrieval for attachments
- [x] Define chunking strategy (page-based vs heading-based) with provenance. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Split extracted text into chunks with attachment/page/section metadata. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Embed chunks and compute similarity to requirement/date/evaluation prompts. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/pipeline/semantic.py`, `src/sap_tender_bot/config.py`
- [x] Select top-K relevant chunks per attachment. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Cap chunks per attachment and per tender. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/config.py`, `config.yaml`
- [x] Add keyword boosting for requirement/date/evaluation signals. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Add per-attachment cost guardrails (token and chunk limits). Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/config.py`
- [x] Wire retrieval output into attachment LLM input. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Add traceable provenance for each selected chunk. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Add tests for chunking stability and selection behavior. Refs: `tests/test_attachments.py`

### 2.2 Table extraction for DOCX
- [x] Extend DOCX extractor to include table rows. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Normalize table output to text with row/column formatting. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Ensure table extraction respects size and row caps. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/config.py`
- [x] Add regression tests for DOCX tables. Refs: `tests/test_attachments.py`

### 2.3 Quality filters
- [x] Add boilerplate de-duplication for repeated vendor/legal text. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Require minimum evidence spans for requirement/date extraction. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Suppress duplicate items across chunks/attachments. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Apply low-signal thresholds to drop weak items. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Add evidence-quality checks to reduce ungrounded claims. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Add tests for de-dup and low-signal suppression. Refs: `tests/test_attachments.py`

## 3) Structured attachment summaries

### 3.1 Schema and provenance
- [x] Define summary schema: submission, evaluation, scope, deliverables, schedule, risks, compliance. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Include citation fields: attachment_id, page, section_heading, offsets. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Add schema validation and defaults for missing fields. Refs: `src/sap_tender_bot/pipeline/attachments.py`

### 3.2 Persistence
- [x] Store summary JSON in `attachment_cache.summary_json`. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/store.py`
- [x] Persist source sections/citations alongside summary. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/store.py`
- [x] Keep backward compatibility with older summaries. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `pages/06_Attachments.py`, `src/sap_tender_bot/pipeline/notify_email.py`
- [x] Add migration or bootstrap updates if schema changes require it. Refs: `src/sap_tender_bot/store.py`

### 3.3 Digest + UI updates
- [x] Display structured items with provenance (attachment + section/page). Refs: `src/sap_tender_bot/pipeline/notify_email.py`, `src/sap_tender_bot/daily_digest.py`, `pages/06_Attachments.py`
- [x] Render missing/approximate provenance clearly. Refs: `src/sap_tender_bot/pipeline/notify_email.py`, `pages/06_Attachments.py`
- [x] Ensure UI handles empty/partial summaries without errors. Refs: `pages/06_Attachments.py`
- [x] Update HTML/email templates to include structured summary sections. Refs: `src/sap_tender_bot/pipeline/notify_email.py`, `src/sap_tender_bot/daily_digest.py`

## 4) Digest enrichment

### 4.1 Coverage metrics
- [x] Track processed/skipped/unsupported per tender. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/daily_digest.py`
- [x] Distinguish "summary exists" vs "requirements extracted". Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/daily_digest.py`
- [x] Add failure taxonomy: unsupported, extraction failed, low-signal, LLM gated. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/daily_digest.py`
- [x] Persist metrics for reporting and debugging. Refs: `src/sap_tender_bot/store.py`, `src/sap_tender_bot/daily_digest.py`

### 4.2 Better attachment notes
- [x] Show attachment name + top 1-2 requirements/dates. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/pipeline/notify_email.py`, `src/sap_tender_bot/daily_digest.py`
- [x] Add "why it matters" line when SAP/ERP signals found. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/pipeline/notify_email.py`
- [x] Keep digest concise; add truncation for long notes. Refs: `src/sap_tender_bot/pipeline/notify_email.py`, `src/sap_tender_bot/daily_digest.py`

### 4.3 Language handling
- [x] Detect non-English attachments (start with French). Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Route to translated prompt or language-specific headings. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/config.py`
- [x] Keep English fallback if detection uncertain. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Add tests for language routing behavior. Refs: `tests/test_attachments.py`

## 5) Cache and invalidation
- [x] Fingerprint attachments by hash + filename + size + last_modified. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/store.py`
- [x] Invalidate cache on amended attachments. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/daily_digest.py`
- [x] Ensure attachment LLM cache keys include amendment-sensitive fingerprint. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Add tests for cache hits/misses on amendment changes. Refs: `tests/test_attachments.py`

## 6) Safety and redaction
- [x] Add email/phone redaction step before LLM or digest output. Refs: `src/sap_tender_bot/pipeline/attachments.py`, `src/sap_tender_bot/pipeline/notify_email.py`
- [x] Ensure redaction does not remove dates or evaluation criteria. Refs: `src/sap_tender_bot/pipeline/attachments.py`
- [x] Add tests for redaction behavior on sample text. Refs: `tests/test_attachments.py`

## 7) Measurement and rollout
- [x] Define baseline window for coverage metrics (last 30 days). Refs: `README.md`, `src/sap_tender_bot/daily_digest.py`
- [x] Implement coverage calculation for attachment-derived requirements/dates/evaluation. Refs: `src/sap_tender_bot/daily_digest.py`, `src/sap_tender_bot/pipeline/attachments.py`
- [x] Track weekly deltas and sample sizes. Refs: `src/sap_tender_bot/daily_digest.py`, `src/sap_tender_bot/store.py`
- [x] Track false-positive rate for requirements/dates extraction. Refs: `src/sap_tender_bot/daily_digest.py`
- [x] Add reporting output or log summaries for coverage stats. Refs: `src/sap_tender_bot/daily_digest.py`, `ui_common.py`

## 8) Testing and validation
- [x] Add fixture attachments: PDF/DOCX/XLSX with known requirements/dates/evaluation. Refs: `tests/test_attachments.py`
- [x] Create golden summaries (10-20) with provenance checks. Refs: `tests/test_attachments.py`
- [x] Add perf budget checks (per-run time, per-tender attachment caps). Refs: `tests/test_attachments.py`, `src/sap_tender_bot/pipeline/attachments.py`
- [x] Manual evidence-quality spot checks (20-30 summaries). Refs: `pages/06_Attachments.py`
- [x] Update CI to run new attachment tests if needed. Refs: `.github/workflows/ci.yml`

## 9) Rollback and guardrails
- [x] Ensure per-extractor feature flags can disable spreadsheets/tables/chunking. Refs: `src/sap_tender_bot/config.py`, `config.yaml`
- [x] Add kill-switch for attachment LLM when costs spike. Refs: `src/sap_tender_bot/config.py`, `src/sap_tender_bot/pipeline/attachments.py`
- [x] Document rollback steps in README or ops notes. Refs: `README.md`

## 10) Acceptance criteria validation
- [x] Verify coverage improves from 57% to 75%+ on flagged tenders. Refs: `src/sap_tender_bot/daily_digest.py`
- [x] Validate 60%+ of summaries include date or evaluation/submission item. Refs: `src/sap_tender_bot/daily_digest.py`, `pages/06_Attachments.py`
- [x] Confirm digest shows provenance for attachment-derived requirements. Refs: `src/sap_tender_bot/pipeline/notify_email.py`, `pages/06_Attachments.py`
- [x] Confirm run duration regression stays within +20%. Refs: `src/sap_tender_bot/daily_digest.py`
- [x] Confirm false positives remain under 10%. Refs: `src/sap_tender_bot/daily_digest.py`
