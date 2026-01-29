# Attachment Enrichment Improvements Plan

Last updated: 2026-01-28

## Goals
- Increase extraction of vital tender data from attachments (requirements, dates, evaluation, submission).
- Improve digest usefulness with clearer, traceable attachment insights.
- Keep costs predictable; avoid regressions to current pipeline.

## Non-goals
- Full bid drafting or vendor evaluation.
- Replacing existing tender summarization logic outside attachments.

## Scope
- Attachments pipeline (download, extract, summarize, store).
- Digest output (HTML/email and UI).
- Minimal, incremental DB changes if needed for traceability.
- Defer OCR/Q&A to a later version; focus this plan on attachment extraction quality.

## Workstreams, Tasks, and Estimates

### 1) Quick wins 
1. Add spreadsheet support for `.xlsx`/`.xls` attachments 
   - Parse tables to text using pandas + openpyxl (fallback to csv-like rows).
   - Enforce max bytes and row/column limits to prevent runaway memory.
   - Extend `SUPPORTED_MIME`/`SUPPORTED_EXT`.
2. Expand heading detection 
   - Add French headings and procurement-specific markers (submission, evaluation).
   - Reduce "sections=0" cases to improve relevance of the excerpt.
3. Improve memo formatting in digest
   - Include attachment name and short provenance hint.
   - Add "Attachment signal" line: counts + whether dates/risks found.

### 2) Targeted extraction upgrade 
1. Chunked retrieval for attachments 
   - Split extracted text into sections/pages.
   - Define chunking boundaries (page-based vs heading-based) for consistent provenance.
   - Embed chunks and select top-K relevant chunks for requirements, dates, evaluation.
   - Feed selected chunks into the attachment LLM.
   - Cap chunks per attachment and per tender for cost control.
   - Use retrieval signals: semantic similarity + requirement/date/evaluation keywords.
2. Table extraction for PDF/DOCX 
   - DOCX: include table rows in extracted text.
   - PDF table extraction removed for now; revisit only if we see persistent misses in table-heavy PDFs.
3. Quality filters 
   - De-dup boilerplate, enforce requirement grounding, drop low-signal items.
   - Apply minimum evidence span, duplicate suppression, and low-signal thresholds.
   - Add evidence quality checks to reduce ungrounded or speculative items.

### 3) Structured attachment summaries 
1. New schema for attachment summaries 
   - Fields: submission, evaluation, scope, deliverables, schedule, risks, compliance.
   - Require citations to section/page where possible.
   - Define a provenance contract (attachment_id, page, section_heading, offsets).
2. Persist citations and source sections 
   - Store in `attachment_cache.summary_json`.
   - Add minimal metadata in tender payload for digest/UI.
3. Update digest + UI to display structured summary 
   - Show key items with provenance (attachment + section/page).

### 4) Digest enrichment 
1. Coverage metrics improvements 
   - Track processed/skipped/unsupported per tender.
   - Distinguish "summary exists" vs "requirements extracted".
   - Add a failure taxonomy: unsupported, extraction failed, low-signal, LLM gated.
2. Better attachment notes 
   - Show attachment name and top 1-2 key requirements + dates.
   - Add a short "why it matters" line when SAP/ERP signals found.
3. Language handling 
   - Detect non-English attachments and route to translated prompt or adjusted headings.
   - Start with French coverage; keep fallback to English extraction.

### 5) Deferred (future version)
- OCR for scanned PDFs and Q&A foundation (out of scope for this plan).

## Testing Plan
- Fixture attachments: PDF/DOCX/XLSX with known requirements/dates/evaluation.
- Golden summaries for 10-20 attachments with provenance checks.
- Basic perf budget: per-run time and per-tender attachment processing caps.
- Evidence-quality spot checks (manual review of 20-30 summaries).

## Measurement Plan
- Define baseline window (e.g., last 30 days) for current 57% coverage.
- Coverage definition: % flagged tenders with any attachment-derived requirements/dates/evaluation.
- Report sample size and change delta per week during rollout.
- Track false-positive rate for requirement/date extraction (target <10%).

## Deliverables
- Enhanced attachments extraction with spreadsheet + tables.
- Chunked retrieval to improve requirement accuracy.
- Structured attachment summaries with provenance.
- Digest improvements that surface attachment insights clearly.

## Risks and Mitigations
- Cost increase from more LLM usage
  - Mitigation: chunk selection, per-run caps, daily cap enforcement, and token guardrails.
- Slower runs for large attachments
  - Mitigation: size caps, chunk limits, and concurrency.
- OCR errors on low-quality scans
  - Mitigation: deferred to later version; revisit with clear gating.
- Incorrect provenance (section/page not reliable)
  - Mitigation: prefer stable anchors; if uncertain, mark provenance as approximate.
- Cache staleness on amended attachments
  - Mitigation: fingerprint by attachment (hash + filename + size + last_modified) and invalidate on change.
- PII leakage in excerpts
  - Mitigation: redact emails/phone numbers before LLM or digest output when required.

## Rollback Plan
- Feature flags per extractor (spreadsheets, tables, OCR, chunking).
- Kill-switch for attachment LLM when costs spike or outputs degrade.

## Schema and Compatibility Notes
- Document any new fields in `attachment_cache.summary_json`.
- Ensure older summaries remain readable in digest/UI without errors.

## Acceptance Criteria
- Attachment coverage improves from 57% to 75%+ on flagged tenders.
- At least 60% of attachment summaries include a date or evaluation/submission item.
- Digest shows provenance for attachment-derived requirements.
- No regression in run duration beyond +20% for typical daily runs.
- False positives (incorrect requirements/dates) stay under 10%.
