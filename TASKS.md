# Attachment Enrichment Feature (Command R)

## Goals
- Extract requirements and key info from tender attachments (PDF/DOCX) using a cost-controlled pipeline.
- Keep Command A for the final structured brief; use Command R for attachment summarization.
- Cache outputs to avoid reprocessing unchanged attachments.

## Milestone 1: Data + Cache Plumbing
- [x] Add a cache store keyed by attachment fingerprint (URL + ETag/Last-Modified; fallback to content hash).
- [x] Define normalization rules for fingerprinting (URL normalization, weak ETag handling, missing headers).
- [x] Extend tender payload to persist `attachment_summaries` and `attachment_requirements` with provenance metadata.
- [x] Add config for attachment processing caps (per-run attachments, max tokens/characters).
- [x] Add per-run concurrency limits (downloads and LLM calls).

## Milestone 2: Attachment Fetch + Text Extraction
- [x] Implement attachment downloader with caching, size limits, and timeouts.
- [x] Add retry/backoff policy and skip reasons (e.g., `too_large`, `unsupported_type`, `download_failed`).
- [x] Add PDF text extraction (fallback to OCR if needed; optional).
- [x] Add DOCX text extraction.
- [x] Normalize extracted text (strip headers/footers, remove empty lines).
  - OCR fallback (optional): if extracted text is too short (e.g., <800 chars) or mostly empty pages.
  - Suggested libraries: `pypdf` or `pdfminer.six` (PDF), `python-docx` (DOCX), `pytesseract`/`ocrmypdf` (OCR).

## Milestone 3: Section Targeting (Token Control)
- [x] Implement simple heading detection to capture likely sections:
      - "Mandatory Requirements", "Scope of Work", "Statement of Work", "Evaluation Criteria".
- [x] Heuristics: case-insensitive, supports numbered headings, includes common synonyms.
- [x] Extract top N sections; enforce max character budget (e.g., 3-5k chars).
- [x] Log extraction stats per attachment (chars kept, sections found).

## Milestone 4: Attachment LLM Summarization (Command R)
- [x] Add a new LLM call for attachment summaries using Command R.
- [x] Prompt to output JSON: `requirements`, `key_risks`, `dates`, `attachments_used`.
- [x] Cache per attachment summary to avoid repeated calls.
- [x] Apply per-run LLM cap for attachment summaries.
- [x] Define acceptance checks: if no requirements found, emit empty array but include `attachments_used`.

## Milestone 5: Merge Into Tender Brief
- [x] Merge attachment requirements into tender `key_requirements` with provenance tags.
- [x] Prefer attachment-derived requirements when present; fall back to tender text.
- [x] Record `attachment_summary_used` flag for transparency.

## Milestone 6: Output + Reporting
- [x] Show attachment-derived requirements in digest (e.g., "Req (doc): ...").
- [x] Add coverage metric: `% tenders with attachment requirements`.
- [x] Add cost metrics for attachment pipeline (requests, cached, skipped).
- [x] Add observability counters: OCR used, empty extraction, bytes/chars processed.

## Milestone 7: Safety + Quality
- [x] Add file size ceiling (skip huge attachments).
- [x] Add MIME/content-type validation.
- [x] Add a "no attachments" mode flag.
- [x] Add PII redaction or basic scrub pass before LLM (optional config).

## Milestone 8: Tests
- [x] Unit tests for text extraction and section targeting.
- [x] Integration test for cached attachment summaries.

## Milestone 9: Acceptance Criteria
- [x] Attachment summaries read like mini requirement memos (not generic summaries).
- [x] Attachment-derived requirements are traceable to a source attachment and section.
