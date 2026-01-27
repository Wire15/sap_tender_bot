# Attachment Enrichment Implementation Plan

## Phase 0: Baseline + Design (0.5–1 day)
- Define attachment schema additions (`attachment_summaries`, `attachment_requirements`, provenance fields).
- Decide fingerprinting rules (URL normalization, ETag/Last-Modified fallback, content hash).
- Define limits (max size, per-run caps, concurrency, timeouts).
- Deliverable: short design note + config defaults.

## Phase 1: Data + Cache Plumbing (1 day)
- Add attachment cache store and migration/bootstrapping.
- Extend tender payload to persist summary + requirements + provenance.
- Add config wiring for caps and concurrency.
- Deliverable: DB schema changes + config model updates.
- Verify: migration bootstraps; cache entries persisted/queried.

## Phase 2: Fetch + Extraction (1.5–2 days)
- Implement downloader with caching, timeouts, size limits.
- Add retry/backoff + skip reason logging.
- Implement PDF/DOCX extraction; optional OCR fallback.
- Normalize extracted text (headers/footers, whitespace).
- Deliverable: attachment text extraction module.
- Verify: unit tests on sample PDF/DOCX; OCR path optional.

## Phase 3: Section Targeting (0.5–1 day)
- Implement heading detection heuristics + synonyms.
- Extract top N sections within char budget.
- Log extraction stats per attachment.
- Deliverable: section targeting utility.
- Verify: unit tests; targeted sections <= char budget.

## Phase 4: LLM Summarization (Command R) (1 day)
- Add LLM call for attachment summary JSON.
- Cache results per attachment fingerprint.
- Enforce per-run LLM caps + concurrency.
- Deliverable: summarizer module + cache integration.
- Verify: JSON schema validation; empty `requirements` allowed but must include `attachments_used`.

## Phase 5: Merge + Output (1 day)
- Merge attachment requirements into tender `key_requirements` with provenance.
- Prefer attachment-derived requirements when present.
- Add `attachment_summary_used` flag.
- Update digest output and coverage metrics.
- Deliverable: digest changes + metrics counters.
- Verify: sample run shows “Req (doc)” and coverage %.

## Phase 6: Safety + Privacy (0.5 day)
- Add MIME/content-type validation and “no attachments” mode.
- Add optional PII scrub pass before LLM.
- Deliverable: safety guardrails.
- Verify: unsupported types are skipped with reason.

## Phase 7: Tests + Acceptance (1 day)
- Unit tests: extraction + section targeting.
- Integration test: caching + repeated runs.
- Acceptance checks: “mini requirement memos” + provenance traceable.
- Deliverable: test coverage and acceptance checklist.

## Suggested Order
1) Phase 1 (schema + cache)
2) Phase 2 (fetch/extract)
3) Phase 3 (section targeting)
4) Phase 4 (LLM + caching)
5) Phase 5 (merge/output)
6) Phase 6 (safety/privacy)
7) Phase 7 (tests/acceptance)
