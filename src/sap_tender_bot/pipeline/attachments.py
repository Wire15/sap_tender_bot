from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import requests

from sap_tender_bot.config import AttachmentsConfig
from sap_tender_bot.store import TenderStore

SUPPORTED_MIME = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}
SUPPORTED_EXT = {".pdf", ".docx"}

HEADING_TERMS = [
    "mandatory requirements",
    "mandatory criteria",
    "minimum requirements",
    "requirements",
    "scope of work",
    "statement of work",
    "sow",
    "deliverables",
    "evaluation criteria",
    "submission instructions",
    "proposal submission",
    "key dates",
    "timeline",
    "schedule",
    "closing date",
    "security requirements",
    "certification requirements",
]

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")


@dataclass
class AttachmentResult:
    url: str
    url_norm: str
    fingerprint: str
    summary: dict
    requirements: list[dict]
    cached: bool


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    scheme = parsed.scheme.lower() or "http"
    host = (parsed.hostname or "").lower()
    port = parsed.port
    netloc = host
    if port:
        netloc = f"{host}:{port}"
    query = "&".join(
        f"{k}={v}" if v != "" else k for k, v in sorted(parse_qsl(parsed.query))
    )
    return urlunsplit((scheme, netloc, parsed.path or "", query, ""))


def _attachment_name(url: str) -> str:
    parsed = urlsplit(url)
    name = Path(parsed.path or "").name
    return name or url


def _normalize_etag(etag: Optional[str]) -> Optional[str]:
    if not etag:
        return None
    cleaned = str(etag).strip()
    cleaned = cleaned.replace("W/", "").strip()
    return cleaned.strip("\"'")


def _guess_extension(url: str, content_type: Optional[str]) -> str:
    url = (url or "").lower()
    for ext in SUPPORTED_EXT:
        if url.endswith(ext):
            return ext
    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()
        return SUPPORTED_MIME.get(mime, "")
    return ""


def _usage_path(cache_dir: Path, day: str) -> Path:
    return cache_dir / f"usage_{day}.json"


def _load_daily_usage(cache_dir: Path, day: str) -> dict:
    path = _usage_path(cache_dir, day)
    if not path.exists():
        return {"day": day, "count": 0, "updated_at": _now_utc_iso()}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"day": day, "count": 0, "updated_at": _now_utc_iso()}
    if not isinstance(data, dict):
        return {"day": day, "count": 0, "updated_at": _now_utc_iso()}
    data.setdefault("day", day)
    data.setdefault("count", 0)
    data.setdefault("updated_at", _now_utc_iso())
    return data


def _save_daily_usage(cache_dir: Path, day: str, count: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"day": day, "count": int(count), "updated_at": _now_utc_iso()}
    _usage_path(cache_dir, day).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _scrub_pii(text: str) -> str:
    if not text:
        return ""
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def _extract_json(text: str) -> dict:
    if not text:
        raise ValueError("Empty LLM response")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def _build_prompt(attachment_name: str, excerpt: str) -> str:
    return (
        "You are extracting procurement requirements from a tender attachment. "
        "Return ONLY valid JSON with these exact keys:\n"
        "- requirements: array of short strings (exact phrases from the text)\n"
        "- key_risks: array of short strings\n"
        "- dates: array of short strings (closing dates, milestones, site visits)\n"
        "- attachments_used: array of strings (attachment names if present)\n"
        "Rules:\n"
        "- Do NOT invent facts or vendors.\n"
        "- If a requirement is not in the text, omit it.\n"
        "- If nothing is found, return empty arrays but include all keys.\n"
        "No markdown, no extra text.\n\n"
        f"ATTACHMENT: {attachment_name}\n"
        f"EXCERPT:\n{excerpt}\n"
    )


def _build_repair_prompt(bad_json: str) -> str:
    return (
        "Re-emit the response as VALID JSON ONLY, no extra text. "
        "Use this exact schema and include ALL keys:\n"
        "{\n"
        '  "requirements": [],\n'
        '  "key_risks": [],\n'
        '  "dates": [],\n'
        '  "attachments_used": []\n'
        "}\n"
        "Ensure double quotes and no trailing commas.\n\n"
        f"BAD_JSON:\n{bad_json}\n"
    )


def _cohere_chat_raw(
    prompt: str,
    *,
    api_key: str,
    api_base: str,
    model: str,
    timeout_s: int,
    logger: Optional[logging.Logger] = None,
) -> str:
    base = api_base.rstrip("/")
    bases = [base]
    if "api.cohere.com" in base:
        bases.append(base.replace("api.cohere.com", "api.cohere.ai"))
    elif "api.cohere.ai" in base:
        bases.append(base.replace("api.cohere.ai", "api.cohere.com"))

    urls: list[str] = []
    for b in bases:
        if b.endswith("/v2"):
            urls.append(f"{b}/chat")
        elif b.endswith("/v1"):
            urls.append(f"{b}/chat")
        else:
            urls.append(f"{b}/v2/chat")
            urls.append(f"{b}/v1/chat")
    urls = list(dict.fromkeys(urls))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    schema = {
        "type": "object",
        "required": ["requirements", "key_risks", "dates", "attachments_used"],
        "properties": {
            "requirements": {"type": "array", "items": {"type": "string"}},
            "key_risks": {"type": "array", "items": {"type": "string"}},
            "dates": {"type": "array", "items": {"type": "string"}},
            "attachments_used": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }

    payload_v2 = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 360,
        "response_format": {"type": "json_object", "schema": schema},
    }
    payload_v1 = {
        "model": model,
        "message": prompt,
        "temperature": 0.1,
        "max_tokens": 360,
    }

    last_err: Exception | None = None
    for url in urls:
        payload = payload_v2 if "/v2/" in url else payload_v1
        for attempt in range(4):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
                if resp.status_code == 404 and url != urls[-1]:
                    break
                if resp.status_code in {429, 500, 502, 503, 504}:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_s = int(retry_after) if retry_after and retry_after.isdigit() else 1 + attempt
                    if logger:
                        logger.warning(
                            "Attachment LLM: retryable HTTP %s on %s (attempt %s/4, sleeping %ss)",
                            resp.status_code,
                            url,
                            attempt + 1,
                            sleep_s,
                        )
                    time.sleep(sleep_s)
                    continue
                if resp.status_code >= 400:
                    detail = resp.text.strip()
                    msg = f"HTTP {resp.status_code} from {url}"
                    if detail:
                        msg = f"{msg} | {detail[:400]}"
                    raise RuntimeError(msg)
                data = resp.json()
                text = data.get("text")
                if not text:
                    message = data.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, list):
                            parts = []
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    parts.append(str(item.get("text", "")))
                                elif isinstance(item, str):
                                    parts.append(item)
                            text = "".join(parts).strip()
                        elif isinstance(content, str):
                            text = content
                if not text and "generations" in data and data["generations"]:
                    text = data["generations"][0].get("text")
                if not text and "message" in data:
                    text = data.get("message")
                return text or ""
            except Exception as exc:
                last_err = exc
                if logger:
                    logger.warning(
                        "Attachment LLM: attempt %s/4 failed for %s (%s)",
                        attempt + 1,
                        url,
                        exc,
                    )
                time.sleep(1 + attempt)

    raise RuntimeError(f"Cohere request failed after retries: {last_err}")


def _cohere_chat(prompt: str, config: AttachmentsConfig, logger=None) -> dict:
    text = _cohere_chat_raw(
        prompt,
        api_key=config.llm_api_key,
        api_base=config.llm_api_base,
        model=config.llm_model,
        timeout_s=config.llm_timeout_s,
        logger=logger,
    )
    try:
        return _extract_json(text)
    except Exception as exc:
        if logger:
            logger.warning("Attachment LLM: invalid JSON, attempting repair (%s)", exc)
        repair_prompt = _build_repair_prompt(text)
        repaired = _cohere_chat_raw(
            repair_prompt,
            api_key=config.llm_api_key,
            api_base=config.llm_api_base,
            model=config.llm_model,
            timeout_s=config.llm_timeout_s,
            logger=logger,
        )
        return _extract_json(repaired)


def _coerce_list(value: Any, max_items: int = 8) -> list[str]:
    if value is None:
        items: list[str] = []
    elif isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
    elif isinstance(value, str):
        items = [value.strip()] if value.strip() else []
    else:
        items = [str(value).strip()] if str(value).strip() else []
    return items[:max_items]


def _normalize_summary(summary: dict) -> dict:
    normalized = {
        "requirements": _coerce_list(summary.get("requirements"), max_items=8),
        "key_risks": _coerce_list(summary.get("key_risks"), max_items=6),
        "dates": _coerce_list(summary.get("dates"), max_items=6),
        "attachments_used": _coerce_list(summary.get("attachments_used"), max_items=4),
    }
    return normalized


def _extract_docx_text(path: Path) -> str:
    try:
        from docx import Document
    except Exception as exc:
        raise RuntimeError("python-docx is required for DOCX extraction") from exc
    doc = Document(path)
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts)


def _extract_pdf_text(path: Path) -> tuple[str, int]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("pypdf is required for PDF extraction") from exc
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    cleaned_pages = _strip_repeated_lines(pages)
    return "\n\n".join(cleaned_pages).strip(), len(reader.pages)


def _ocr_pdf_text(path: Path, logger: Optional[logging.Logger] = None) -> str:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        if logger:
            logger.warning("OCR dependencies missing; install pdf2image + pytesseract to enable OCR.")
        return ""
    try:
        images = convert_from_path(str(path), dpi=200)
        text_parts = [pytesseract.image_to_string(img) for img in images]
        return "\n".join(part for part in text_parts if part).strip()
    except Exception as exc:
        if logger:
            logger.warning("OCR failed for %s: %s", path.name, exc)
        return ""


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    if len(pages) < 2:
        return [p.strip() for p in pages]
    counts: Counter[str] = Counter()
    for page in pages:
        for line in page.splitlines():
            line = line.strip()
            if line:
                counts[line] += 1
    repeated = {line for line, count in counts.items() if count >= 2}
    cleaned_pages = []
    for page in pages:
        lines = [line for line in page.splitlines() if line.strip() and line.strip() not in repeated]
        cleaned_pages.append("\n".join(lines).strip())
    return cleaned_pages


def _normalize_extracted_text(text: str) -> str:
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _looks_like_heading(line: str) -> bool:
    if not line:
        return False
    if len(line) > 120:
        return False
    if re.match(r"^\d+(\.\d+)*\.?\s+\S", line):
        return True
    if line.endswith(":"):
        return True
    if line.isupper() and len(line) > 4:
        return True
    return False


def _sectionize(text: str, max_chars: int, max_sections: int) -> tuple[str, list[dict]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    headings: list[int] = []
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if not any(term in lowered for term in HEADING_TERMS):
            continue
        if _looks_like_heading(line):
            headings.append(idx)

    sections: list[dict] = []
    if not headings:
        excerpt = text[:max_chars].strip()
        return excerpt, []

    for i, h_idx in enumerate(headings[:max_sections]):
        start = h_idx
        end = headings[i + 1] if i + 1 < len(headings) else len(lines)
        heading = lines[h_idx]
        body = "\n".join(lines[start + 1 : end]).strip()
        sections.append(
            {"heading": heading, "text": body, "chars": len(body)}
        )

    assembled_parts = []
    for section in sections:
        if section["text"]:
            assembled_parts.append(f"{section['heading']}\n{section['text']}")
        else:
            assembled_parts.append(section["heading"])
    assembled = "\n\n".join(assembled_parts)
    if len(assembled) > max_chars:
        assembled = assembled[:max_chars].rsplit("\n", 1)[0].strip()
    return assembled, sections


def _build_attachment_memo(summary: dict) -> str:
    if not summary:
        return ""
    reqs = summary.get("requirements") or []
    dates = summary.get("dates") or []
    risks = summary.get("key_risks") or []
    parts: list[str] = []
    if reqs:
        parts.append(f"Requirements: {', '.join(str(r) for r in reqs[:3])}.")
    if dates:
        parts.append(f"Dates: {', '.join(str(d) for d in dates[:2])}.")
    if risks:
        parts.append(f"Risks: {', '.join(str(r) for r in risks[:2])}.")
    memo = " ".join(p.strip() for p in parts if p).strip()
    return memo[:300]


def _download_attachment(
    url: str,
    *,
    config: AttachmentsConfig,
    store: TenderStore,
    cache_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> tuple[Optional[dict], Optional[Path], Optional[str], bool]:
    url_norm = _normalize_url(url)
    if not url_norm:
        return None, None, "invalid_url", False

    cache_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "sap-tender-bot/attachment/0.1"}
    latest = store.fetch_latest_attachment_meta(url_norm=url_norm)
    if latest:
        if latest.get("etag"):
            headers["If-None-Match"] = latest.get("etag")
        if latest.get("last_modified"):
            headers["If-Modified-Since"] = latest.get("last_modified")

    def _retry_delay(attempt: int) -> None:
        time.sleep(1 + attempt)

    for attempt in range(config.download_retries + 1):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=config.timeout_s)
            if resp.status_code == 304 and latest:
                cached = store.fetch_attachment_cache_by_meta(
                    url_norm=url_norm,
                    etag=latest.get("etag"),
                    last_modified=latest.get("last_modified"),
                )
                if cached and cached.get("text_path") and Path(cached["text_path"]).exists():
                    return cached, Path(cached["text_path"]), None, True
                headers.pop("If-None-Match", None)
                headers.pop("If-Modified-Since", None)
                resp = requests.get(url, headers=headers, stream=True, timeout=config.timeout_s)

            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}")

            content_type = resp.headers.get("Content-Type", "")
            extension = _guess_extension(url, content_type)
            if extension not in SUPPORTED_EXT:
                return None, None, "unsupported_type", False

            content_length = resp.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                if int(content_length) > config.max_bytes:
                    return None, None, "too_large", False

            tmp_path = cache_dir / f"tmp_{hashlib.sha256(url_norm.encode('utf-8')).hexdigest()}.bin"
            digest = hashlib.sha256()
            size = 0
            with tmp_path.open("wb") as fp:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > config.max_bytes:
                        fp.close()
                        tmp_path.unlink(missing_ok=True)
                        return None, None, "too_large", False
                    digest.update(chunk)
                    fp.write(chunk)

            content_hash = digest.hexdigest()
            etag = _normalize_etag(resp.headers.get("ETag"))
            last_modified = resp.headers.get("Last-Modified")
            if etag or last_modified:
                fingerprint_seed = f"{url_norm}|{etag or ''}|{last_modified or ''}"
            else:
                fingerprint_seed = f"{url_norm}|{content_hash}"
            fingerprint = hashlib.sha256(fingerprint_seed.encode("utf-8")).hexdigest()

            ext_path = cache_dir / f"{fingerprint}{extension}"
            if not ext_path.exists():
                tmp_path.replace(ext_path)
            else:
                tmp_path.unlink(missing_ok=True)

            meta = {
                "fingerprint": fingerprint,
                "url_norm": url_norm,
                "url": url,
                "etag": etag,
                "last_modified": last_modified,
                "content_hash": content_hash,
                "mime_type": content_type.split(";", 1)[0].strip().lower() if content_type else "",
                "size_bytes": size,
                "fetched_at": _now_utc_iso(),
                "text_path": str(ext_path),
            }
            return meta, ext_path, None, False
        except Exception as exc:
            if logger:
                logger.warning(
                    "Attachment download failed (attempt %s/%s) for %s: %s",
                    attempt + 1,
                    config.download_retries + 1,
                    url,
                    exc,
                )
            if attempt >= config.download_retries:
                break
            _retry_delay(attempt)

    return None, None, "download_failed", False


def _extract_text(
    path: Path,
    *,
    mime_type: str,
    config: AttachmentsConfig,
    logger: Optional[logging.Logger] = None,
) -> tuple[str, dict]:
    page_count = 0
    if path.suffix.lower() == ".pdf":
        text, page_count = _extract_pdf_text(path)
    elif path.suffix.lower() == ".docx":
        text = _extract_docx_text(path)
    else:
        return "", {"page_count": 0}

    text = _normalize_extracted_text(text)
    meta = {"page_count": page_count}

    if config.ocr_enabled and len(text) < config.min_text_chars_for_ocr and path.suffix.lower() == ".pdf":
        ocr_text = _ocr_pdf_text(path, logger=logger)
        if ocr_text:
            text = _normalize_extracted_text(ocr_text)
            meta["ocr_used"] = True
        else:
            meta["ocr_used"] = False
    return text, meta


def _process_attachment_job(
    tender: dict,
    url: str,
    *,
    config: AttachmentsConfig,
    attachment_cache_dir: Path,
    db_path: Path,
    stats: dict,
    llm_state: dict,
    tender_state: dict[int, dict],
    stats_lock: threading.Lock,
    tender_lock: threading.Lock,
    logger: Optional[logging.Logger] = None,
) -> None:
    local_store = TenderStore(db_path, logger=logger).open()
    try:
        meta, path, skip_reason, from_cache = _download_attachment(
            url,
            config=config,
            store=local_store,
            cache_dir=attachment_cache_dir,
            logger=logger,
        )
        if skip_reason:
            with stats_lock:
                stats["attachments_skipped"] += 1
            if logger:
                logger.info("Attachment skipped (%s): %s", skip_reason, url)
            return

        if not meta or not path:
            with stats_lock:
                stats["attachments_skipped"] += 1
            return

        with stats_lock:
            if from_cache:
                stats["cached"] += 1
            else:
                stats["downloaded"] += 1
            stats["bytes"] += int(meta.get("size_bytes") or 0)

        local_store.upsert_attachment_cache(entry=meta)

        existing = None
        if meta.get("etag") or meta.get("last_modified"):
            existing = local_store.fetch_attachment_cache_by_meta(
                url_norm=meta["url_norm"],
                etag=meta.get("etag"),
                last_modified=meta.get("last_modified"),
            )
        if not existing and meta.get("content_hash"):
            existing = local_store.fetch_attachment_cache_by_hash(
                url_norm=meta["url_norm"],
                content_hash=meta.get("content_hash"),
            )

        summary = None
        sections: list[dict] = []
        if existing and existing.get("summary_json"):
            try:
                summary = json.loads(existing["summary_json"])
                with stats_lock:
                    stats["llm_cached"] += 1
            except Exception:
                summary = None

        if summary is None:
            text, extract_meta = _extract_text(
                path,
                mime_type=str(meta.get("mime_type") or ""),
                config=config,
                logger=logger,
            )
            if not text:
                with stats_lock:
                    stats["empty_text"] += 1
                    stats["attachments_skipped"] += 1
                if logger:
                    logger.info("Attachment extracted empty text: %s", url)
                return

            if extract_meta.get("ocr_used"):
                with stats_lock:
                    stats["ocr_used"] += 1
            with stats_lock:
                stats["chars"] += len(text)

            excerpt, sections = _sectionize(
                text,
                max_chars=config.section_char_budget,
                max_sections=config.max_sections,
            )
            if logger:
                logger.info(
                    "Attachment extract: kept=%s chars sections=%s url=%s",
                    len(excerpt),
                    len(sections),
                    url,
                )
            if config.pii_scrub:
                excerpt = _scrub_pii(excerpt)

            if not config.llm_enabled:
                summary = {
                    "requirements": [],
                    "key_risks": [],
                    "dates": [],
                    "attachments_used": [],
                }
                with stats_lock:
                    stats["llm_skipped"] += 1
            else:
                allowed = False
                with stats_lock:
                    remaining = llm_state.get("remaining")
                    daily_cap = llm_state.get("daily_cap", 0)
                    daily_used = llm_state.get("daily_used", 0)
                    if remaining is not None and remaining <= 0:
                        allowed = False
                    elif daily_cap > 0 and daily_used >= daily_cap:
                        allowed = False
                    else:
                        allowed = True
                        if remaining is not None:
                            llm_state["remaining"] = remaining - 1
                        llm_state["daily_used"] = daily_used + 1

                if allowed:
                    try:
                        prompt = _build_prompt(_attachment_name(url), excerpt)
                        if logger:
                            logger.info("Attachment LLM: requesting summary for %s", url)
                        summary = _normalize_summary(_cohere_chat(prompt, config, logger=logger))
                        with stats_lock:
                            stats["llm_requested"] += 1
                    except Exception:
                        if logger:
                            logger.exception("Attachment LLM failed for %s; skipping summary", url)
                        summary = {
                            "requirements": [],
                            "key_risks": [],
                            "dates": [],
                            "attachments_used": [],
                        }
                        with stats_lock:
                            stats["llm_skipped"] += 1
                else:
                    summary = {
                        "requirements": [],
                        "key_risks": [],
                        "dates": [],
                        "attachments_used": [],
                    }
                    with stats_lock:
                        stats["llm_skipped"] += 1

            meta["text_chars"] = len(text)
            meta["summary_cached_at"] = _now_utc_iso()
            meta["summary_json"] = json.dumps(summary, ensure_ascii=False)
            meta["summary_model"] = config.llm_model
            meta["requirements_json"] = json.dumps(summary.get("requirements", []), ensure_ascii=False)
            local_store.upsert_attachment_cache(entry=meta)

        summary = _normalize_summary(summary or {})
        memo = _build_attachment_memo(summary)
        attachment_summary = {
            "attachment_url": url,
            "attachment_name": _attachment_name(url),
            "fingerprint": meta.get("fingerprint"),
            "summary": summary,
            "memo": memo,
            "source_sections": [s.get("heading") for s in sections],
            "text_chars": meta.get("text_chars"),
            "size_bytes": meta.get("size_bytes"),
            "mime_type": meta.get("mime_type"),
        }

        requirements = []
        for req in summary.get("requirements", []):
            requirements.append(
                {
                    "requirement": req,
                    "attachment_url": url,
                    "attachment_name": _attachment_name(url),
                    "fingerprint": meta.get("fingerprint"),
                    "source_sections": attachment_summary.get("source_sections", []),
                }
            )

        with tender_lock:
            state = tender_state.setdefault(
                id(tender), {"summary_fps": set(), "req_keys": set()}
            )
            summary_fps: set[str] = state["summary_fps"]
            req_keys: set[tuple[str, str]] = state["req_keys"]

            fingerprint = str(meta.get("fingerprint") or "")
            if fingerprint and fingerprint not in summary_fps:
                tender.setdefault("attachment_summaries", []).append(attachment_summary)
                summary_fps.add(fingerprint)
            elif not fingerprint:
                tender.setdefault("attachment_summaries", []).append(attachment_summary)

            for req in requirements:
                key = (
                    fingerprint,
                    re.sub(r"\s+", " ", str(req.get("requirement") or "").strip().lower()),
                )
                if key in req_keys:
                    continue
                req_keys.add(key)
                tender.setdefault("attachment_requirements", []).append(req)

        with stats_lock:
            stats["attachments_processed"] += 1
    finally:
        local_store.close()


def enrich_tenders_with_attachments(
    tenders: Iterable[dict],
    *,
    config: AttachmentsConfig,
    store: TenderStore,
    cache_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> dict:
    stats = {
        "attachments_total": 0,
        "attachments_processed": 0,
        "attachments_skipped": 0,
        "downloaded": 0,
        "cached": 0,
        "llm_requested": 0,
        "llm_cached": 0,
        "llm_skipped": 0,
        "ocr_used": 0,
        "empty_text": 0,
        "bytes": 0,
        "chars": 0,
    }

    if not config.enabled:
        return stats

    if not config.llm_api_key and config.llm_enabled:
        if logger:
            logger.warning("Attachment LLM API key missing; skipping attachment summaries")
        config.llm_enabled = False

    attachment_cache_dir = cache_dir / "attachments"
    attachment_cache_dir.mkdir(parents=True, exist_ok=True)

    today = _today_utc()
    daily_used = 0
    if config.llm_daily_cap > 0:
        usage = _load_daily_usage(attachment_cache_dir, today)
        daily_used = int(usage.get("count", 0))

    remaining_llm = config.llm_per_run_max if config.llm_per_run_max > 0 else None
    llm_state = {
        "remaining": remaining_llm,
        "daily_cap": int(config.llm_daily_cap or 0),
        "daily_used": daily_used,
    }

    processed_urls = set()
    jobs: list[tuple[dict, str]] = []
    tender_state: dict[int, dict] = {}

    for t in tenders:
        existing_summaries = t.get("attachment_summaries") or []
        existing_summary_fps = {
            s.get("fingerprint") for s in existing_summaries if isinstance(s, dict)
        }
        existing_req_keys = set()
        for req in t.get("attachment_requirements") or []:
            if isinstance(req, dict):
                key = (
                    str(req.get("fingerprint") or ""),
                    re.sub(r"\s+", " ", str(req.get("requirement") or "").strip().lower()),
                )
                existing_req_keys.add(key)
        tender_state[id(t)] = {
            "summary_fps": set(existing_summary_fps),
            "req_keys": set(existing_req_keys),
        }

        attachments = t.get("attachments") or []
        if isinstance(attachments, str):
            attachments = [attachments]
        urls = [a for a in attachments if isinstance(a, str) and a.startswith("http")]
        if not urls:
            continue
        if config.max_attachments_per_tender > 0:
            urls = urls[: config.max_attachments_per_tender]

        for url in urls:
            if config.max_attachments_per_run > 0 and len(jobs) >= config.max_attachments_per_run:
                break
            if url in processed_urls:
                continue
            processed_urls.add(url)
            jobs.append((t, url))

    stats["attachments_total"] = len(jobs)
    if not jobs:
        return stats

    stats_lock = threading.Lock()
    tender_lock = threading.Lock()
    max_workers = max(1, int(config.download_concurrency or 1))
    db_path = store.db_path

    if max_workers <= 1:
        for t, url in jobs:
            _process_attachment_job(
                t,
                url,
                config=config,
                attachment_cache_dir=attachment_cache_dir,
                db_path=db_path,
                stats=stats,
                llm_state=llm_state,
                tender_state=tender_state,
                stats_lock=stats_lock,
                tender_lock=tender_lock,
                logger=logger,
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _process_attachment_job,
                    t,
                    url,
                    config=config,
                    attachment_cache_dir=attachment_cache_dir,
                    db_path=db_path,
                    stats=stats,
                    llm_state=llm_state,
                    tender_state=tender_state,
                    stats_lock=stats_lock,
                    tender_lock=tender_lock,
                    logger=logger,
                )
                for t, url in jobs
            ]
            for future in as_completed(futures):
                future.result()

    if config.llm_daily_cap > 0:
        _save_daily_usage(attachment_cache_dir, today, llm_state.get("daily_used", daily_used))

    return stats


def merge_attachment_requirements(tender: dict) -> None:
    attachment_reqs = tender.get("attachment_requirements") or []
    if not attachment_reqs:
        return

    seen = set()
    merged: list[str] = []
    for req in attachment_reqs:
        text = req.get("requirement") if isinstance(req, dict) else str(req)
        if not text:
            continue
        key = re.sub(r"\s+", " ", text.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(text.strip())

    if not merged:
        return

    llm = tender.get("llm")
    if isinstance(llm, dict):
        if "key_requirements_notice" not in llm:
            llm["key_requirements_notice"] = llm.get("key_requirements", [])
        llm["key_requirements"] = merged[:6]
        llm["attachment_summary_used"] = True

    tender["attachment_summary_used"] = True
