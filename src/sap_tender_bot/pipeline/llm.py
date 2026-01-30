from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests

from sap_tender_bot.config import LLMConfig
from sap_tender_bot.utils.rate_limit import rate_limit_wait

DEFAULT_SCHEMA = {
    "relevance": "medium",
    "opportunity_category": "other",
    "SAP_solution_fit": {"fit": "unknown", "notes": []},
    "key_requirements": [],
    "stakeholder_entities": {"buyer_org": "", "program_name": "", "other_entities": []},
    "timeline": {"close_date": "", "expected_start": "", "milestones": [], "notes": ""},
    "competitive_signals": [],
    "recommended_next_step": "",
    "summary": "",
    "source_confidence": "low",
}

_APOSTROPHES = {
    "’": "'",
    "‘": "'",
    "‛": "'",
    "ʼ": "'",
    "`": "'",
    "´": "'",
}

_RELEVANCE_MAP = {
    "high": "high",
    "medium": "medium",
    "moderate": "medium",
    "med": "medium",
    "low": "low",
}

_CATEGORY_MAP = {
    "erp modernization": "ERP modernization",
    "erp modernisation": "ERP modernization",
    "erp upgrade": "ERP modernization",
    "erp migration": "ERP modernization",
    "financial systems": "ERP modernization",
    "procurement": "procurement",
    "source to pay": "procurement",
    "source-to-pay": "procurement",
    "procure to pay": "procurement",
    "procure-to-pay": "procurement",
    "hr": "HR/payroll",
    "hcm": "HR/payroll",
    "hris": "HR/payroll",
    "payroll": "HR/payroll",
    "integration": "integration/data",
    "data integration": "integration/data",
    "data": "integration/data",
    "master data": "integration/data",
    "ams": "AMS",
    "application management": "AMS",
    "application management services": "AMS",
    "managed services": "AMS",
}

_FIT_MAP = {
    "strong": "strong",
    "high": "strong",
    "good": "strong",
    "moderate": "moderate",
    "medium": "moderate",
    "fair": "moderate",
    "weak": "weak",
    "low": "weak",
    "none": "weak",
    "unknown": "unknown",
    "unclear": "unknown",
}

_SAP_HINTS = [
    "sap",
    "s/4hana",
    "s4hana",
    "ecc",
    "fiori",
    "successfactors",
    "ariba",
    "concur",
    "sap btp",
    "sap cpi",
]

_MOJIBAKE_MARKERS = ("Ã", "â", "Â")




def _fix_mojibake(value: str) -> str:
    if not value:
        return value
    if any(token in value for token in _MOJIBAKE_MARKERS):
        try:
            return value.encode("latin-1").decode("utf-8")
        except Exception:
            return value
    return value


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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


def _cache_path(cache_key: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _load_cache(cache_keys: Iterable[str], cache_dir: Path) -> Dict[str, Any] | None:
    for cache_key in cache_keys:
        path = _cache_path(cache_key, cache_dir)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload["cache_key"] = cache_key
            return payload
        except Exception:
            continue
    return None


def _save_cache(cache_key: str, data: Dict[str, Any], cache_dir: Path, cached_at: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_key, cache_dir)
    payload = {"cache_key": cache_key, "cached_at": cached_at, "llm": data}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_prompt(t: Dict[str, Any]) -> str:
    title = t.get("title") or ""
    title_en = t.get("title_en") or ""
    title_fr = t.get("title_fr") or ""
    desc = t.get("description") or ""
    desc_en = t.get("description_en") or ""
    desc_fr = t.get("description_fr") or ""
    if not desc:
        desc = " ".join(str(p) for p in [desc_en, desc_fr] if p)
    org = t.get("org") or ""
    org_en = t.get("org_en") or ""
    org_fr = t.get("org_fr") or ""
    close_date = t.get("close_date") or ""
    updated_at = t.get("updated_at") or ""
    unspsc_en = ", ".join(t.get("unspsc_desc_en", "").splitlines())
    unspsc_fr = ", ".join(t.get("unspsc_desc_fr", "").splitlines())
    gsin_en = ", ".join(t.get("gsin_desc_en", "").splitlines())
    gsin_fr = ", ".join(t.get("gsin_desc_fr", "").splitlines())
    attachments = t.get("attachments") or []
    if isinstance(attachments, str):
        attachments = [attachments]
    attachments_text = ", ".join(str(a) for a in attachments if a)

    return (
        "You are analyzing a Canadian public sector tender for SAP/ERP pursuit planning. "
        "Return ONLY valid JSON with these exact keys and types:\n"
        "- relevance: string (one of: high, medium, low)\n"
        "- opportunity_category: string (one of: ERP modernization, procurement, HR/payroll, "
        "integration/data, AMS, other)\n"
        "- SAP_solution_fit: object with keys fit (strong, moderate, weak, unknown) and notes "
        "(array of short strings)\n"
        "- key_requirements: array of short strings (2-6 items)\n"
        "- stakeholder_entities: object with keys buyer_org, program_name, other_entities "
        "(array of strings)\n"
        "- timeline: object with keys close_date (ISO if known), expected_start, milestones "
        "(array), notes\n"
        "- competitive_signals: array of short strings (empty if none)\n"
        "- recommended_next_step: short string (imperative)\n"
        "If unknown, use empty strings/lists. Do NOT invent facts or vendors. "
        "Write like a concise pursuit brief.\n"
        "Hard rules:\n"
        "- competitive_signals must be EMPTY unless explicitly stated in the tender text.\n"
        "- Avoid generic phrases (e.g., 'enterprise systems scope', 'prepare a proposal').\n"
        "- key_requirements must be exact phrases from the text (verbatim or near-verbatim).\n"
        "- If you cannot ground a requirement in the text, leave it out.\n"
        "- recommended_next_step must be ONE of:\n"
        "  * Review RFP for scope and mandatory requirements\n"
        "  * Confirm incumbent and contract vehicle\n"
        "  * Validate SAP/ERP scope with buyer questions\n"
        "  * Identify required certifications and security clearances\n"
        "  * Decide bid/no-bid based on capability gaps\n"
        "If none apply, leave recommended_next_step empty.\n"
        "Use both English and French fields below if present; Cohere is multilingual.\n"
        "No markdown, no extra text.\n\n"
        f"TITLE: {title}\n"
        f"TITLE_EN: {title_en}\n"
        f"TITLE_FR: {title_fr}\n"
        f"BUYER ORG: {org}\n"
        f"BUYER ORG_EN: {org_en}\n"
        f"BUYER ORG_FR: {org_fr}\n"
        f"UPDATED_AT: {updated_at}\n"
        f"CLOSE_DATE: {close_date}\n"
        f"UNSPSC_DESC_EN: {unspsc_en}\n"
        f"UNSPSC_DESC_FR: {unspsc_fr}\n"
        f"GSIN_DESC_EN: {gsin_en}\n"
        f"GSIN_DESC_FR: {gsin_fr}\n"
        f"ATTACHMENTS: {attachments_text}\n"
        f"DESCRIPTION: {desc}\n"
        f"DESCRIPTION_EN: {desc_en}\n"
        f"DESCRIPTION_FR: {desc_fr}\n"
    )


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty LLM response")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def _build_repair_prompt(bad_json: str) -> str:
    return (
        "Re-emit the response as VALID JSON ONLY, no extra text. "
        "Use this exact schema and include ALL keys:\n"
        "{\n"
        '  "relevance": "high|medium|low",\n'
        '  "opportunity_category": "ERP modernization|procurement|HR/payroll|integration/data|AMS|other",\n'
        '  "SAP_solution_fit": {"fit": "strong|moderate|weak|unknown", "notes": []},\n'
        '  "key_requirements": [],\n'
        '  "stakeholder_entities": {"buyer_org": "", "program_name": "", "other_entities": []},\n'
        '  "timeline": {"close_date": "", "expected_start": "", "milestones": [], "notes": ""},\n'
        '  "competitive_signals": [],\n'
        '  "recommended_next_step": "",\n'
        '  "summary": ""\n'
        "}\n"
        "Ensure double quotes and no trailing commas.\n\n"
        f"BAD_JSON:\n{bad_json}\n"
    )


def _cohere_chat_raw(prompt: str, config: LLMConfig, logger=None) -> str:
    base_url = config.api_base
    model = config.model
    timeout_s = config.timeout_s

    base = base_url.rstrip("/")
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
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    schema = {
        "type": "object",
        "required": [
            "relevance",
            "opportunity_category",
            "SAP_solution_fit",
            "key_requirements",
            "stakeholder_entities",
            "timeline",
            "competitive_signals",
            "recommended_next_step",
            "summary",
        ],
        "properties": {
            "relevance": {"type": "string", "enum": ["high", "medium", "low"]},
            "opportunity_category": {
                "type": "string",
                "enum": [
                    "ERP modernization",
                    "procurement",
                    "HR/payroll",
                    "integration/data",
                    "AMS",
                    "other",
                ],
            },
            "SAP_solution_fit": {
                "type": "object",
                "required": ["fit", "notes"],
                "properties": {
                    "fit": {"type": "string", "enum": ["strong", "moderate", "weak", "unknown"]},
                    "notes": {"type": "array", "items": {"type": "string"}},
                },
            },
            "key_requirements": {"type": "array", "items": {"type": "string"}},
            "stakeholder_entities": {
                "type": "object",
                "required": ["buyer_org", "program_name", "other_entities"],
                "properties": {
                    "buyer_org": {"type": "string"},
                    "program_name": {"type": "string"},
                    "other_entities": {"type": "array", "items": {"type": "string"}},
                },
            },
            "timeline": {
                "type": "object",
                "required": ["close_date", "expected_start", "milestones", "notes"],
                "properties": {
                    "close_date": {"type": "string"},
                    "expected_start": {"type": "string"},
                    "milestones": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
            },
            "competitive_signals": {"type": "array", "items": {"type": "string"}},
            "recommended_next_step": {"type": "string"},
            "summary": {"type": "string"},
        },
        "additionalProperties": False,
    }

    payload_v2 = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 400,
        "response_format": {"type": "json_object", "schema": schema},
    }
    payload_v1 = {
        "model": model,
        "message": prompt,
        "temperature": 0.2,
        "max_tokens": 400,
    }

    last_err: Exception | None = None
    for url in urls:
        had_429 = False
        payload = payload_v2 if "/v2/" in url else payload_v1
        for attempt in range(4):
            try:
                rate_limit_wait("cohere", config.max_requests_per_minute)
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
                if resp.status_code == 404 and url != urls[-1]:
                    break
                if resp.status_code in {429, 500, 502, 503, 504}:
                    if resp.status_code == 429:
                        had_429 = True
                    retry_after = resp.headers.get("Retry-After")
                    sleep_s = int(retry_after) if retry_after and retry_after.isdigit() else 1 + attempt
                    if logger:
                        logger.warning(
                            "LLM: retryable HTTP %s on %s (attempt %s/4, sleeping %ss)",
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
                        "LLM: attempt %s/4 failed for %s (%s)",
                        attempt + 1,
                        url,
                        exc,
                    )
                time.sleep(1 + attempt)
        if had_429:
            break

    raise RuntimeError(f"Cohere request failed after retries: {last_err}")


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    for src, dst in _APOSTROPHES.items():
        text = text.replace(src, dst)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _tender_text(t: Dict[str, Any]) -> str:
    parts = [
        t.get("title", ""),
        t.get("title_en", ""),
        t.get("title_fr", ""),
        t.get("description", ""),
        t.get("description_en", ""),
        t.get("description_fr", ""),
        t.get("unspsc_desc_en", ""),
        t.get("unspsc_desc_fr", ""),
        t.get("gsin_desc_en", ""),
        t.get("gsin_desc_fr", ""),
    ]
    return _normalize_text(" ".join(str(p) for p in parts if p))


def _tender_text_raw(t: Dict[str, Any]) -> str:
    parts = [
        t.get("title", ""),
        t.get("title_en", ""),
        t.get("title_fr", ""),
        t.get("description", ""),
        t.get("description_en", ""),
        t.get("description_fr", ""),
        t.get("unspsc_desc_en", ""),
        t.get("unspsc_desc_fr", ""),
        t.get("gsin_desc_en", ""),
        t.get("gsin_desc_fr", ""),
    ]
    return " ".join(str(p) for p in parts if p)


def _is_requirement_grounded(requirement: str, tender_text_norm: str, tender_text_raw: str) -> bool:
    if not requirement:
        return False
    req_norm = _normalize_text(requirement)
    if not req_norm:
        return False
    raw = tender_text_raw.lower()
    req_raw = requirement.lower().strip()
    if req_raw and req_raw in raw:
        return True
    return req_norm in tender_text_norm


def _coerce_list(value: Any, max_items: int | None = None) -> list[str]:
    if value is None:
        items: list[str] = []
    elif isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
    elif isinstance(value, str):
        items = [value.strip()] if value.strip() else []
    else:
        items = [str(value).strip()] if str(value).strip() else []
    if max_items is not None:
        items = items[:max_items]
    return items


def _coerce_entities(value: Any, t: Dict[str, Any]) -> dict:
    buyer_org = ""
    program_name = ""
    if isinstance(value, dict):
        buyer_org = str(
            value.get("buyer_org")
            or value.get("buyer")
            or value.get("org")
            or ""
        ).strip()
        program_name = str(
            value.get("program_name")
            or value.get("program")
            or value.get("project")
            or ""
        ).strip()
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                role = str(item.get("role", "")).lower()
                name = str(item.get("name") or item.get("org") or "").strip()
                if not buyer_org and name:
                    buyer_org = name
                if "buyer" in role or "client" in role or "owner" in role:
                    buyer_org = name or buyer_org
                if "program" in role or "project" in role or "initiative" in role:
                    program_name = name or program_name
            elif isinstance(item, str) and not buyer_org:
                buyer_org = item.strip()
    elif isinstance(value, str):
        buyer_org = value.strip()

    if not buyer_org:
        buyer_org = str(t.get("org") or "").strip()

    return {"buyer_org": buyer_org, "program_name": program_name}


def _normalize_category(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    raw = _normalize_text(str(value or ""))
    if not raw:
        return "other"
    for key, mapped in _CATEGORY_MAP.items():
        if key in raw:
            return mapped
    if raw in _CATEGORY_MAP:
        return _CATEGORY_MAP[raw]
    if "erp" in raw or "financial" in raw:
        return "ERP modernization"
    if "procure" in raw or "supply" in raw:
        return "procurement"
    if "payroll" in raw or "hr" in raw:
        return "HR/payroll"
    if "integration" in raw or "data" in raw:
        return "integration/data"
    if "ams" in raw or "managed" in raw:
        return "AMS"
    return "other"


def _normalize_fit(value: Any) -> str:
    raw = _normalize_text(str(value or ""))
    if not raw:
        return "unknown"
    for key, mapped in _FIT_MAP.items():
        if key in raw:
            return mapped
    return "unknown"


def _coerce_fit(value: Any) -> dict:
    fit = "unknown"
    notes: list[str] = []
    if isinstance(value, dict):
        fit = _normalize_fit(value.get("fit") or value.get("fit_level") or value.get("level"))
        notes = _coerce_list(
            value.get("notes") or value.get("details") or value.get("evidence"),
            max_items=4,
        )
    elif isinstance(value, list):
        notes = _coerce_list(value, max_items=4)
    elif isinstance(value, str):
        fit = _normalize_fit(value)
        if ":" in value:
            parts = value.split(":", 1)
            if parts[1].strip():
                notes = _coerce_list(parts[1].strip(), max_items=2)
    return {"fit": fit or "unknown", "notes": notes}


def _coerce_stakeholders(value: Any, t: Dict[str, Any]) -> dict:
    base = _coerce_entities(value, t)
    buyer_org = base.get("buyer_org", "")
    program_name = base.get("program_name", "")
    other_entities: list[str] = []

    if isinstance(value, dict):
        for key in ("incumbent", "prime", "prime_vendor", "partner", "vendor", "vendors", "stakeholders"):
            other_entities.extend(_coerce_list(value.get(key), max_items=4))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("org") or "").strip()
                if name:
                    other_entities.append(name)
            elif isinstance(item, str):
                other_entities.append(item.strip())
    elif isinstance(value, str):
        other_entities.extend(_coerce_list(value, max_items=3))

    cleaned = []
    for name in other_entities:
        if not name:
            continue
        if name == buyer_org or name == program_name:
            continue
        if name not in cleaned:
            cleaned.append(name)

    return {"buyer_org": buyer_org, "program_name": program_name, "other_entities": cleaned[:6]}


def _coerce_timeline(value: Any, t: Dict[str, Any]) -> dict:
    close_date = ""
    expected_start = ""
    milestones: list[str] = []
    notes = ""

    if isinstance(value, dict):
        close_date = str(value.get("close_date") or value.get("deadline") or "").strip()
        expected_start = str(value.get("expected_start") or value.get("start") or "").strip()
        milestones = _coerce_list(value.get("milestones"), max_items=4)
        notes = str(value.get("notes") or value.get("note") or "").strip()
    elif isinstance(value, list):
        milestones = _coerce_list(value, max_items=4)
    elif isinstance(value, str):
        notes = value.strip()

    if not close_date:
        close_date = str(t.get("close_date") or "").strip()

    return {
        "close_date": close_date,
        "expected_start": expected_start,
        "milestones": milestones,
        "notes": notes,
    }


def _build_brief_summary(brief: dict, t: Dict[str, Any]) -> str:
    buyer = (
        (brief.get("stakeholder_entities") or {}).get("buyer_org")
        or str(t.get("org") or "").strip()
    )
    buyer = _fix_mojibake(buyer)
    category = brief.get("opportunity_category") or "other"
    fit = (brief.get("SAP_solution_fit") or {}).get("fit") or "unknown"
    key_reqs = brief.get("key_requirements") or []
    next_step = brief.get("recommended_next_step") or ""
    confidence = brief.get("source_confidence") or ""
    timeline = brief.get("timeline") or {}
    close_date = str(timeline.get("close_date") or "").strip()

    if category == "other" or not key_reqs:
        head = f"{buyer}: limited detail in notice; needs review." if buyer else "Limited detail in notice; needs review."
        parts = [head]
        if close_date:
            parts.append(f"Closes {close_date}.")
        if next_step:
            parts.append(f"Next: {next_step}.")
        if confidence:
            parts.append(f"Confidence: {confidence}.")
        summary = " ".join(p.strip() for p in parts if p).strip()
        return summary[:300]

    category_phrase = category
    head = f"{buyer}: {category_phrase} opportunity" if buyer else f"{category_phrase} opportunity"
    if fit and fit != "unknown":
        head = f"{head} (SAP fit: {fit})."
    else:
        head = f"{head}."

    parts: list[str] = [head]
    if key_reqs:
        parts.append(f"Key needs: {', '.join(str(r) for r in key_reqs[:2])}.")
    if close_date:
        parts.append(f"Closes {close_date}.")
    if next_step:
        parts.append(f"Next: {next_step}.")
    if confidence:
        parts.append(f"Confidence: {confidence}.")

    summary = " ".join(p.strip() for p in parts if p).strip()
    return summary[:300]


def _cache_key_for_tender(t: Dict[str, Any]) -> Optional[str]:
    tender_key = str(t.get("tender_key") or "").strip()
    if not tender_key:
        tender_key = str(t.get("uid") or "").strip()
    if not tender_key:
        return None
    amendment = str(t.get("amendment_number") or t.get("amendment") or "").strip()
    if amendment:
        return f"{tender_key}:am{amendment}"
    return tender_key


def _cache_key_candidates(t: Dict[str, Any]) -> list[str]:
    keys: list[str] = []
    primary = _cache_key_for_tender(t)
    if primary:
        keys.append(primary)
    uid = str(t.get("uid") or "").strip()
    if uid and uid not in keys:
        keys.append(uid)
    return keys


def _normalize_llm_result(result: Dict[str, Any], t: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    relevance_raw = str(result.get("relevance", "")).strip().lower()
    normalized["relevance"] = _RELEVANCE_MAP.get(relevance_raw, "medium")

    normalized["opportunity_category"] = _normalize_category(
        result.get("opportunity_category") or result.get("opportunity_type")
    )
    normalized["SAP_solution_fit"] = _coerce_fit(result.get("SAP_solution_fit"))
    normalized["key_requirements"] = _coerce_list(result.get("key_requirements"), max_items=6)
    normalized["stakeholder_entities"] = _coerce_stakeholders(
        result.get("stakeholder_entities") or result.get("entities"),
        t,
    )
    normalized["timeline"] = _coerce_timeline(result.get("timeline"), t)
    normalized["competitive_signals"] = _coerce_list(
        result.get("competitive_signals"),
        max_items=4,
    )
    recommended = result.get("recommended_next_step") or result.get("recommended_next_steps")
    if isinstance(recommended, list):
        recommended = recommended[0] if recommended else ""
    normalized["recommended_next_step"] = str(recommended or "").strip()

    tender_text = _tender_text(t)
    tender_text_raw = _tender_text_raw(t)
    tender_text_norm = tender_text
    sap_explicit = any(h in tender_text_norm for h in _SAP_HINTS)

    if not sap_explicit:
        normalized["SAP_solution_fit"] = {"fit": "unknown", "notes": []}

    grounded_requirements = [
        r
        for r in normalized.get("key_requirements", [])
        if _is_requirement_grounded(r, tender_text, tender_text_raw)
    ]
    if len(grounded_requirements) < 2:
        grounded_requirements = []
    normalized["key_requirements"] = grounded_requirements

    if sap_explicit and len(grounded_requirements) >= 3:
        normalized["source_confidence"] = "high"
    elif len(grounded_requirements) >= 2:
        normalized["source_confidence"] = "medium"
    else:
        normalized["source_confidence"] = "low"

    if normalized["source_confidence"] == "low":
        normalized["key_requirements"] = []

    for key in DEFAULT_SCHEMA:
        normalized.setdefault(key, DEFAULT_SCHEMA[key])

    normalized["summary"] = _build_brief_summary(normalized, t)

    return normalized


def _cohere_chat(prompt: str, config: LLMConfig, logger=None) -> Dict[str, Any]:
    text = _cohere_chat_raw(prompt, config=config, logger=logger)
    try:
        return _extract_json(text)
    except Exception as exc:
        if logger:
            logger.warning("LLM: invalid JSON, attempting repair (%s)", exc)
        repair_prompt = _build_repair_prompt(text)
        repaired = _cohere_chat_raw(repair_prompt, config=config, logger=logger)
        try:
            return _extract_json(repaired)
        except Exception as exc2:
            if logger:
                snippet = repaired.replace("\n", " ")[:240] if repaired else ""
                logger.error("LLM: repair failed (%s). Snippet: %s", exc2, snippet)
            raise


def enrich_tenders_with_llm(
    tenders: Iterable[Dict[str, Any]],
    config: LLMConfig,
    cache_dir: Path,
    logger=None,
    max_requests: Optional[int] = None,
) -> dict:
    if not config.enabled or not config.api_key:
        if logger:
            logger.warning("COHERE_API_KEY missing; skipping LLM enrichment")
        return {"requested": 0, "cached": 0, "skipped": 0}

    cache_dir = cache_dir / "llm"
    remaining = max_requests if max_requests is not None else None
    stats = {
        "requested": 0,
        "cached": 0,
        "skipped": 0,
        "daily_remaining": None,
        "with_requirements": 0,
        "confidence_high": 0,
        "confidence_medium": 0,
        "confidence_low": 0,
    }

    daily_cap = int(getattr(config, "daily_cap", 0) or 0)
    today = _today_utc()
    daily_used = 0
    if daily_cap > 0:
        usage = _load_daily_usage(cache_dir, today)
        daily_used = int(usage.get("count", 0))
        stats["daily_remaining"] = max(0, daily_cap - daily_used)

    for t in tenders:
        cache_keys = _cache_key_candidates(t)
        if not cache_keys:
            continue

        cached = _load_cache(cache_keys, cache_dir)
        if cached and cached.get("llm"):
            cached_llm = cached["llm"]
            if isinstance(cached_llm, dict) and "opportunity_category" not in cached_llm:
                cached_llm = _normalize_llm_result(cached_llm, t)
            t["llm"] = cached_llm
            t["_llm_cached_at"] = cached.get("cached_at")
            stats["cached"] += 1
            continue

        if remaining is not None and remaining <= 0:
            stats["skipped"] += 1
            continue
        if daily_cap > 0 and daily_used >= daily_cap:
            stats["skipped"] += 1
            continue

        prompt = _build_prompt(t)
        if logger:
            logger.info("LLM: requesting summary for %s", cache_keys[0])

        start = time.perf_counter()
        try:
            result = _cohere_chat(prompt, config=config, logger=logger)
            elapsed = time.perf_counter() - start
            if logger:
                logger.info("LLM: completed %s in %.1fs", cache_keys[0], elapsed)
            result = _normalize_llm_result(result, t)
            if result.get("key_requirements"):
                stats["with_requirements"] += 1
            confidence = result.get("source_confidence")
            if confidence == "high":
                stats["confidence_high"] += 1
            elif confidence == "medium":
                stats["confidence_medium"] += 1
            else:
                stats["confidence_low"] += 1
        except Exception:
            if logger:
                logger.exception("LLM: failed for %s; skipping item", cache_keys[0])
            stats["skipped"] += 1
            continue

        t["llm"] = result
        cached_at = _now_utc_iso()
        t["_llm_cached_at"] = cached_at
        _save_cache(cache_keys[0], result, cache_dir, cached_at=cached_at)
        stats["requested"] += 1
        daily_used += 1
        if remaining is not None:
            remaining -= 1

    if daily_cap > 0:
        _save_daily_usage(cache_dir, today, daily_used)
        stats["daily_remaining"] = max(0, daily_cap - daily_used)
    return stats
