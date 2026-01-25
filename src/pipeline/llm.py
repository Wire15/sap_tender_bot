from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import requests

from config import CACHE_DIR

LLM_CACHE_DIR = CACHE_DIR / "llm"

DEFAULT_SCHEMA = {
    "relevance": "high|medium|low",
    "reasoning_bullets": [],
    "opportunity_type": [],
    "sap_products": [],
    "entities": {"buyer_org": "", "program_name": ""},
    "red_flags": [],
    "recommended_next_steps": [],
}


def _cache_path(uid: str) -> Path:
    digest = hashlib.sha256(uid.encode("utf-8")).hexdigest()
    return LLM_CACHE_DIR / f"{digest}.json"


def _load_cache(uid: str) -> Dict[str, Any] | None:
    path = _cache_path(uid)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache(uid: str, data: Dict[str, Any]) -> None:
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(uid)
    payload = {"uid": uid, "llm": data}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_prompt(t: Dict[str, Any]) -> str:
    title = t.get("title") or ""
    desc = t.get("description") or ""
    org = t.get("org") or ""
    close_date = t.get("close_date") or ""
    updated_at = t.get("updated_at") or ""
    unspsc = ", ".join(t.get("unspsc_desc_en", "").splitlines())
    gsin = ", ".join(t.get("gsin_desc_en", "").splitlines())

    return (
        "You are analyzing a Canadian public sector tender for SAP/ERP relevance. "
        "Return ONLY valid JSON matching this schema keys: "
        "relevance, reasoning_bullets, opportunity_type, sap_products, entities, red_flags, "
        "recommended_next_steps. No markdown, no extra text.\n\n"
        f"TITLE: {title}\n"
        f"BUYER ORG: {org}\n"
        f"UPDATED_AT: {updated_at}\n"
        f"CLOSE_DATE: {close_date}\n"
        f"UNSPSC_DESC: {unspsc}\n"
        f"GSIN_DESC: {gsin}\n"
        f"DESCRIPTION: {desc}\n"
    )


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty LLM response")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def _cohere_chat(prompt: str, api_key: str) -> Dict[str, Any]:
    base_url = os.environ.get("COHERE_API_BASE", "https://api.cohere.ai")
    model = os.environ.get("COHERE_MODEL", "command-r")
    timeout_s = int(os.environ.get("COHERE_TIMEOUT", "45"))

    url = f"{base_url.rstrip('/')}/v1/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "message": prompt,
        "temperature": 0.2,
        "max_tokens": 400,
    }

    last_err: Exception | None = None
    for attempt in range(4):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code in {429, 500, 502, 503, 504}:
                retry_after = resp.headers.get("Retry-After")
                sleep_s = int(retry_after) if retry_after and retry_after.isdigit() else 2 + attempt * 2
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data.get("text")
            if not text and "generations" in data and data["generations"]:
                text = data["generations"][0].get("text")
            if not text and "message" in data:
                text = data.get("message")
            return _extract_json(text)
        except Exception as exc:
            last_err = exc
            time.sleep(1 + attempt * 2)

    raise RuntimeError(f"Cohere request failed after retries: {last_err}")


def enrich_tenders_with_llm(tenders: Iterable[Dict[str, Any]], logger=None) -> None:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        if logger:
            logger.warning("COHERE_API_KEY missing; skipping LLM enrichment")
        return

    for t in tenders:
        uid = t.get("uid")
        if not uid:
            continue

        cached = _load_cache(uid)
        if cached and cached.get("llm"):
            t["llm"] = cached["llm"]
            continue

        prompt = _build_prompt(t)
        if logger:
            logger.info("LLM: requesting summary for %s", uid)

        result = _cohere_chat(prompt, api_key=api_key)
        for key in DEFAULT_SCHEMA:
            result.setdefault(key, DEFAULT_SCHEMA[key])

        t["llm"] = result
        _save_cache(uid, result)
