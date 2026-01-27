from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests

from sap_tender_bot.config import EmbeddingsConfig

_APOSTROPHES = {
    "’": "'",
    "‘": "'",
    "‛": "'",
    "ʼ": "'",
    "`": "'",
    "´": "'",
}


INTENT_BUCKETS: list[dict[str, object]] = [
    {
        "name": "ERP modernization",
        "description": "Core ERP replacement, upgrade, or migration programs.",
        "keywords": [
            "enterprise resource planning",
            "erp modernization",
            "erp migration",
            "erp replacement",
            "s/4hana",
            "sap ecc",
            "financial system replacement",
            "general ledger",
            "accounts payable",
            "accounts receivable",
        ],
        "prototypes": [
            "ERP modernization and migration to S/4HANA",
            "Enterprise business system replacement for finance",
            "Upgrade SAP ECC to S/4HANA with remediation",
        ],
    },
    {
        "name": "Procurement",
        "description": "Source-to-pay, procure-to-pay, e-procurement platforms.",
        "keywords": [
            "procure-to-pay",
            "source-to-pay",
            "e-procurement",
            "procurement system",
            "supplier management",
            "ariba",
            "sap business network",
            "supply chain management",
        ],
        "prototypes": [
            "Implement a source-to-pay platform",
            "E-procurement system replacement",
            "Procurement modernization with supplier onboarding",
        ],
    },
    {
        "name": "HR/payroll",
        "description": "HRIS, HCM, payroll modernization and replacement.",
        "keywords": [
            "human capital management",
            "hcm",
            "hris",
            "payroll system",
            "pension payroll",
            "successfactors",
            "workday",
        ],
        "prototypes": [
            "HRIS and payroll system modernization",
            "HCM implementation and payroll replacement",
            "SuccessFactors rollout",
        ],
    },
    {
        "name": "Integration/data",
        "description": "ERP integration, data migration, middleware, and APIs.",
        "keywords": [
            "system integration",
            "integration services",
            "middleware",
            "sap cpi",
            "api integration",
            "data migration",
            "data conversion",
            "master data",
            "sap btp",
        ],
        "prototypes": [
            "ERP integration services and data migration",
            "Middleware implementation for ERP integration",
            "API and data conversion for enterprise systems",
        ],
    },
    {
        "name": "AMS",
        "description": "Application managed services and operational support.",
        "keywords": [
            "application managed services",
            "ams",
            "sap support",
            "production support",
            "maintenance services",
            "basis support",
            "managed services",
        ],
        "prototypes": [
            "SAP application managed services",
            "ERP operational support and maintenance",
            "Managed services for SAP landscape",
        ],
    },
]


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    for src, dst in _APOSTROPHES.items():
        text = text.replace(src, dst)
    text = text.lower()
    return " ".join(text.split())


def _tender_text(t: dict) -> str:
    parts = [
        t.get("title", ""),
        t.get("title_en", ""),
        t.get("title_fr", ""),
        t.get("description_en", ""),
        t.get("description_fr", ""),
        t.get("unspsc_desc_en", ""),
        t.get("unspsc_desc_fr", ""),
        t.get("gsin_desc_en", ""),
        t.get("gsin_desc_fr", ""),
        t.get("notice_type_en", ""),
        t.get("notice_type_fr", ""),
        t.get("proc_method_en", ""),
        t.get("proc_method_fr", ""),
    ]
    return _normalize_text(" ".join(str(p) for p in parts if p))


def _bucket_text(bucket: dict) -> str:
    parts: list[str] = [str(bucket.get("name", "")), str(bucket.get("description", ""))]
    parts.extend(bucket.get("keywords", []) or [])
    parts.extend(bucket.get("prototypes", []) or [])
    return _normalize_text(" ".join(str(p) for p in parts if p))


def _keyword_similarity(text: str, bucket: dict) -> float:
    text_norm = _normalize_text(text)
    keywords = [str(k) for k in (bucket.get("keywords") or [])]
    if not keywords:
        return 0.0
    hits = 0
    for term in keywords:
        term_norm = _normalize_text(term)
        if term_norm and term_norm in text_norm:
            hits += 1
    return min(1.0, hits / max(1, len(keywords)))


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path(key: str, model: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(f"{model}:{key}".encode("utf-8")).hexdigest()
    return cache_dir / "embeddings" / f"{digest}.json"


def _load_cached_embedding(key: str, text: str, config: EmbeddingsConfig, cache_dir: Path):
    path = _cache_path(key, config.model, cache_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("model") != config.model:
        return None
    if payload.get("text_hash") != _text_hash(text):
        return None
    embedding = payload.get("embedding")
    if isinstance(embedding, list):
        return embedding
    return None


def _save_cached_embedding(
    key: str,
    text: str,
    embedding: list[float],
    config: EmbeddingsConfig,
    cache_dir: Path,
) -> None:
    path = _cache_path(key, config.model, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": key,
        "model": config.model,
        "text_hash": _text_hash(text),
        "embedding": embedding,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _extract_embeddings(data: dict) -> list[list[float]]:
    embeddings = data.get("embeddings")
    if isinstance(embeddings, dict):
        for key in ("float", "embeddings", "vector"):
            if key in embeddings:
                embeddings = embeddings[key]
                break
    if embeddings is None and isinstance(data.get("data"), list):
        embeddings = [row.get("embedding") for row in data["data"]]
    if not isinstance(embeddings, list):
        raise ValueError("Embeddings missing from response")
    return embeddings


def _cohere_embed(texts: list[str], config: EmbeddingsConfig) -> list[list[float]]:
    base = config.api_base.rstrip("/")
    bases = [base]
    if "api.cohere.com" in base:
        bases.append(base.replace("api.cohere.com", "api.cohere.ai"))
    elif "api.cohere.ai" in base:
        bases.append(base.replace("api.cohere.ai", "api.cohere.com"))

    urls: list[str] = []
    for b in bases:
        if b.endswith("/v2"):
            urls.append(f"{b}/embed")
        elif b.endswith("/v1"):
            urls.append(f"{b}/embed")
        else:
            urls.append(f"{b}/v2/embed")
            urls.append(f"{b}/v1/embed")
    urls = list(dict.fromkeys(urls))
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload_v2: dict[str, object] = {
        "model": config.model,
        "texts": texts,
        "truncate": "END",
    }
    if config.input_type:
        payload_v2["input_type"] = config.input_type
    payload_v1: dict[str, object] = {
        "model": config.model,
        "texts": texts,
        "truncate": "END",
    }

    last_err: Exception | None = None
    for url in urls:
        payload = payload_v2 if "/v2/" in url else payload_v1
        for attempt in range(4):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=config.timeout_s)
                if resp.status_code == 404 and url != urls[-1]:
                    break
                if resp.status_code in {429, 500, 502, 503, 504}:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_s = int(retry_after) if retry_after and retry_after.isdigit() else 2 + attempt * 2
                    time.sleep(sleep_s)
                    continue
                if resp.status_code >= 400:
                    detail = resp.text.strip()
                    msg = f"HTTP {resp.status_code} from {url}"
                    if detail:
                        msg = f"{msg} | {detail[:400]}"
                    raise RuntimeError(msg)
                data = resp.json()
                return _extract_embeddings(data)
            except Exception as exc:
                last_err = exc
                time.sleep(1 + attempt * 2)
    raise RuntimeError(f"Embedding request failed after retries: {last_err}")


def embed_texts(
    texts_by_key: Dict[str, str],
    config: EmbeddingsConfig,
    cache_dir: Path,
    logger=None,
) -> tuple[Dict[str, list[float]], bool]:
    if not config.enabled or not config.api_key:
        if logger:
            logger.info("Embeddings disabled or COHERE_EMBED_API_KEY missing; using keyword fallback.")
        return {}, False
    if not cache_dir:
        if logger:
            logger.info("No cache_dir provided; using keyword fallback.")
        return {}, False

    cached: dict[str, list[float]] = {}
    missing: list[tuple[str, str]] = []
    for key, text in texts_by_key.items():
        embedding = _load_cached_embedding(key, text, config, cache_dir)
        if embedding is not None:
            cached[key] = embedding
        else:
            missing.append((key, text))

    if not missing:
        return cached, True

    batch_size = max(1, int(config.batch_size))
    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        keys = [k for k, _ in batch]
        texts = [t for _, t in batch]
        try:
            embeddings = _cohere_embed(texts, config=config)
        except Exception as exc:
            if logger:
                logger.warning("Embeddings request failed; falling back to keyword scoring: %s", exc)
            return cached, False
        for key, text, embedding in zip(keys, texts, embeddings, strict=False):
            if isinstance(embedding, list):
                cached[key] = embedding
                _save_cached_embedding(key, text, embedding, config, cache_dir)

    return cached, True


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b, strict=False):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / math.sqrt(norm_a * norm_b)))


def _evidence_score(evidence: dict) -> int:
    erp_hits = evidence.get("erp_hits") or []
    transform_hits = evidence.get("transform_hits") or []
    score = 0
    score += 8 if evidence.get("unspsc_it") else 0
    score += min(18, 6 * len(erp_hits))
    score += min(15, 5 * len(transform_hits))
    score += 6 if evidence.get("allowlist") else 0
    return score


def _score_similarity(similarity: float, evidence: dict) -> int:
    sim_score = int(round(similarity * 100))
    return min(120, sim_score + _evidence_score(evidence))


def semantic_rank_tenders(
    tenders: Iterable[dict],
    *,
    config: EmbeddingsConfig | None,
    cache_dir: Path | None,
    logger=None,
) -> None:
    tenders = list(tenders)
    if not tenders:
        return
    config = config or EmbeddingsConfig()

    bucket_texts = {f"bucket:{b['name']}": _bucket_text(b) for b in INTENT_BUCKETS}
    bucket_embeddings, buckets_embedded = embed_texts(bucket_texts, config, cache_dir, logger=logger)

    tender_texts: dict[str, str] = {}
    for idx, t in enumerate(tenders):
        key = str(t.get("uid") or t.get("tender_key") or f"tender:{idx}")
        tender_texts[key] = _tender_text(t)
        t["_semantic_key"] = key

    tender_embeddings, tenders_embedded = embed_texts(tender_texts, config, cache_dir, logger=logger)
    use_embeddings = buckets_embedded and tenders_embedded and bucket_embeddings and tender_embeddings

    for t in tenders:
        key = t.get("_semantic_key")
        text = tender_texts.get(key, "")
        evidence = dict(t.get("_evidence") or {})

        best_bucket = "unknown"
        best_similarity = 0.0

        if use_embeddings and key in tender_embeddings:
            tender_vec = tender_embeddings[key]
            for bucket in INTENT_BUCKETS:
                bucket_key = f"bucket:{bucket['name']}"
                bucket_vec = bucket_embeddings.get(bucket_key)
                if not bucket_vec:
                    continue
                sim = _cosine_similarity(tender_vec, bucket_vec)
                if sim > best_similarity:
                    best_similarity = sim
                    best_bucket = str(bucket.get("name") or "unknown")
        else:
            for bucket in INTENT_BUCKETS:
                sim = _keyword_similarity(text, bucket)
                if sim > best_similarity:
                    best_similarity = sim
                    best_bucket = str(bucket.get("name") or "unknown")

        semantic_score = _score_similarity(best_similarity, evidence)
        t["semantic_bucket"] = best_bucket
        t["semantic_similarity"] = round(best_similarity, 4)
        t["semantic_score"] = semantic_score
        t["semantic_evidence"] = evidence
        t["semantic_used_embeddings"] = bool(use_embeddings)
        t.pop("_semantic_key", None)
