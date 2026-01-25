from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from dateutil import parser as dtparser

from config import CACHE_DIR
from connectors.open_canada_ckan import CKANClient

CANADABUYS_TENDER_NOTICES_DATASET_ID = "6abd20d4-7a1c-4b38-baa2-9525d0bb2fd2"
CACHE_DIR = CACHE_DIR / "canadabuys"


def _norm_key(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _pick_field(row: Dict[str, str], candidates: List[str]) -> Optional[str]:
    """
    Returns the value of the first matching field among candidates.
    Matching is robust to punctuation/case differences.
    """
    norm_map = {_norm_key(k): k for k in row.keys()}
    for c in candidates:
        nk = _norm_key(c)
        if nk in norm_map:
            val = row.get(norm_map[nk], "")
            return val.strip() if isinstance(val, str) else val
    return None


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = dtparser.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _resource_score(name: str, prefer: str) -> int:
    """
    prefer: "open" | "new" | "all"
    """
    n = (name or "").lower()
    score = 0

    # baseline: must look like the tender notice dataset
    if "tender" in n and "notice" in n:
        score += 10

    # prefer the variant you want
    if prefer == "open":
        if "open" in n:
            score += 100
        if "new" in n:
            score += 50
    elif prefer == "new":
        if "new" in n:
            score += 100
        if "open" in n:
            score += 50
    elif prefer == "all":
        if "all" in n:
            score += 100

    # small bumps
    if "english" in n or "en" in n:
        score += 5
    if "french" in n or "fr" in n:
        score += 5

    return score


def _choose_best_resource(resources: list[dict], prefer: str = "open") -> dict:
    candidates = []
    for r in resources:
        fmt = (r.get("format") or "").lower()
        title = r.get("name") or r.get("title") or ""
        if fmt in {"csv", "zip"} and "tender" in title.lower():
            candidates.append(r)

    if not candidates:
        raise RuntimeError("No suitable CSV/ZIP resources found in dataset.")

    candidates.sort(
        key=lambda r: _resource_score((r.get("name") or r.get("title") or ""), prefer),
        reverse=True,
    )
    return candidates[0]



def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "sap-tender-bot/0.1"}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest


def _iter_csv_rows_from_file(path: Path) -> Iterable[Dict[str, str]]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            # take first CSV inside
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise RuntimeError("ZIP contained no CSV files.")
            with zf.open(csv_names[0], "r") as fp:
                text = io.TextIOWrapper(fp, encoding="utf-8", errors="replace")
                yield from csv.DictReader(text)
    else:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fp:
            yield from csv.DictReader(fp)


def _first_nonempty(*vals: str) -> str:
    for v in vals:
        if v and v.strip():
            return v.strip()
    return ""

def _split_multiline(s: str) -> list[str]:
    if not s:
        return []
    # values often look like "*40141600\n*78142000"
    parts = []
    for line in s.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        parts.append(line)
    return parts

def _parse_codes_starred(s: str) -> list[str]:
    codes = []
    for p in _split_multiline(s):
        p = p.lstrip("*").strip()
        # keep only digits
        digits = "".join(ch for ch in p if ch.isdigit())
        if digits:
            codes.append(digits)
    return codes

def _pick_best_url(row: dict[str, str]) -> str:
    url = _first_nonempty(row.get("noticeURL-URLavis-eng",""), row.get("noticeURL-URLavis-fra",""))
    if url:
        return url

    # fallback: attachment field usually contains direct PDF links
    att = _first_nonempty(row.get("attachment-piecesJointes-eng",""), row.get("attachment-piecesJointes-fra",""))
    for p in _split_multiline(att):
        if p.startswith("http"):
            return p.strip()
    return ""

def _normalize_row(row: dict[str, str]) -> dict[str, object]:
    ref = (row.get("referenceNumber-numeroReference") or "").strip()
    amend_no = (row.get("amendmentNumber-numeroModification") or "").strip() or "0"
    solic = (row.get("solicitationNumber-numeroSollicitation") or "").strip()

    pub_dt = _parse_dt((row.get("publicationDate-datePublication") or "").strip())
    amend_dt = _parse_dt((row.get("amendmentDate-dateModification") or "").strip())
    close_dt = _parse_dt((row.get("tenderClosingDate-appelOffresDateCloture") or "").strip())

    updated_dt = None
    for d in (pub_dt, amend_dt):
        if d and (updated_dt is None or d > updated_dt):
            updated_dt = d

    title_en = (row.get("title-titre-eng") or "").strip()
    title_fr = (row.get("title-titre-fra") or "").strip()
    desc_en = (row.get("tenderDescription-descriptionAppelOffres-eng") or "").strip()
    desc_fr = (row.get("tenderDescription-descriptionAppelOffres-fra") or "").strip()

    status_en = (row.get("tenderStatus-appelOffresStatut-eng") or "").strip()
    status_fr = (row.get("tenderStatus-appelOffresStatut-fra") or "").strip()

    org_en = (row.get("contractingEntityName-nomEntitContractante-eng") or "").strip()
    org_fr = (row.get("contractingEntityName-nomEntitContractante-fra") or "").strip()

    url = _pick_best_url(row)

    unspsc_codes = _parse_codes_starred(row.get("unspsc","") or "")
    gsin_codes = _parse_codes_starred(row.get("gsin-nibs","") or "")

    # include amendment number so amendments re-trigger
    uid = f"canadabuys:{ref}:am{amend_no}" if ref else f"canadabuys:{url}:am{amend_no}"

    return {
        "uid": uid,
        "source": "canadabuys",
        "id": ref,
        "amendment_number": amend_no,
        "solicitation_number": solic,

        "title": _first_nonempty(title_en, title_fr),
        "title_en": title_en,
        "title_fr": title_fr,

        "description": _first_nonempty(desc_en, desc_fr),
        "description_en": desc_en,
        "description_fr": desc_fr,

        "org": _first_nonempty(org_en, org_fr),
        "org_en": org_en,
        "org_fr": org_fr,

        "status_en": status_en,
        "status_fr": status_fr,

        "publish_date": pub_dt.isoformat() if pub_dt else None,
        "amendment_date": amend_dt.isoformat() if amend_dt else None,
        "updated_at": updated_dt.isoformat() if updated_dt else None,
        "close_date": close_dt.isoformat() if close_dt else None,

        "url": url,
        "url_en": (row.get("noticeURL-URLavis-eng") or "").strip(),
        "url_fr": (row.get("noticeURL-URLavis-fra") or "").strip(),

        "unspsc": unspsc_codes,  # list[str]
        "unspsc_desc_en": (row.get("unspscDescription-eng") or "").strip(),
        "unspsc_desc_fr": (row.get("unspscDescription-fra") or "").strip(),

        "gsin": gsin_codes,      # list[str]
        "gsin_desc_en": (row.get("gsinDescription-nibsDescription-eng") or "").strip(),
        "gsin_desc_fr": (row.get("gsinDescription-nibsDescription-fra") or "").strip(),

        "procurement_category": (row.get("procurementCategory-categorieApprovisionnement") or "").strip(),
        "notice_type_en": (row.get("noticeType-avisType-eng") or "").strip(),
        "notice_type_fr": (row.get("noticeType-avisType-fra") or "").strip(),
        "proc_method_en": (row.get("procurementMethod-methodeApprovisionnement-eng") or "").strip(),
        "proc_method_fr": (row.get("procurementMethod-methodeApprovisionnement-fra") or "").strip(),

        "_raw": row,
    }


@dataclass
class CanadaBuysTenderNoticesConnector:
    ckan: CKANClient = field(default_factory=CKANClient)

    def fetch_open_tender_notices(self, since_iso: Optional[str] = None) -> List[Dict[str, object]]:
        pkg = self.ckan.package_show(CANADABUYS_TENDER_NOTICES_DATASET_ID)
        res = _choose_best_resource(pkg.get("resources", []), prefer="open")

        resource_url = res.get("url")
        if not resource_url:
            raise RuntimeError("Chosen resource has no URL.")

        fmt = (res.get("format") or "").lower()
        filename = f"new_tender_notices.{fmt if fmt in {'csv','zip'} else 'csv'}"
        path = CACHE_DIR / filename

        _download(resource_url, path)

        since_dt = _parse_dt(since_iso) if since_iso else None

        out = []
        for row in _iter_csv_rows_from_file(path):
            item = _normalize_row(row)

            # enforce OPEN (belt + suspenders)
            status_ok = (item.get("status_en","").lower() == "open") or (item.get("status_fr","").lower() == "ouvert")
            if not status_ok:
                continue

            if since_dt:
                upd = _parse_dt(item.get("updated_at"))
                if not upd or upd < since_dt:
                    continue

            out.append(item)

        return out

