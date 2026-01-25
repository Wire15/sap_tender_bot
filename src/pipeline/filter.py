from __future__ import annotations

"""
SAP Tender Bot - Filtering & Scoring

This module is intentionally conservative: it aims to surface SAP / ERP transformation
opportunities (and closely related enterprise-app modernization work) while filtering out
high-volume noise like:
- staffing/resource augmentation ("niveau 2", "programmeur", etc.)
- standing offers / supply arrangements (SPICT/SPICS/DAMA/TSPS/etc.)
- goods/hardware (laptops, peripherals, valves, furniture, etc.)
- non-IT services (janitorial, plumbing, medical services, etc.)

It returns:
- kept: list of tenders with a numeric "score"
- reasons: Counter of rejection reasons for debugging

You can tune behavior via the toggles in the CONFIG section below.
"""

import re  # noqa: E402
import unicodedata  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Dict, List, Pattern, Tuple  # noqa: E402

# ---------------------------
# CONFIG (tune as you like)
# ---------------------------

# Include tenders that are supply arrangements / standing offers?
INCLUDE_SUPPLY_ARRANGEMENTS = False

# Include staffing/resource-augmentation style tenders?
INCLUDE_STAFFING = False

# If a non-SAP tender closes very far in the future, it is usually a standing offer.
MAX_NON_SAP_CLOSE_DAYS = 365

# Minimum score to keep (for non-SAP-direct tenders). SAP-direct always kept.
MIN_SCORE_NON_SAP = 70

# Store hit diagnostics on the tender dict (adds _hits key).
STORE_HITS = True


# ---------------------------
# Helpers: normalization + regex compilation
# ---------------------------

_APOSTROPHES = {
    "’": "'",
    "‘": "'",
    "‛": "'",
    "ʼ": "'",
    "`": "'",
    "´": "'",
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    for src, dst in _APOSTROPHES.items():
        text = text.replace(src, dst)
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compile_terms(terms: list[str]) -> list[tuple[str, Pattern]]:
    """
    Compile a list of terms into regex patterns.

    - For short abbreviations like "erp", "hcm", "iam", enforce word boundaries.
      This prevents accidental substring matches inside longer words/URLs.
    - For phrases (contain spaces), use simple substring regex (escaped).
    """
    pats: list[tuple[str, Pattern]] = []
    for t in terms:
        t_norm = normalize_text((t or "").strip())
        if not t_norm:
            continue

        if re.fullmatch(r"[a-z0-9.-]{2,8}", t_norm) and " " not in t_norm:
            pats.append((t_norm, re.compile(rf"\b{re.escape(t_norm)}\b", re.IGNORECASE)))
        else:
            pats.append((t_norm, re.compile(re.escape(t_norm), re.IGNORECASE)))
    return pats


def find_matches(text: str, patterns: list[tuple[str, Pattern]]) -> list[str]:
    hits: list[str] = []
    for term, pat in patterns:
        if pat.search(text):
            hits.append(term)
    return hits


# ---------------------------
# Term dictionaries
# ---------------------------

# High-confidence SAP signals (explicit SAP ecosystem keywords)
SAP_DIRECT_TERMS = [
    "sap",
    "abap",
    "s/4hana",
    "s4hana",
    "ecc",
    "fiori",
    "successfactors",
    "ariba",
    "concur",
    "bw/4hana",
    "businessobjects",
    "sap btp",
    "sap cpi",
    "cloud platform integration",
    "sap grc",
]

# ERP/business-platform domain signals (keep these fairly specific)
ERP_DOMAIN_TERMS = [
    # English
    "enterprise resource planning",
    "erp",
    "financial management system",
    "financial system replacement",
    "general ledger",
    "accounts payable",
    "accounts receivable",
    "procure-to-pay",
    "source-to-pay",
    "e-procurement",
    "procurement system",
    "hris",
    "human capital management",
    "hcm",
    "payroll system",
    "pension payroll",
    "asset management system",
    "inventory management system",

    # French (normalized to ASCII)
    "progiciel de gestion integre",
    "pgi",
    "systeme de gestion financiere",
    "systeme financier",
    "systeme de paie",
    "gestion des approvisionnements",
    "approvisionnement electronique",
    "gestion des ressources humaines",
    "systeme de gestion des ressources humaines",
    "grand livre",
    "comptes crediteurs",
    "comptes debiteurs",
]

# Transformation intent (requires implementation/upgrade/migration/replacement...)
TRANSFORM_TERMS = [
    # English
    "implementation",
    "implement",
    "upgrade",
    "migration",
    "modernization",
    "modernisation",
    "replacement",
    "replace",
    "transformation",
    "rollout",
    "deployment",
    "configuration",
    "integrat",
    "data conversion",
    "system integration",

    # French (normalized to ASCII)
    "mise en oeuvre",
    "implantation",
    "deploiement",
    "migration",
    "modernisation",
    "remplacement",
    "integrat",
    "conversion de donnees",
]

# IT context (avoid generic "application"/"data"/"solution" - too many false positives)
IT_CONTEXT_TERMS = [
    # English
    "enterprise application",
    "cots",
    "saas",
    "software-as-a-service",
    "systems integrator",
    "systems integration",
    "integration services",
    "middleware",
    "api",
    "interface",
    "interoperability",
    "identity and access management",
    "iam",
    "single sign-on",
    "sso",
    "erp implementation",
    "erp upgrade",
    "erp migration",

    # French (normalized to ASCII)
    "systeme d'information",
    "integration de systemes",
    "services d'integration",
    "progiciel",
    "solution infonuagique",
    "infonuagique",
    "gestion des identites et des acces",
]

SAP_DIRECT = compile_terms(SAP_DIRECT_TERMS)
ERP_DOMAIN = compile_terms(ERP_DOMAIN_TERMS)
TRANSFORM = compile_terms(TRANSFORM_TERMS)
IT_CONTEXT = compile_terms(IT_CONTEXT_TERMS)

# Allowlists are intentionally TITLE-ONLY to avoid boilerplate in descriptions.
ERP_TITLE_ALLOWLIST_TERMS = [
    # ERP domain phrases (title only)
    "enterprise resource planning",
    "erp",
    "enterprise business system",
    "enterprise business systems",
    "business system replacement",
    "financial management system",
    "financial system",
    "financial system replacement",
    "general ledger",
    "accounts payable",
    "accounts receivable",
    "procure-to-pay",
    "source-to-pay",
    "e-procurement",
    "procurement system",
    "hris",
    "human resources information system",
    "human capital management",
    "payroll system",
    "pension payroll",
    "asset management system",
    "inventory management system",

    # French
    "progiciel de gestion integre",
    "pgi",
    "systeme de gestion financiere",
    "systeme financier",
    "systeme de paie",
    "systeme de gestion des ressources humaines",
    "gestion des ressources humaines",
    "approvisionnement electronique",
    "gestion des approvisionnements",
]

ERP_TITLE_ALLOWLIST = compile_terms(ERP_TITLE_ALLOWLIST_TERMS)

ERP_PRODUCT_ALLOWLIST_TERMS = [
    # SAP
    "sap s/4hana",
    "s/4hana",
    "sap ecc",
    "sap erp",
    "sap bw",
    "successfactors",
    "concur",
    "ariba",
    # Oracle
    "oracle erp",
    "oracle e-business",
    "oracle ebs",
    "oracle fusion",
    "oracle financials",
    # Microsoft
    "microsoft dynamics",
    "dynamics 365",
    # Other ERP suites
    "workday",
    "peoplesoft",
    "jd edwards",
    "netsuite",
    "infor",
    "unit4",
    "ifs",
    "sage x3",
]

ERP_PRODUCT_ALLOWLIST = compile_terms(ERP_PRODUCT_ALLOWLIST_TERMS)

SAP_PORTAL_CONTEXT_TERMS = [
    "sap ariba",
    "ariba discovery",
    "ariba account",
    "ariba network",
    "sap business network",
    "business network discovery",
    "ariba discovery posting",
    "sap business network discovery",
    "supplier portal",
    "submit bids",
    "bids received",
    "submit your bid",
    "submission via",
    "login",
    "tendering portal",
    "government of canada profile",
    "buyandsell",
    "achatsetventes",
    "achatscanada",
    "canadabuys",
]

SAP_PORTAL_CONTEXT = compile_terms(SAP_PORTAL_CONTEXT_TERMS)


# ---------------------------
# Noise filters
# ---------------------------

SUPPLY_ARRANGEMENT_TERMS = [
    "spict",
    "spics",
    "spts",
    "tsps",
    "tbips",
    "sbips",
    "dama",
    "rfsa",
    "rfso",
    "supply arrangement",
    "standing offer",
    "offer to supply arrangement",
    "offre a commandes",
    "offre a commande",
    "arrangement en matiere d'approvisionnement",
    "arrangement en matiere dapprovisionnement",
    "repertoire ouvert",
    "liste de fournisseurs",
    "demande d'arrangement",
    "services d'aide temporaire",
    "services daide temporaire",
]

STAFFING_TITLE_PATTERNS = [
    r"\bniveau[\s-]*[12345]\b",
    r"\bprogrammeur\b",
    r"\banalyste\b",
    r"\badministrateur\b",
    r"\bconseiller\b",
    r"\barchitecte\b",
    r"\bdeveloper\b",
    r"\bdeveloppeur\b",
    r"\bresource\b",
    r"\bressource\b",
    r"\bstaffing\b",
    r"\baugmentation\b",
]

HARDWARE_TERMS = [
    "laptop",
    "ordinateur portable",
    "peripheral",
    "peripherique",
    "boitier",
    "diplexeur",
    "supports de rack",
    "kits radio",
    "valve",
    "clapet",
    "telecommunications batteries",
    "batteries de telecommunications",
    "printer",
    "scanner",
    "copier",
    "ink cartridge",
    "toner",
    "paper",
    "fournitures",
    "equipement",
    "equipment",
    "mobilier",
    "chaises",
    "fauteuils",
    "furniture",
    "chairs",
]

SUPPLY_ARRANGEMENT = compile_terms(SUPPLY_ARRANGEMENT_TERMS)
HARDWARE = compile_terms(HARDWARE_TERMS)

NEGATIVE_TITLE_TERMS = [
    # construction / facilities
    "construction",
    "renovation",
    "rehabilitation",
    "rehabiliter",
    "reparation",
    "toiture",
    "roof",
    "pont",
    "bridge",
    "batiment",
    "building",
    "infrastructure",
    "centre d'accueil",
    "visitor centre",
    "lieu historique",
    "historical site",
    "parc national",
    "national park",
    "repertoire ouvert",

    # facilities / maintenance
    "maintenance",
    "entretien",
    "hvac",
    "elevator",
    "plumbing",
    "electrical",
    "mechanical",
    "boiler",
    "deneigement",
    "de-neigement",
    "landscap",

    # goods / lab / medical
    "vaccin",
    "laboratoire",
    "laboratory",
    "microscope",
    "medical",
    "dental",
    "clinic",
    "fournitures",
    "equipement",
    "equipment",
    "mobilier",
    "chaises",
    "furniture",

    # transport / logistics
    "transport",
    "autobus",
    "camion",
    "truck",
    "vehicle",
    "fleet",
    "messagerie",
    "courier",
    "air",
    "aerien",

    # cleaning / waste / pest control
    "janitorial",
    "nettoyage",
    "cleaning",
    "dechets",
    "waste",
    "recycling",
    "pesticides",
    "fumigation",

    # banking services
    "services bancaires",
    "banking services",
]

NEGATIVE_TITLE = compile_terms(NEGATIVE_TITLE_TERMS)


# ---------------------------
# Tender field helpers
# ---------------------------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _text_blob(t: Dict[str, object]) -> str:
    """
    Only include human-language fields and classification descriptors.
    Do NOT include URLs or attachment filenames here.
    """
    parts = [
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
    return normalize_text(" ".join(str(p) for p in parts if p))


def _looks_like_supply_arrangement(text: str) -> bool:
    return bool(find_matches(text, SUPPLY_ARRANGEMENT))


def _looks_like_staffing(title: str) -> bool:
    title_norm = normalize_text(title or "")
    return any(re.search(p, title_norm) for p in STAFFING_TITLE_PATTERNS)


def _looks_like_negative_title(title: str) -> bool:
    title_norm = normalize_text(title or "")
    return bool(find_matches(title_norm, NEGATIVE_TITLE))


def _is_goods_or_hardware(t: Dict[str, object], text: str) -> bool:
    # CanadaBuys uses procurementCategory; "GD" appears in many goods postings
    proc_cat = (t.get("procurement_category", "") or "").upper()
    if "GD" in proc_cat:
        return True
    return bool(find_matches(text, HARDWARE))


def _close_date_far_future(t: Dict[str, object], days: int) -> bool:
    close_iso = t.get("close_date")
    if not close_iso:
        return False
    try:
        close_dt = datetime.fromisoformat(str(close_iso).replace("Z", "+00:00"))
        return (close_dt - _now_utc()).days > days
    except Exception:
        return False


def is_it_unspsc(t: dict) -> bool:
    """
    Heuristic: UNSPSC 43xxxxxx = IT/Telecom.
    Also allow common IT/pro services classes used in government procurement.
    """
    codes = t.get("unspsc") or []
    if isinstance(codes, str):
        codes = [codes]

    it_prefixes = ("43", "432", "4321", "4322", "4323", "8111", "8112", "8113")
    return any(str(c).startswith(it_prefixes) for c in codes if c)


def is_non_it_unspsc(t: dict) -> bool:
    """
    High-signal non-IT prefixes (construction, janitorial, transport, health, etc.)
    If tender is SAP-direct, we ignore this.
    """
    codes = t.get("unspsc") or []
    if isinstance(codes, str):
        codes = [codes]

    bad_prefixes = ("72", "73", "76", "78", "85", "90")  # keep 81 allowed (too broad)
    return any(str(c).startswith(bad_prefixes) for c in codes if c)


# ---------------------------
# Main filter
# ---------------------------


def score_and_filter(
    tenders: List[Dict[str, object]],
    return_rejected: bool = False,
    config: Dict[str, object] | None = None,
) -> Tuple[List[Dict[str, object]], Counter] | Tuple[List[Dict[str, object]], Counter, List[Dict[str, object]]]:
    config = config or {}
    include_supply_arrangements = bool(
        config.get("include_supply_arrangements", INCLUDE_SUPPLY_ARRANGEMENTS)
    )
    include_staffing = bool(config.get("include_staffing", INCLUDE_STAFFING))
    max_non_sap_close_days = int(config.get("max_non_sap_close_days", MAX_NON_SAP_CLOSE_DAYS))
    min_score_non_sap = int(config.get("min_score_non_sap", MIN_SCORE_NON_SAP))
    store_hits = bool(config.get("store_hits", STORE_HITS))

    kept: List[Dict[str, object]] = []
    reasons: Counter = Counter()
    rejected: List[Dict[str, object]] = []

    for t in tenders:
        text = _text_blob(t)
        title = str(t.get("title", "") or "")

        sap_hits = find_matches(text, SAP_DIRECT)
        erp_hits = find_matches(text, ERP_DOMAIN)
        xform_hits = find_matches(text, TRANSFORM)
        it_hits = find_matches(text, IT_CONTEXT)
        title_norm = normalize_text(title or "")
        allowlist_product_hits = find_matches(title_norm, ERP_PRODUCT_ALLOWLIST)
        allowlist_title_hits = find_matches(title_norm, ERP_TITLE_ALLOWLIST)
        allowlist_any = bool(allowlist_product_hits or allowlist_title_hits)
        portal_hits = find_matches(text, SAP_PORTAL_CONTEXT)

        sap_direct = bool(sap_hits)

        if sap_direct:
            sap_terms = set(sap_hits)
            if sap_terms.issubset({"sap", "ariba"}):
                if portal_hits and "sap" not in title_norm and "ariba" not in title_norm and not allowlist_any:
                    sap_direct = False
                elif not erp_hits and not it_hits and not allowlist_any:
                    if "sap" not in title_norm and "ariba" not in title_norm:
                        sap_direct = False

        hits_payload = {
            "sap": sap_hits,
            "erp": erp_hits,
            "transform": xform_hits,
            "it": it_hits,
            "allowlist_title": allowlist_title_hits,
            "allowlist_product": allowlist_product_hits,
            "sap_portal": portal_hits,
            "unspsc_it": is_it_unspsc(t),
            "unspsc_non_it": is_non_it_unspsc(t),
        }

        reasons_all: List[str] = []

        if not include_supply_arrangements and _looks_like_supply_arrangement(text):
            reasons_all.append("exclude_supply_arrangement")

        if not sap_direct:
            if not include_staffing and _looks_like_staffing(title):
                reasons_all.append("exclude_staffing")

            if _looks_like_negative_title(title) and not allowlist_any:
                reasons_all.append("exclude_negative_title")

            if is_non_it_unspsc(t):
                reasons_all.append("exclude_non_it_unspsc")

            if _is_goods_or_hardware(t, text):
                reasons_all.append("exclude_goods_or_hardware")

            if not erp_hits and not allowlist_any:
                reasons_all.append("exclude_no_erp_domain")
            if not xform_hits:
                reasons_all.append("exclude_no_transformation")

            if not is_it_unspsc(t) and len(it_hits) < 1 and not allowlist_any:
                reasons_all.append("exclude_no_it_context")

            if _close_date_far_future(t, days=max_non_sap_close_days):
                reasons_all.append("exclude_far_future_close")

        # Scoring (needed for threshold)
        score = 0
        if sap_direct:
            score = 100
            score += min(20, 5 * (len(sap_hits) - 1)) if len(sap_hits) > 1 else 0
        else:
            score = 55
            score += 15 if is_it_unspsc(t) else 0
            score += min(20, 7 * len(erp_hits))
            score += min(15, 5 * len(xform_hits))
            score += min(10, 4 * len(it_hits))
            score = min(score, 95)

            if score < min_score_non_sap:
                reasons_all.append("exclude_below_score_threshold")

        if reasons_all:
            reasons[reasons_all[0]] += 1
            if return_rejected:
                item = dict(t)
                item["_reject_reason"] = reasons_all[0]
                item["_reject_reasons"] = reasons_all
                if store_hits:
                    item["_hits"] = hits_payload
                rejected.append(item)
            continue

        t["score"] = int(score)

        if store_hits:
            t["_hits"] = hits_payload

        kept.append(t)

    kept.sort(key=lambda x: x.get("score", 0), reverse=True)
    if return_rejected:
        return kept, reasons, rejected
    return kept, reasons
