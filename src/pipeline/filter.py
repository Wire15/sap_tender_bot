from __future__ import annotations

"""
SAP Tender Bot — Filtering & Scoring

This module is intentionally conservative: it aims to surface *SAP / ERP transformation*
opportunities (and closely related enterprise-app modernization work) while filtering out
high-volume noise like:
- staffing/resource augmentation ("niveau 2", "programmeur", etc.)
- standing offers / supply arrangements (SPICT/SPICS/DAMA/AMA SAT, etc.)
- goods/hardware (laptops, peripherals, valves, furniture, etc.)
- non-IT services (janitorial, plumbing, medical services, etc.)

It returns:
- kept: list of tenders with a numeric "score"
- reasons: Counter of rejection reasons for debugging

You can tune behavior via the toggles in the CONFIG section below.
"""

import re  # noqa: E402
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
# Helpers: regex compilation
# ---------------------------

def compile_terms(terms: list[str]) -> list[Pattern]:
    """
    Compile a list of terms into regex patterns.

    - For short abbreviations like "erp", "hcm", "iam", enforce word boundaries.
      This prevents accidental substring matches inside longer words/URLs.
    - For phrases (contain spaces), use simple substring regex (escaped).
    """
    pats: list[Pattern] = []
    for t in terms:
        t = (t or "").strip().lower()
        if not t:
            continue

        # treat "word-ish" abbreviations specially
        if re.fullmatch(r"[a-z0-9.-]{2,8}", t) and " " not in t:
            pats.append(re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE))
        else:
            pats.append(re.compile(re.escape(t), re.IGNORECASE))
    return pats


def find_matches(text: str, patterns: list[Pattern]) -> list[str]:
    hits: list[str] = []
    for p in patterns:
        if p.search(text):
            hits.append(p.pattern)
    return hits


# ---------------------------
# Term dictionaries
# ---------------------------

# High-confidence SAP signals (explicit SAP ecosystem keywords)
SAP_DIRECT_TERMS = [
    "sap", "abap", "s/4hana", "s4hana", "ecc", "fiori", "basis",
    "successfactors", "ariba", "concur", "bw/4hana", "businessobjects",
    "sap btp", "sap cpi", "cloud platform integration", "sap grc",
]

# ERP/business-platform domain signals (keep these fairly specific)
ERP_DOMAIN_TERMS = [
    # English
    "enterprise resource planning", "erp",
    "financial management system", "financial system replacement",
    "general ledger", "accounts payable", "accounts receivable",
    "procure-to-pay", "source-to-pay", "e-procurement", "procurement system",
    "hris", "human capital management", "hcm",
    "payroll system", "pension payroll",
    "asset management system", "inventory management system",

    # French
    "progiciel de gestion intégré", "pgi",
    "système de gestion financière", "systeme de gestion financiere",
    "système financier", "systeme financier",
    "système de paie", "systeme de paie",
    "gestion des approvisionnements", "approvisionnement électronique", "approvisionnement electronique",
    "gestion des ressources humaines", "système de gestion des ressources humaines", "systeme de gestion des ressources humaines",
    "grand livre", "comptes créditeurs", "comptes debiteurs",
]

# Transformation intent (requires implementation/upgrade/migration/replacement…)
TRANSFORM_TERMS = [
    # English
    "implementation", "implement", "upgrade", "migration", "modernization", "modernisation",
    "replacement", "replace", "transformation", "rollout", "deployment",
    "configuration", "integrat", "data conversion", "system integration",

    # French
    "mise en œuvre", "mise en oeuvre", "implantation", "déploiement", "deploiement",
    "migration", "modernisation", "remplacement", "intégrat", "integration",
    "conversion de données", "conversion de donnees",
]

# IT context (avoid generic "application"/"data"/"solution" — too many false positives)
IT_CONTEXT_TERMS = [
    # English
    "enterprise application", "cots", "saas", "software-as-a-service",
    "systems integrator", "systems integration", "integration services",
    "middleware", "api", "interface", "interoperability",
    "identity and access management", "iam", "single sign-on", "sso",
    "erp implementation", "erp upgrade", "erp migration",

    # French
    "système d'information", "systeme d'information",
    "intégration de systèmes", "integration de systemes",
    "services d'intégration", "services d'integration",
    "progiciel", "solution infonuagique", "infonuagique",
    "gestion des identités et des accès", "gestion des identites et des acces",
]

SAP_DIRECT = compile_terms(SAP_DIRECT_TERMS)
ERP_DOMAIN = compile_terms(ERP_DOMAIN_TERMS)
TRANSFORM = compile_terms(TRANSFORM_TERMS)
IT_CONTEXT = compile_terms(IT_CONTEXT_TERMS)


# ---------------------------
# Noise filters
# ---------------------------

SUPPLY_ARRANGEMENT_TERMS = [
    "spict", "spics", "dama",
    "arrangement en matière d'approvisionnement", "arrangement en matiere d'approvisionnement",
    "standing offer", "offre à commandes", "offre a commandes",
    "ama sat", "services d'aide temporaire", "services daide temporaire",
    "rfsa", "rfso",  # often standing offer/supply arrangement
]

STAFFING_TITLE_PATTERNS = [
    r"\bniveau\s*[12345]\b",
    r"\bprogrammeur\b", r"\banalyste\b", r"\badministrateur\b", r"\bconseiller\b",
    r"\barchitecte\b", r"\bdeveloper\b", r"\bdéveloppeur\b", r"\bdeveloppeur\b",
    r"\bresource\b", r"\bressource\b", r"\bstaffing\b", r"\baugmentation\b",
]

HARDWARE_TERMS = [
    "laptop", "ordinateur portable", "peripheral", "périphérique", "peripherique",
    "boîtier", "boitier", "diplexeur", "supports de rack", "kits radio",
    "valve", "clapet", "batteries de télécommunications", "telecommunications batteries",
    "mobilier", "chaises", "furniture", "chairs",
]


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
        t.get("title_en", ""), t.get("title_fr", ""),
        t.get("description_en", ""), t.get("description_fr", ""),
        t.get("unspsc_desc_en", ""), t.get("unspsc_desc_fr", ""),
        t.get("gsin_desc_en", ""), t.get("gsin_desc_fr", ""),
        t.get("notice_type_en", ""), t.get("notice_type_fr", ""),
        t.get("proc_method_en", ""), t.get("proc_method_fr", ""),
    ]
    return (" ".join(str(p) for p in parts if p)).lower()


def _looks_like_supply_arrangement(text: str) -> bool:
    txt = text.lower()
    return any(term in txt for term in SUPPLY_ARRANGEMENT_TERMS)


def _looks_like_staffing(title: str) -> bool:
    title = (title or "").lower()
    return any(re.search(p, title) for p in STAFFING_TITLE_PATTERNS)


def _is_goods_or_hardware(t: Dict[str, object], text: str) -> bool:
    # CanadaBuys uses procurementCategory; "GD" appears in many goods postings
    proc_cat = (t.get("procurement_category", "") or "").upper()
    if "GD" in proc_cat:
        return True
    return any(term in text for term in HARDWARE_TERMS)


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

def score_and_filter(tenders: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], Counter]:
    kept: List[Dict[str, object]] = []
    reasons: Counter = Counter()

    for t in tenders:
        text = _text_blob(t)
        title = str(t.get("title", "") or "")

        sap_hits = find_matches(text, SAP_DIRECT)
        erp_hits = find_matches(text, ERP_DOMAIN)
        xform_hits = find_matches(text, TRANSFORM)
        it_hits = find_matches(text, IT_CONTEXT)

        sap_direct = bool(sap_hits)

        # Noise exclusions (unless SAP-direct)
        if not sap_direct:
            if not INCLUDE_SUPPLY_ARRANGEMENTS and _looks_like_supply_arrangement(text):
                reasons["exclude_supply_arrangement"] += 1
                continue

            if not INCLUDE_STAFFING and _looks_like_staffing(title):
                reasons["exclude_staffing"] += 1
                continue

            if is_non_it_unspsc(t):
                reasons["exclude_non_it_unspsc"] += 1
                continue

            if _is_goods_or_hardware(t, text):
                reasons["exclude_goods_or_hardware"] += 1
                continue

            # Core gate: ERP domain + transformation intent
            if not erp_hits:
                reasons["exclude_no_erp_domain"] += 1
                continue
            if not xform_hits:
                reasons["exclude_no_transformation"] += 1
                continue

            # Need either: IT UNSPSC OR strong IT context
            if not is_it_unspsc(t) and len(it_hits) < 1:
                reasons["exclude_no_it_context"] += 1
                continue

            # Long-dated closes are usually frameworks
            if _close_date_far_future(t, days=MAX_NON_SAP_CLOSE_DAYS):
                reasons["exclude_far_future_close"] += 1
                continue

        # Scoring
        score = 0
        if sap_direct:
            score = 100
            # small bonus for multiple SAP keywords (caps at +20)
            score += min(20, 5 * (len(sap_hits) - 1)) if len(sap_hits) > 1 else 0
        else:
            score = 55
            score += 15 if is_it_unspsc(t) else 0
            score += min(20, 7 * len(erp_hits))
            score += min(15, 5 * len(xform_hits))
            score += min(10, 4 * len(it_hits))
            score = min(score, 95)

            if score < MIN_SCORE_NON_SAP:
                reasons["exclude_below_score_threshold"] += 1
                continue

        t["score"] = int(score)

        if STORE_HITS:
            t["_hits"] = {
                "sap": sap_hits,
                "erp": erp_hits,
                "transform": xform_hits,
                "it": it_hits,
                "unspsc_it": is_it_unspsc(t),
                "unspsc_non_it": is_non_it_unspsc(t),
            }

        kept.append(t)

    kept.sort(key=lambda x: x.get("score", 0), reverse=True)
    return kept, reasons