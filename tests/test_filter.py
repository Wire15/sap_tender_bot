import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.filter import score_and_filter


def _tender(**overrides):
    base = {
        "uid": "t1",
        "title": "",
        "title_en": "",
        "title_fr": "",
        "description_en": "",
        "description_fr": "",
        "unspsc": [],
        "unspsc_desc_en": "",
        "unspsc_desc_fr": "",
        "gsin_desc_en": "",
        "gsin_desc_fr": "",
        "notice_type_en": "",
        "notice_type_fr": "",
        "proc_method_en": "",
        "proc_method_fr": "",
        "procurement_category": "",
    }
    base.update(overrides)
    return base


def test_filter_rejects_obvious_non_it_items():
    tenders = [
        _tender(uid="t1", title="Janitorial services", unspsc=["76101500"]),
        _tender(uid="t2", title="Plumbing services", unspsc=["72103300"]),
        _tender(uid="t3", title="Office furniture", unspsc=["56101500"]),
    ]
    kept, _ = score_and_filter(tenders)
    assert kept == []


def test_filter_keeps_sap_direct_items():
    t = _tender(
        uid="t4",
        title="SAP S/4HANA ABAP modernization",
        description_en="Upgrade SAP ECC to S/4HANA with ABAP remediation.",
        unspsc=["43211500"],
    )
    kept, _ = score_and_filter([t])
    assert len(kept) == 1
    assert kept[0]["score"] >= 100
    assert "sap" in kept[0].get("_hits", {}).get("sap", [])


def test_accent_and_apostrophe_normalization():
    t = _tender(
        uid="t5",
        title_fr="Mise en œuvre d’un système de gestion financière",
        description_fr="Implantation d'un progiciel de gestion intégré pour la paie.",
        unspsc=["43211500"],
    )
    kept, _ = score_and_filter([t])
    assert len(kept) == 1


def test_negative_title_blocks_non_sap():
    t = _tender(
        uid="t6",
        title="Roof replacement at building A",
        description_en="ERP implementation and migration for finance and payroll.",
        unspsc=["43211500"],
    )
    kept, _, rejected = score_and_filter([t], return_rejected=True)
    assert kept == []
    assert any(r.get("_reject_reason") == "exclude_negative_title" for r in rejected)


def test_supply_arrangement_excluded():
    t = _tender(
        uid="t7",
        title="Arrangement en matière d'approvisionnement - Services d'impression",
        description_en="ERP modernization project with system integration.",
        unspsc=["43211500"],
    )
    kept, _, rejected = score_and_filter([t], return_rejected=True)
    assert kept == []
    assert any(r.get("_reject_reason") == "exclude_supply_arrangement" for r in rejected)
