import sys
from datetime import datetime, timezone

import sap_tender_bot.daily_digest as daily_digest


def test_digest_dry_run(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    cache_dir = tmp_path / "cache"
    db_path = tmp_path / "data" / "sap_tender.db"

    monkeypatch.setenv("SAP_TENDER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SAP_TENDER_LOG_DIR", str(log_dir))
    monkeypatch.setenv("SAP_TENDER_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("SAP_TENDER_DB_PATH", str(db_path))
    monkeypatch.setenv("SAP_TENDER_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("SAP_TENDER_LLM_ENABLED", "0")

    sample = {
        "uid": "sample:1",
        "title": "SAP S/4HANA migration",
        "title_en": "SAP S/4HANA migration",
        "title_fr": "",
        "description_en": "Upgrade SAP ECC to S/4HANA.",
        "description_fr": "",
        "unspsc": ["43211500"],
        "notice_type_en": "",
        "notice_type_fr": "",
        "procurement_category": "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "close_date": datetime.now(timezone.utc).isoformat(),
        "org": "Test Org",
        "url": "https://example.com",
    }

    def _fake_ingest(*_args, **_kwargs):
        return [sample]

    monkeypatch.setattr(daily_digest, "ingest", _fake_ingest)

    argv = [
        "sap-tender-digest",
        "--dry-run",
        "--export-csv",
        "--export-near-miss",
        "--export-report",
        "--no-llm",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert daily_digest.main() == 0
    assert (data_dir / "exports").exists()
