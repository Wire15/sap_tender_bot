import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.filter import score_and_filter
from pipeline.ingest import ingest


def test_smoke():
    tenders = ingest()
    picked, _ = score_and_filter(tenders)
    assert len(tenders) >= 1
    assert any("ERP" in t["title"] for t in tenders)
    assert len(picked) >= 1
