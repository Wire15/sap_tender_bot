from __future__ import annotations

from pathlib import Path

from sap_tender_bot.pipeline import attachments as att


def _build_minimal_pdf(text: str) -> bytes:
    text = text.replace("(", "\\(").replace(")", "\\)")
    header = "%PDF-1.4\n"
    objects = []
    objects.append("1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append("2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
    )
    stream_text = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    objects.append(
        f"4 0 obj << /Length {len(stream_text)} >>\n"
        f"stream\n{stream_text}\nendstream\nendobj\n"
    )
    objects.append("5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")

    offsets = [0]
    body = ""
    current = len(header.encode("ascii"))
    for obj in objects:
        offsets.append(current)
        body += obj
        current += len(obj.encode("ascii"))

    xref_start = current
    xref = f"xref\n0 {len(offsets)}\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        xref += f"{offset:010d} 00000 n \n"
    trailer = (
        f"trailer << /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    )
    return (header + body + xref + trailer).encode("ascii")


def test_pdf_extraction(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(_build_minimal_pdf("Mandatory Requirements"))
    text, pages = att._extract_pdf_text(pdf_path)
    assert pages == 1
    assert "Mandatory Requirements" in text


def test_docx_extraction(tmp_path: Path) -> None:
    from docx import Document

    doc_path = tmp_path / "sample.docx"
    doc = Document()
    doc.add_paragraph("Scope of Work")
    doc.add_paragraph("Deliverables")
    doc.save(doc_path)

    text = att._extract_docx_text(doc_path)
    assert "Scope of Work" in text
    assert "Deliverables" in text


def test_section_targeting() -> None:
    text = (
        "1. Mandatory Requirements\n"
        "Vendor must provide implementation plan.\n"
        "2. Scope of Work\n"
        "Integrate SAP with CRM.\n"
        "3. Evaluation Criteria\n"
        "Rated criteria include experience.\n"
    )
    excerpt, sections = att._sectionize(text, max_chars=500, max_sections=3)
    headings = [s.get("heading") for s in sections]
    assert "1. Mandatory Requirements" in headings
    assert "2. Scope of Work" in headings
    assert "Evaluation Criteria" in excerpt


def test_attachment_cache_reuse(tmp_path, monkeypatch) -> None:
    from sap_tender_bot.config import AttachmentsConfig
    from sap_tender_bot.store import TenderStore

    pdf_bytes = _build_minimal_pdf("Mandatory Requirements")

    class FakeResponse:
        def __init__(self, status_code, headers, content):
            self.status_code = status_code
            self.headers = headers
            self._content = content

        def iter_content(self, chunk_size=1024 * 256):
            if not self._content:
                return iter(())
            return iter([self._content])

    def fake_get(url, headers=None, stream=True, timeout=None):
        if headers and headers.get("If-None-Match"):
            return FakeResponse(304, {}, b"")
        return FakeResponse(
            200,
            {
                "Content-Type": "application/pdf",
                "ETag": '"abc123"',
                "Content-Length": str(len(pdf_bytes)),
            },
            pdf_bytes,
        )

    monkeypatch.setattr(att.requests, "get", fake_get)

    db_path = tmp_path / "sap_tender.db"
    cache_dir = tmp_path / "cache"
    store = TenderStore(db_path).open()
    config = AttachmentsConfig(
        enabled=True,
        llm_enabled=False,
        max_attachments_per_run=5,
        max_attachments_per_tender=2,
        max_bytes=5_000_000,
    )

    tender1 = {"attachments": ["https://example.com/doc.pdf"]}
    stats1 = att.enrich_tenders_with_attachments(
        [tender1],
        config=config,
        store=store,
        cache_dir=cache_dir,
    )
    assert stats1["downloaded"] == 1

    tender2 = {"attachments": ["https://example.com/doc.pdf"]}
    stats2 = att.enrich_tenders_with_attachments(
        [tender2],
        config=config,
        store=store,
        cache_dir=cache_dir,
    )
    assert stats2["downloaded"] == 0
    assert stats2["cached"] >= 1

    store.close()
