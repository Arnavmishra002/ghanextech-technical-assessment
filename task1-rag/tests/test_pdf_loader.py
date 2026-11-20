import pytest

from conftest import blank_pdf, encrypt_pdf, make_pdf
from rag.pdf_loader import DocumentError, extract_pages


def test_extracts_text_with_page_numbers():
    pdf = make_pdf(["First page says hello.", "Second page talks about invoices."])
    pages = extract_pages(pdf, "report.pdf")
    assert [p.page_number for p in pages] == [1, 2]
    assert "hello" in pages[0].text
    assert "invoices" in pages[1].text
    assert all(p.source == "report.pdf" for p in pages)


def test_skips_empty_pages_but_keeps_real_page_numbers():
    pdf = make_pdf(["Intro text.", "", "Third page text."])
    pages = extract_pages(pdf, "doc.pdf")
    assert [p.page_number for p in pages] == [1, 3]


def test_rejects_non_pdf_extension():
    with pytest.raises(DocumentError, match="not a PDF"):
        extract_pages(b"just some text", "notes.txt")


def test_rejects_renamed_file_that_is_not_a_pdf():
    with pytest.raises(DocumentError, match="not a PDF"):
        extract_pages(b"PK\x03\x04 this is really a zip", "fake.pdf")


def test_password_protected_pdf_gives_clear_error():
    locked = encrypt_pdf(make_pdf(["secret content"]), user_password="hunter2")
    with pytest.raises(DocumentError, match="password-protected"):
        extract_pages(locked, "locked.pdf")


def test_pdf_encrypted_with_empty_password_is_still_read():
    pdf = encrypt_pdf(make_pdf(["readable content"]), user_password="")
    pages = extract_pages(pdf, "owner-only.pdf")
    assert "readable content" in pages[0].text


def test_scanned_or_blank_pdf_gives_clear_error():
    with pytest.raises(DocumentError, match="no text found"):
        extract_pages(blank_pdf(), "scan.pdf")


def test_damaged_pdf_gives_clear_error():
    with pytest.raises(DocumentError, match="damaged"):
        extract_pages(b"%PDF-1.4\nthis is not really a pdf body", "broken.pdf")
