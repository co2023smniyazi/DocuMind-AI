"""
Unit Tests for Document Loader (RAG-6)
Verifies handling of valid, missing, empty, and invalid PDF files.
"""

import os
import pytest
from document_loader import load_pdf_with_metadata, read_pdf, load_documents_from_directory

DATA_DIR = "./data"


def test_load_existing_pdf():
    pdf_path = os.path.join(DATA_DIR, "academic_regulations.pdf")
    assert os.path.exists(pdf_path), f"File {pdf_path} must exist"

    pages, summary = load_pdf_with_metadata(pdf_path)
    assert len(pages) > 0
    assert summary["source"] == "academic_regulations.pdf"
    assert summary["total_chars"] > 100
    assert "attendance" in pages[0]["text"].lower()


def test_read_pdf():
    pdf_path = os.path.join(DATA_DIR, "examination_guidelines.pdf")
    text = read_pdf(pdf_path)
    assert len(text) > 0
    assert "revaluation" in text.lower()


def test_missing_pdf_handling():
    missing_path = os.path.join(DATA_DIR, "non_existent_file.pdf")
    pages, summary = load_pdf_with_metadata(missing_path)
    assert len(pages) == 0
    assert "not found" in summary["error"].lower()

    with pytest.raises(FileNotFoundError):
        read_pdf(missing_path)


def test_corrupted_file_handling(tmp_path):
    bad_pdf = tmp_path / "corrupted.pdf"
    bad_pdf.write_text("This is not a real PDF file header.")
    pages, summary = load_pdf_with_metadata(str(bad_pdf))
    assert len(pages) == 0
    assert summary["error"] is not None


def test_load_documents_from_directory():
    all_pages, summaries = load_documents_from_directory(DATA_DIR)
    assert len(all_pages) >= 5
    sources = {s["source"] for s in summaries}
    assert "student_handbook.pdf" in sources
    assert "placement_guidelines.pdf" in sources
    assert "academic_regulations.pdf" in sources
