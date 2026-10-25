"""
Unit Tests for Text Chunking Module (RAG-7)
Verifies chunking size, overlap logic, parameter guards, and metadata preservation.
"""

import pytest
from chunker import create_chunks, chunk_document_pages


def test_create_chunks_small_text():
    text = "Short sentence under 1500 characters."
    chunks = create_chunks(text, chunk_size=1500, overlap=200)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_create_chunks_empty_and_whitespace():
    assert create_chunks("") == []
    assert create_chunks("   \n\t  ") == []


def test_create_chunks_overlapping():
    text = "A" * 3000
    chunks = create_chunks(text, chunk_size=1500, overlap=200)
    assert len(chunks) > 1
    assert len(chunks[0]) == 1500
    assert len(chunks[1]) > 0


def test_create_chunks_invalid_overlap_guard():
    # If overlap >= chunk_size, the guard prevents infinite loop
    text = "A" * 1000
    chunks = create_chunks(text, chunk_size=300, overlap=500)
    assert len(chunks) > 1


def test_chunk_document_pages_metadata():
    pages = [
        {
            "source": "academic_regulations.pdf",
            "page": 1,
            "text": "Paragraph 1 about college attendance requirements."
        }
    ]
    chunks = chunk_document_pages(pages, chunk_size=1500, overlap=200)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["source"] == "academic_regulations.pdf"
    assert chunk["page"] == 1
    assert chunk["chunk_index"] == 1
    assert chunk["id"] == "academic_regulations.pdf_p1_c1"
    assert chunk["char_length"] == len(pages[0]["text"])
