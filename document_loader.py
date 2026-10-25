"""
Document Loader Module (RAG-6)
Extracts clean text and metadata from PDF files using PyPDF.
Supports both file path and in-memory bytes loading.
Handles missing, encrypted, corrupted, empty, and scanned PDFs gracefully.
"""

import os
import io
from typing import List, Dict, Any, Tuple, Union
from pypdf import PdfReader
from pypdf.errors import PdfReadError


def read_pdf(file_path: str) -> str:
    """
    Extracts concatenated text from all readable pages of a single PDF file.
    Follows Step 12 & 29 specification with robust error handling.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file does not exist: {file_path}")

    try:
        reader = PdfReader(file_path)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return "[Error: PDF document is encrypted and password-protected.]"

        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except PdfReadError as e:
        return f"[Error: Corrupted or invalid PDF format - {e}]"
    except Exception as e:
        return f"[Error reading PDF: {e}]"


def load_pdf_from_bytes(file_bytes: bytes, filename: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts text page-by-page from in-memory PDF bytes without saving to disk.
    Ensures Streamlit Cloud ephemeral persistence requirement is respected.
    """
    pages_data = []
    summary = {
        "source": filename,
        "file_path": f"memory://{filename}",
        "total_pages": 0,
        "total_chars": 0,
        "is_scanned": False,
        "is_encrypted": False,
        "error": None
    }

    if not file_bytes:
        summary["error"] = "Uploaded file buffer is empty."
        return pages_data, summary

    try:
        stream = io.BytesIO(file_bytes)
        reader = PdfReader(stream)
        summary["total_pages"] = len(reader.pages)

        if reader.is_encrypted:
            summary["is_encrypted"] = True
            try:
                reader.decrypt("")
            except Exception:
                summary["error"] = "PDF is password-encrypted."
                return pages_data, summary

        total_extracted_chars = 0
        for page_idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as pe:
                page_text = f"[Page extraction error: {pe}]"

            clean_text = page_text.strip()
            total_extracted_chars += len(clean_text)

            pages_data.append({
                "source": filename,
                "file_path": f"memory://{filename}",
                "page": page_idx + 1,
                "text": clean_text
            })

        summary["total_chars"] = total_extracted_chars
        if summary["total_pages"] > 0 and total_extracted_chars == 0:
            summary["is_scanned"] = True
            summary["warning"] = "Document contains no extractable text. Scanned or image-based PDFs require OCR."

    except PdfReadError as pre:
        summary["error"] = f"Invalid or corrupted PDF file: {pre}"
    except Exception as e:
        summary["error"] = f"Failed to load PDF: {e}"

    return pages_data, summary


def load_pdf_with_metadata(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts text page-by-page from a file path, retaining source filename and page numbers.
    """
    filename = os.path.basename(file_path)
    if not os.path.exists(file_path):
        return [], {
            "source": filename,
            "file_path": os.path.abspath(file_path),
            "total_pages": 0,
            "total_chars": 0,
            "is_scanned": False,
            "is_encrypted": False,
            "error": f"File not found: {file_path}"
        }

    try:
        with open(file_path, "rb") as f:
            content = f.read()
        pages, summary = load_pdf_from_bytes(content, filename)
        summary["file_path"] = os.path.abspath(file_path)
        for p in pages:
            p["file_path"] = os.path.abspath(file_path)
        return pages, summary
    except Exception as e:
        return [], {
            "source": filename,
            "file_path": os.path.abspath(file_path),
            "total_pages": 0,
            "total_chars": 0,
            "is_scanned": False,
            "is_encrypted": False,
            "error": f"Failed to read file: {e}"
        }


def load_documents_from_directory(dir_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Scans a directory for all .pdf files and extracts page data.
    Skips unreadable files without failing the entire batch.
    """
    all_pages = []
    doc_summaries = []

    if not os.path.exists(dir_path):
        return all_pages, doc_summaries

    for item in sorted(os.listdir(dir_path)):
        if item.lower().endswith(".pdf"):
            full_path = os.path.join(dir_path, item)
            pages, summary = load_pdf_with_metadata(full_path)
            doc_summaries.append(summary)
            if not summary.get("error"):
                all_pages.extend(pages)

    return all_pages, doc_summaries
