"""
Unit Tests for Cloud-Native RAG Components:
1. FastEmbed ONNX Embedder (BAAI/bge-small-en-v1.5)
2. TestDeterministicEmbedder
3. Qdrant Cloud Vector Store (in-memory mode)
4. Gemini LLM Service (mocked)
5. Groq LLM Service (mocked)
6. In-memory PDF Upload & Payload Indexing
"""

import io
import pytest
from unittest.mock import MagicMock, patch
from pypdf import PdfWriter

from embeddings import FastEmbedEmbedder, TestDeterministicEmbedder
from vector_store import QdrantCloudVectorStore
from llm_service import GeminiLLMService, GroqLLMService, UNKNOWN_ANSWER_PHRASE
from rag_service import RAGService
from document_loader import load_pdf_from_bytes


def test_fastembed_embedder():
    embedder = FastEmbedEmbedder()
    assert embedder.is_available() is True
    assert embedder.dimension == 384

    vec = embedder.get_embedding("College examination attendance requirement.")
    assert len(vec) == 384
    assert any(abs(v) > 1e-4 for v in vec)

    batch = embedder.get_embeddings_batch(["Sentence A", "Sentence B"])
    assert len(batch) == 2
    assert len(batch[0]) == 384
    assert len(batch[1]) == 384


def test_test_deterministic_embedder():
    embedder = TestDeterministicEmbedder()
    assert embedder.dimension == 384
    assert embedder.is_available() is True

    vec1 = embedder.get_embedding("test attendance")
    vec2 = embedder.get_embedding("test attendance")
    vec3 = embedder.get_embedding("different query")

    assert vec1 == vec2
    assert len(vec1) == 384
    assert vec1 != vec3


def test_qdrant_in_memory_vector_store():
    # Test real qdrant-client operations in-memory without network dependency
    store = QdrantCloudVectorStore(url=":memory:", collection_name="test_col", embedding_dimension=384)
    
    chunks = [
        {
            "id": "doc1_c1",
            "text": "75% minimum attendance is required for exams.",
            "source": "regulations.pdf",
            "page": 1,
            "chunk_index": 1
        },
        {
            "id": "doc2_c1",
            "text": "Library books must be returned within 14 days.",
            "source": "library.pdf",
            "page": 2,
            "chunk_index": 1
        }
    ]
    # Simple unit vectors for testing
    v1 = [0.1] * 384
    v2 = [-0.1] * 384

    added = store.add_chunks(chunks, [v1, v2])
    assert added == 2

    stats = store.get_stats()
    assert stats["total_chunks"] == 2
    assert "regulations.pdf" in stats["unique_documents"]
    assert "library.pdf" in stats["unique_documents"]

    # Search
    results = store.similarity_search(v1, top_k=1)
    assert len(results) == 1
    assert results[0]["source"] == "regulations.pdf"
    assert "75%" in results[0]["text"]

    # Delete by source
    deleted = store.delete_document("regulations.pdf")
    assert deleted >= 1

    stats_after = store.get_stats()
    assert stats_after["total_chunks"] == 1
    assert "regulations.pdf" not in stats_after["unique_documents"]
    assert "library.pdf" in stats_after["unique_documents"]


def test_gemini_llm_service_missing_key():
    service = GeminiLLMService(api_key="")
    assert service.is_available() is False

    with pytest.raises(RuntimeError) as exc:
        service.generate_response("test prompt")
    assert "gemini api key is not configured" in str(exc.value).lower()


def test_gemini_llm_service_mocked_success():
    service = GeminiLLMService(api_key="AIzaSy_fake_test_key_12345")
    assert service.is_available() is True

    # Mock google-genai client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "According to regulations.pdf, minimum attendance is 75%."
    mock_client.models.generate_content.return_value = mock_response
    service._client = mock_client

    ans = service.generate_response("What is attendance?")
    assert "75%" in ans
    assert "regulations.pdf" in ans


def test_groq_llm_service_mocked():
    service = GroqLLMService(api_key="gsk_fake_key_12345")
    assert service.is_available() is True

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "According to library rules, loan period is 14 days."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    service._client = mock_client

    ans = service.generate_response("Library loan period?")
    assert "14 days" in ans


def test_in_memory_pdf_upload_and_indexing(tmp_path):
    # Create in-memory synthetic PDF
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    stream = io.BytesIO()
    writer.write(stream)
    pdf_bytes = stream.getvalue()

    # Load from bytes directly
    pages, summary = load_pdf_from_bytes(pdf_bytes, "uploaded_test.pdf")
    assert summary["source"] == "uploaded_test.pdf"
    assert summary["total_pages"] == 1
    assert summary["error"] is None

    # Test indexing in RAGService with in-memory Qdrant store
    temp_dir = str(tmp_path / "temp_qdrant")
    rag = RAGService(
        llm_provider="mock",
        embed_provider="test",
        vector_provider="memory"
    )

    # Ingest synthetic PDF bytes with text
    synthetic_page_data = [{
        "source": "syllabi.pdf",
        "file_path": "memory://syllabi.pdf",
        "page": 1,
        "text": "Computer Science degree requires passing Data Structures and Algorithms with at least 50% marks."
    }]
    
    with patch("rag_service.load_pdf_from_bytes", return_value=(synthetic_page_data, {"source": "syllabi.pdf", "error": None})):
        res = rag.index_pdf_bytes(b"dummy_bytes", "syllabi.pdf")
        assert res["status"] == "success"
        assert res["indexed_chunks"] >= 1

    # Answer question from indexed in-memory payload
    ans = rag.answer_question("What is required for Data Structures?")
    assert ans["is_error"] is False
    assert len(ans["retrieved_chunks"]) >= 1
    assert ans["retrieved_chunks"][0]["source"] == "syllabi.pdf"
