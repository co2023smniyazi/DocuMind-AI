"""
Mocked Unit Tests for RAG Pipeline (STEP 14-19, RAG-8 to RAG-14)
Uses unittest.mock to verify end-to-end RAG logic independently from live Ollama runtime.
"""

import pytest
from unittest.mock import MagicMock, patch
from rag_service import RAGService
from llm_service import UNKNOWN_ANSWER_PHRASE


@pytest.fixture
def mocked_rag(tmp_path):
    """
    Creates an isolated RAGService with mocked Ollama calls and a temporary ChromaDB directory.
    """
    db_dir = str(tmp_path / "mock_chroma")
    service = RAGService(vector_store_dir=db_dir)

    # Mock embedder to return deterministic 768-dim vector
    fake_vector = [0.1] * 768
    service.embedder.get_embedding = MagicMock(return_value=fake_vector)
    service.embedder.get_embeddings_batch = MagicMock(
        side_effect=lambda texts: [fake_vector for _ in texts]
    )

    return service


def test_mocked_indexing_and_retrieval(mocked_rag):
    # Add fake chunks
    fake_chunks = [
        {
            "id": "doc1_p1_c1",
            "text": "Students must maintain 75% attendance to appear for exams.",
            "source": "doc1.pdf",
            "page": 1,
            "chunk_index": 1
        }
    ]
    fake_embeddings = [[0.1] * 768]
    count = mocked_rag.vector_store.add_chunks(fake_chunks, fake_embeddings)
    assert count == 1

    retrieved = mocked_rag.retriever.retrieve("What is the attendance?")
    assert len(retrieved) > 0
    assert retrieved[0]["source"] == "doc1.pdf"
    assert "75%" in retrieved[0]["text"]


def test_mocked_answer_generation(mocked_rag):
    # Seed one chunk
    fake_chunks = [
        {
            "id": "academic_p1_c1",
            "text": "The minimum attendance required is 75%.",
            "source": "academic_regulations.pdf",
            "page": 1,
            "chunk_index": 1
        }
    ]
    mocked_rag.vector_store.add_chunks(fake_chunks, [[0.1] * 768])

    # Mock LLM response
    mocked_rag.llm_service.generate_response = MagicMock(
        return_value="According to academic_regulations.pdf, the required attendance is 75%."
    )

    res = mocked_rag.answer_question("What is the attendance requirement?")
    assert res["is_refusal"] is False
    assert "75%" in res["answer"]
    assert "academic_regulations.pdf" in res["sources"]


def test_mocked_safe_refusal_when_unavailable(mocked_rag):
    # Empty vector store
    mocked_rag.vector_store.reset_collection()

    res = mocked_rag.answer_question("What is the hostel fee?")
    assert res["is_refusal"] is True
    assert UNKNOWN_ANSWER_PHRASE in res["answer"]


def test_mocked_llm_error_handling(mocked_rag):
    fake_chunks = [
        {
            "id": "doc_p1_c1",
            "text": "Sample text",
            "source": "doc.pdf",
            "page": 1,
            "chunk_index": 1
        }
    ]
    mocked_rag.vector_store.add_chunks(fake_chunks, [[0.1] * 768])

    # Simulate Ollama down exception
    mocked_rag.llm_service.generate_response = MagicMock(
        side_effect=RuntimeError("Connection refused by Ollama")
    )

    res = mocked_rag.answer_question("Sample query")
    assert res["is_error"] is True
    assert "Connection refused by Ollama" in res["answer"]

