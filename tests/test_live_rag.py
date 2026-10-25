"""
Live Integration Tests for Local Ollama RAG (RAG-10, RAG-11, RAG-14)
Uses the active local Ollama models (llama3.2:3b and nomic-embed-text).
Skips gracefully if local Ollama daemon is offline.
"""

import pytest
from rag_service import RAGService
from embeddings import OllamaEmbedder
from llm_service import OllamaLLMService, UNKNOWN_ANSWER_PHRASE


@pytest.fixture(scope="module")
def live_rag(tmp_path_factory):
    embedder = OllamaEmbedder()
    if not embedder.is_available():
        pytest.skip("Local Ollama embedding service is not reachable. Skipping live integration tests.")

    llm = OllamaLLMService()
    if not llm.is_available():
        pytest.skip("Local Ollama LLM service is not reachable. Skipping live integration tests.")

    temp_dir = str(tmp_path_factory.mktemp("ollama_live_chroma"))
    service = RAGService(
        llm_provider="ollama",
        embed_provider="ollama",
        vector_store_dir=temp_dir
    )
    service.index_directory("./data")
    return service


def test_live_retrieval_attendance(live_rag):
    query = "What is the minimum attendance required for semester exams?"
    chunks = live_rag.retriever.retrieve(query, top_k=4)
    assert len(chunks) > 0
    top_sources = [c["source"] for c in chunks]
    assert "academic_regulations.pdf" in top_sources


def test_live_answer_attendance(live_rag):
    query = "What is the minimum attendance required to appear for the semester examination?"
    res = live_rag.answer_question(query)
    assert res["is_refusal"] is False
    assert res["is_error"] is False
    assert "75%" in res["answer"]
    assert "academic_regulations.pdf" in res["sources"]


def test_live_hallucination_prevention_unanswerable(live_rag):
    query = "What is the swimming pool membership fee?"
    res = live_rag.answer_question(query)
    assert res["is_refusal"] is True
    assert (
        UNKNOWN_ANSWER_PHRASE.lower() in res["answer"].lower()
        or "not found" in res["answer"].lower()
    )
