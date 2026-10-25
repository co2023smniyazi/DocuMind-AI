"""
RAG Pipeline Service Module (RAG-11, RAG-13, RAG-14, RAG-16, Cloud-Ready)
Orchestrates end-to-end RAG flow:
PDF Ingestion -> Chunking -> Embeddings -> Vector Search -> Context Injection -> LLM Answer Generation -> Citation.
Supports Anthropic Claude, Cloud Embeddings, ChromaDB / Qdrant Cloud, and safe re-indexing.
"""

import os
from typing import Dict, Any, List, Optional
from document_loader import load_documents_from_directory, load_pdf_with_metadata, load_pdf_from_bytes
from chunker import chunk_document_pages
from embeddings import BaseEmbedder, get_embedder
from vector_store import BaseVectorStore, get_vector_store
from retriever import SemanticRetriever
from llm_service import BaseLLMService, get_llm_service, UNKNOWN_ANSWER_PHRASE


class RAGService:
    """
    Complete Retrieval-Augmented Generation Service supporting Cloud and Local Providers.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        embed_provider: Optional[str] = None,
        vector_provider: Optional[str] = None,
        vector_store_dir: str = "./chroma_db",
        top_k: int = 4
    ):
        self.embedder: BaseEmbedder = get_embedder(provider=embed_provider)
        embed_dim = getattr(self.embedder, "dimension", 384)
        self.vector_store: BaseVectorStore = get_vector_store(
            provider=vector_provider,
            persist_directory=vector_store_dir,
            embedding_dimension=embed_dim
        )
        self.retriever = SemanticRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder
        )
        self.llm_service: BaseLLMService = get_llm_service(provider=llm_provider)
        self.top_k = top_k

    def check_health(self) -> Dict[str, Any]:
        """
        Validates connection status across cloud and local services.
        """
        llm_ok = self.llm_service.is_available()
        embed_ok = self.embedder.is_available()
        stats = self.vector_store.get_stats()

        llm_type = self.llm_service.__class__.__name__
        embed_type = self.embedder.__class__.__name__

        return {
            "llm_service": llm_type,
            "llm_online": llm_ok,
            "llm_model": getattr(self.llm_service, "model_name", "Mock"),
            "embedding_service": embed_type,
            "embedding_online": embed_ok,
            "vector_store_provider": stats.get("provider", "ChromaDB"),
            "indexed_chunks": stats.get("total_chunks", 0),
            "indexed_documents": stats.get("unique_documents", []),
            "all_healthy": bool(llm_ok and embed_ok)
        }

    def index_directory(
        self,
        dir_path: str = "./data",
        chunk_size: int = 1500,
        overlap: int = 200,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Indexes all PDF files in the target directory into the vector store.
        Safely clears collection if force_reindex=True.
        Detects scanned and unreadable PDFs.
        """
        if not os.path.exists(dir_path):
            return {"status": "error", "message": f"Directory not found: {dir_path}", "indexed_chunks": 0}

        if force_reindex:
            self.reset_index()

        pages_data, summaries = load_documents_from_directory(dir_path)
        if not pages_data:
            return {
                "status": "empty",
                "message": "No readable PDF pages found in directory.",
                "indexed_chunks": 0,
                "summaries": summaries
            }

        chunks = chunk_document_pages(pages_data, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return {
                "status": "empty",
                "message": "No text chunks generated from pages.",
                "indexed_chunks": 0,
                "summaries": summaries
            }

        try:
            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.get_embeddings_batch(texts)
            count = self.vector_store.add_chunks(chunks, embeddings)

            scanned_docs = [s["source"] for s in summaries if s.get("is_scanned")]
            error_docs = [s["source"] for s in summaries if s.get("error")]

            return {
                "status": "success",
                "indexed_chunks": count,
                "total_pages": len(pages_data),
                "documents": list({p["source"] for p in pages_data}),
                "scanned_documents": scanned_docs,
                "failed_documents": error_docs,
                "summaries": summaries
            }
        except Exception as e:
            return {"status": "error", "message": f"Embedding or storage failure: {e}", "indexed_chunks": 0}

    def index_pdf(
        self,
        file_path: str,
        chunk_size: int = 1500,
        overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Indexes a single PDF into the vector store.
        Purges prior chunks for this document first, preventing duplicates.
        """
        pages_data, summary = load_pdf_with_metadata(file_path)
        filename = os.path.basename(file_path)

        if summary.get("error"):
            return {
                "status": "error",
                "message": summary["error"],
                "filename": filename,
                "indexed_chunks": 0
            }

        if summary.get("is_scanned"):
            return {
                "status": "warning",
                "message": "Document contains no extractable text. Scanned image PDFs require an OCR engine.",
                "filename": filename,
                "indexed_chunks": 0
            }

        chunks = chunk_document_pages(pages_data, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return {
                "status": "empty",
                "message": "Document produced no text chunks.",
                "filename": filename,
                "indexed_chunks": 0
            }

        try:
            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.get_embeddings_batch(texts)
            count = self.vector_store.safe_reindex_document(chunks, embeddings)
            return {
                "status": "success",
                "indexed_chunks": count,
                "filename": filename
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Embedding failure: {e}",
                "filename": filename,
                "indexed_chunks": 0
            }

    def index_pdf_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_size: int = 1500,
        overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Indexes a PDF directly from in-memory bytes without persisting raw PDF to disk.
        Safely clears existing chunks for the same filename to avoid duplication.
        All text chunks and source metadata are stored in the vector database payload.
        """
        pages_data, summary = load_pdf_from_bytes(file_bytes, filename)

        if summary.get("error"):
            return {
                "status": "error",
                "message": summary["error"],
                "filename": filename,
                "indexed_chunks": 0
            }

        if summary.get("is_scanned"):
            return {
                "status": "warning",
                "message": "Document contains no extractable text. Scanned image PDFs require an OCR engine.",
                "filename": filename,
                "indexed_chunks": 0
            }

        chunks = chunk_document_pages(pages_data, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return {
                "status": "empty",
                "message": "Document produced no text chunks.",
                "filename": filename,
                "indexed_chunks": 0
            }

        try:
            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.get_embeddings_batch(texts)
            count = self.vector_store.safe_reindex_document(chunks, embeddings)
            return {
                "status": "success",
                "indexed_chunks": count,
                "filename": filename
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Embedding or storage failure: {e}",
                "filename": filename,
                "indexed_chunks": 0
            }

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks into clear contextual blocks for the LLM prompt.
        """
        context_blocks = []
        for chunk in retrieved_chunks:
            source = chunk.get("source", "Unknown Document")
            page = chunk.get("page", 1)
            text = chunk.get("text", "").strip()
            block = f"[Document: {source} | Page: {page}]\n{text}"
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

    def answer_question(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Complete question answering workflow:
        1. Retrieve top-k chunks.
        2. Format context and prompt.
        3. Query configured LLM (Claude or local fallback).
        4. Validate response & format citations.
        """
        k = top_k or self.top_k
        clean_question = question.strip() if question else ""

        if not clean_question:
            return {
                "answer": "Please enter a valid question.",
                "sources": [],
                "retrieved_chunks": [],
                "is_refusal": False,
                "is_error": False
            }

        # 1. Retrieve relevant chunks
        try:
            retrieved_chunks = self.retriever.retrieve(clean_question, top_k=k)
        except Exception as e:
            return {
                "answer": f"Retrieval error: {e}",
                "sources": [],
                "retrieved_chunks": [],
                "is_refusal": False,
                "is_error": True
            }

        if not retrieved_chunks:
            return {
                "answer": UNKNOWN_ANSWER_PHRASE,
                "sources": [],
                "retrieved_chunks": [],
                "is_refusal": True,
                "is_error": False
            }

        # 2. Build context
        context = self.format_context(retrieved_chunks)

        # 3. Construct prompt
        prompt = self.llm_service.build_prompt(clean_question, context)

        # 4. Generate answer from configured LLM
        try:
            raw_answer = self.llm_service.generate_response(prompt)
        except Exception as e:
            err_msg = str(e)
            return {
                "answer": f"⚠️ {err_msg}",
                "sources": [],
                "retrieved_chunks": retrieved_chunks,
                "is_refusal": False,
                "is_error": True
            }

        # 5. Check if LLM indicates answer not found
        is_refusal = (
            UNKNOWN_ANSWER_PHRASE.lower() in raw_answer.lower()
            or "not found in the available documents" in raw_answer.lower()
            or "information not found" in raw_answer.lower()
        )

        # 6. Extract unique sources
        sources = list(dict.fromkeys(c.get("source") for c in retrieved_chunks if c.get("source")))

        return {
            "answer": raw_answer,
            "sources": [] if is_refusal else sources,
            "retrieved_chunks": retrieved_chunks,
            "is_refusal": is_refusal,
            "is_error": False,
            "prompt_used": prompt
        }

    def get_stats(self) -> Dict[str, Any]:
        return self.vector_store.get_stats()

    def reset_index(self):
        self.vector_store.reset_collection()
