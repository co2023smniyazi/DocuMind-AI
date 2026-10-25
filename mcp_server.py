"""
DocuMind AI - Model Context Protocol (MCP) Server
Exposes tools for external LLMs / agents over the college document vector database:
1. `search_documents(query: str, top_k: int = 4)` -> Retrieves semantic passages with source and page metadata.
2. `ask_documents(question: str)` -> Synthesizes an answer using the RAG pipeline.

Deployment Status: IMPLEMENTED, UNIT-TESTED WITH MOCKS, NOT DEPLOYED.
(Requires user-approved free hosting environment).
Security: Read-only access; Bearer token authentication required.
"""

import os
import json
from typing import Dict, Any, List, Optional
from rag_service import RAGService
from retriever import SemanticRetriever
from vector_store import get_vector_store
from embeddings import get_embedder


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default


class DocuMindMCPServer:
    """
    Model Context Protocol (MCP) Server for DocuMind AI.
    Exposes read-only semantic retrieval and Q&A tools.
    """

    def __init__(self, auth_token: Optional[str] = None, rag_service: Optional[RAGService] = None):
        self.auth_token = auth_token or get_secret("MCP_AUTH_TOKEN", "documind-mcp-secret-token")
        self._rag_service = rag_service

    def _get_rag_service(self) -> RAGService:
        if self._rag_service is None:
            self._rag_service = RAGService()
        return self._rag_service

    def authenticate(self, provided_token: Optional[str]) -> bool:
        """Validates bearer auth token."""
        if not self.auth_token:
            return True
        return bool(provided_token and provided_token.strip() == self.auth_token.strip())

    def search_documents(self, query: str, top_k: int = 4, auth_token: Optional[str] = None) -> Dict[str, Any]:
        """
        MCP Tool: search_documents
        Performs semantic vector search across college documents.
        Returns top-k matching chunks with document titles and page citations.
        """
        if not self.authenticate(auth_token):
            return {"status": "error", "message": "Unauthorized. Valid Bearer token required.", "results": []}

        if not query or not query.strip():
            return {"status": "error", "message": "Query parameter cannot be empty.", "results": []}

        rag = self._get_rag_service()
        try:
            chunks = rag.retriever.retrieve(query.strip(), top_k=top_k)
            formatted = []
            for c in chunks:
                formatted.append({
                    "source": c.get("source", "unknown"),
                    "page": c.get("page", 1),
                    "similarity": c.get("similarity", 0.0),
                    "text": c.get("text", "")
                })
            return {
                "status": "success",
                "tool": "search_documents",
                "query": query,
                "count": len(formatted),
                "results": formatted
            }
        except Exception as e:
            return {"status": "error", "message": f"Retrieval failure: {e}", "results": []}

    def ask_documents(self, question: str, auth_token: Optional[str] = None) -> Dict[str, Any]:
        """
        MCP Tool: ask_documents
        Answers questions about college rules, exams, and conduct using RAG.
        Strictly refuses questions with information absent from indexed documents.
        """
        if not self.authenticate(auth_token):
            return {"status": "error", "message": "Unauthorized. Valid Bearer token required.", "answer": ""}

        if not question or not question.strip():
            return {"status": "error", "message": "Question parameter cannot be empty.", "answer": ""}

        rag = self._get_rag_service()
        try:
            response = rag.answer_question(question.strip())
            return {
                "status": "success",
                "tool": "ask_documents",
                "question": question,
                "answer": response.get("answer", ""),
                "sources": response.get("sources", []),
                "is_refusal": response.get("is_refusal", False)
            }
        except Exception as e:
            return {"status": "error", "message": f"Execution failure: {e}", "answer": ""}

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns MCP tool definitions."""
        return [
            {
                "name": "search_documents",
                "description": "Semantic search over college regulations, exam guidelines, and handbooks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query or natural language phrase"},
                        "top_k": {"type": "integer", "description": "Number of passages to return", "default": 4}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "ask_documents",
                "description": "Question answering over college policies using zero-hallucination RAG.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Question regarding college regulations"}
                    },
                    "required": ["question"]
                }
            }
        ]


if __name__ == "__main__":
    print("DocuMind AI Model Context Protocol (MCP) Server")
    print("Status: Implemented, unit-tested with mocks, NOT deployed.")
    server = DocuMindMCPServer()
    print("Available tools:", [t["name"] for t in server.list_tools()])
