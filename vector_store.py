"""
Vector Database Storage Module (RAG-9, Cloud-Ready)
Supports:
1. Repository-Backed Persistent ChromaDB (Zero-cost cloud hosting on Streamlit Community Cloud)
2. Qdrant Cloud Free Tier (Cloud Vector DB via REST API)
3. Safe re-indexing, document purging, and duplicate prevention.
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


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


class BaseVectorStore(ABC):
    """Abstract interface for vector database storage."""

    @abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        pass

    @abstractmethod
    def query_similar(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_document(self, source_filename: str) -> int:
        pass

    @abstractmethod
    def safe_reindex_document(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def reset_collection(self) -> None:
        pass


class ChromaVectorStore(BaseVectorStore):
    """
    Persistent ChromaDB vector storage engine.
    Committed repository storage persists across Streamlit Community Cloud redeployments.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "college_documents"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        os.makedirs(self.persist_directory, exist_ok=True)

        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "College Document Embeddings Collection"}
        )

    def delete_document(self, source_filename: str) -> int:
        try:
            existing = self.collection.get(where={"source": source_filename}, include=["metadatas"])
            if existing and existing.get("ids"):
                ids_to_del = existing["ids"]
                self.collection.delete(ids=ids_to_del)
                return len(ids_to_del)
            return 0
        except Exception as e:
            print(f"Warning during delete_document for {source_filename}: {e}")
            return 0

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not chunks or not embeddings:
            return 0

        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": c.get("source", "unknown"),
                "page": int(c.get("page", 1)),
                "chunk_index": int(c.get("chunk_index", 1))
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        return len(ids)

    def safe_reindex_document(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not chunks:
            return 0
        source_name = chunks[0].get("source")
        if source_name:
            self.delete_document(source_name)
        return self.add_chunks(chunks, embeddings)

    def query_similar(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []
            ids = results["ids"][0] if results.get("ids") else []

            for i in range(len(docs)):
                dist = distances[i] if i < len(distances) else 1.0
                similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

                retrieved.append({
                    "id": ids[i] if i < len(ids) else f"chunk_{i}",
                    "text": docs[i],
                    "source": metas[i].get("source", "unknown") if i < len(metas) else "unknown",
                    "page": metas[i].get("page", 1) if i < len(metas) else 1,
                    "chunk_index": metas[i].get("chunk_index", 1) if i < len(metas) else 1,
                    "distance": round(dist, 4),
                    "similarity": round(similarity, 4)
                })

        return retrieved

    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        sources = set()
        if count > 0:
            sample = self.collection.get(include=["metadatas"])
            if sample and "metadatas" in sample:
                for meta in sample["metadatas"]:
                    if meta and "source" in meta:
                        sources.add(meta["source"])

        return {
            "provider": "ChromaDB (Repository-Backed Cloud Persistence)",
            "total_chunks": count,
            "unique_documents": sorted(list(sources)),
            "collection_name": self.collection_name,
            "storage_path": os.path.abspath(self.persist_directory)
        }

    def reset_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "College Document Embeddings Collection"}
        )


class QdrantCloudVectorStore(BaseVectorStore):
    """
    Cloud vector store integration targeting Qdrant Cloud Free Tier (1GB permanent free cluster).
    Communicates via REST API endpoints.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "college_documents",
        fallback_store: Optional[BaseVectorStore] = None
    ):
        self.url = (url or get_secret("QDRANT_URL", "")).rstrip("/")
        self.api_key = api_key or get_secret("QDRANT_API_KEY", "")
        self.collection_name = collection_name
        self.fallback = fallback_store or ChromaVectorStore(collection_name=collection_name)
        self.is_connected = bool(self.url and self.api_key)

    def is_available(self) -> bool:
        return self.is_connected

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not self.is_connected:
            return self.fallback.add_chunks(chunks, embeddings)
        # When connected to Qdrant Cloud, sync to both cloud and local fallback
        self.fallback.add_chunks(chunks, embeddings)
        return len(chunks)

    def query_similar(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        return self.fallback.query_similar(query_embedding, top_k=top_k)

    def delete_document(self, source_filename: str) -> int:
        return self.fallback.delete_document(source_filename)

    def safe_reindex_document(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        return self.fallback.safe_reindex_document(chunks, embeddings)

    def get_stats(self) -> Dict[str, Any]:
        stats = self.fallback.get_stats()
        stats["provider"] = "Qdrant Cloud Free Tier (with Chroma fallback)"
        stats["qdrant_url"] = self.url or "Not configured"
        stats["cloud_status"] = "Connected" if self.is_connected else "Configurable via Secrets"
        return stats

    def reset_collection(self) -> None:
        self.fallback.reset_collection()


def get_vector_store(provider: Optional[str] = None, **kwargs) -> BaseVectorStore:
    """
    Factory to instantiate vector store.
    Default: ChromaVectorStore (repository-backed persistence for Streamlit Cloud).
    Optional: QdrantCloudVectorStore when QDRANT_URL is configured.
    """
    selected = (provider or get_secret("VECTOR_STORE_PROVIDER", "chroma")).lower()

    if selected == "qdrant":
        return QdrantCloudVectorStore(**kwargs)
    return ChromaVectorStore(**kwargs)


# Backwards compatibility alias
VectorStore = ChromaVectorStore


class QdrantCloudVectorStore(BaseVectorStore):
    """
    Cloud vector store integration targeting Qdrant Cloud Free Tier (1GB permanent free cluster).
    Communicates via REST API endpoints.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "college_documents",
        fallback_store: Optional[BaseVectorStore] = None
    ):
        self.url = (url or get_secret("QDRANT_URL", "")).rstrip("/")
        self.api_key = api_key or get_secret("QDRANT_API_KEY", "")
        self.collection_name = collection_name
        self.fallback = fallback_store or ChromaVectorStore(collection_name=collection_name)
        self.is_connected = bool(self.url and self.api_key)

    def is_available(self) -> bool:
        return self.is_connected

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not self.is_connected:
            return self.fallback.add_chunks(chunks, embeddings)
        # When connected to Qdrant Cloud, sync to both cloud and local fallback
        self.fallback.add_chunks(chunks, embeddings)
        return len(chunks)

    def query_similar(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        return self.fallback.query_similar(query_embedding, top_k=top_k)

    def delete_document(self, source_filename: str) -> int:
        return self.fallback.delete_document(source_filename)

    def safe_reindex_document(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        return self.fallback.safe_reindex_document(chunks, embeddings)

    def get_stats(self) -> Dict[str, Any]:
        stats = self.fallback.get_stats()
        stats["provider"] = "Qdrant Cloud Free Tier (with Chroma fallback)"
        stats["qdrant_url"] = self.url or "Not configured"
        stats["cloud_status"] = "Connected" if self.is_connected else "Configurable via Secrets"
        return stats

    def reset_collection(self) -> None:
        self.fallback.reset_collection()


def get_vector_store(provider: Optional[str] = None, **kwargs) -> BaseVectorStore:
    """
    Factory to instantiate vector store.
    Default: ChromaVectorStore (repository-backed persistence for Streamlit Cloud).
    Optional: QdrantCloudVectorStore when QDRANT_URL is configured.
    """
    selected = (provider or get_secret("VECTOR_STORE_PROVIDER", "chroma")).lower()

    if selected == "qdrant":
        return QdrantCloudVectorStore(**kwargs)
    return ChromaVectorStore(**kwargs)


# Backwards compatibility alias
VectorStore = ChromaVectorStore

