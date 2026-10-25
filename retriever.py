"""
Semantic Retriever Module (RAG-10, Scalable & Deduplicated)
Converts user questions into embeddings and searches vector storage for the most relevant document chunks.
Works with both Cloud and Local vector stores and embedding providers.
Features:
- Near-duplicate suppression across chunks.
- Configurable minimum similarity threshold.
- Namespace / source filtering.
"""

from typing import List, Dict, Any, Optional
from embeddings import BaseEmbedder, get_embedder
from vector_store import BaseVectorStore, get_vector_store


class SemanticRetriever:
    """
    Coordinates query embedding generation and similarity retrieval against the vector store.
    """

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedder: Optional[BaseEmbedder] = None
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or get_embedder()

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]], max_chars_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Suppresses near-duplicate chunks from the same document to maximize context diversity.
        """
        seen_snippets = set()
        unique = []
        for c in chunks:
            text = c.get("text", "").strip()
            # Key based on first 120 normalized chars
            snippet_key = (c.get("source", ""), c.get("page", 1), text[:120].lower())
            if snippet_key in seen_snippets:
                continue
            seen_snippets.add(snippet_key)
            unique.append(c)
        return unique

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        min_similarity: float = 0.0,
        filter_source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic retrieval for a natural-language query.
        1. User question is converted into an embedding.
        2. Vector database is searched with top_k * 2 candidates.
        3. Near-duplicate chunks are removed.
        4. Top relevant chunks are returned with source information preserved.
        """
        if not query or not query.strip():
            return []

        # 1. Generate query embedding vector
        try:
            query_embedding = self.embedder.get_embedding(query)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []

        # 2. Retrieve top closest chunks (fetch extra for deduplication)
        fetch_k = top_k * 2 if top_k > 1 else 1
        results = self.vector_store.query_similar(query_embedding, top_k=fetch_k)

        # 3. Optional source filter
        if filter_source:
            results = [r for r in results if r.get("source") == filter_source]

        # 4. Filter by minimum similarity threshold if specified
        if min_similarity > 0.0:
            results = [r for r in results if r.get("similarity", 0) >= min_similarity]

        # 5. Deduplicate and trim to requested top_k
        deduped = self._deduplicate_chunks(results)
        return deduped[:top_k]
