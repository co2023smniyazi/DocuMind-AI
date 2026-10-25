"""
Semantic Retriever Module (RAG-10, Cloud-Ready)
Converts user questions into embeddings and searches vector storage for the most relevant document chunks.
Works with both Cloud and Local vector stores and embedding providers.
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

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic retrieval for a natural-language query.
        1. User question is converted into an embedding.
        2. Vector database is searched.
        3. Top relevant chunks are returned with source information preserved.
        """
        if not query or not query.strip():
            return []

        # 1. Generate query embedding vector
        try:
            query_embedding = self.embedder.get_embedding(query)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []

        # 2. Retrieve top-k closest chunks
        results = self.vector_store.query_similar(query_embedding, top_k=top_k)

        # 3. Filter by minimum similarity threshold if specified
        if min_similarity > 0.0:
            results = [r for r in results if r.get("similarity", 0) >= min_similarity]

        return results
