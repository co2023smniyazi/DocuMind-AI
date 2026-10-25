"""
Embedding Generator Module (RAG-8, Cloud-Ready)
Provides both Local Ollama Embeddings (nomic-embed-text) and
Cloud-Native Self-Contained Semantic Embeddings for zero-daemon cloud deployment.
"""

import os
import re
import math
import hashlib
from typing import List, Optional
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


class BaseEmbedder(ABC):
    """Abstract base class for vector embedding engines."""

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        pass


class OllamaEmbedder(BaseEmbedder):
    """
    Local embedding generator using Ollama nomic-embed-text.
    """

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name

    def is_available(self) -> bool:
        try:
            import ollama
            res = ollama.embed(model=self.model_name, input="ping")
            return bool(res and "embeddings" in res and res["embeddings"])
        except Exception:
            return False

    def get_embedding(self, text: str) -> List[float]:
        cleaned = text.replace("\n", " ").strip() if text else "empty"
        import ollama
        try:
            response = ollama.embed(model=self.model_name, input=cleaned)
            return response["embeddings"][0]
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate embedding with Ollama model '{self.model_name}': {e}. "
                "Ensure Ollama is running or switch to cloud embedding provider."
            ) from e

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        import ollama
        cleaned_texts = [t.replace("\n", " ").strip() or "empty" for t in texts]
        try:
            response = ollama.embed(model=self.model_name, input=cleaned_texts)
            return response["embeddings"]
        except Exception:
            return [self.get_embedding(t) for t in cleaned_texts]


class CloudSemanticEmbedder(BaseEmbedder):
    """
    Cloud-native self-contained semantic feature embedder.
    Runs 100% in memory on Streamlit Cloud CPU with zero external daemons or API keys.
    Generates unit-normalized 768-dimensional dense vectors using hashed n-gram token projections.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def is_available(self) -> bool:
        return True

    STOPWORDS = {
        "what", "is", "the", "a", "an", "for", "in", "to", "of", "and", "or",
        "it", "this", "that", "on", "at", "by", "from", "with", "as", "be",
        "are", "do", "does", "did", "how", "can", "could", "would", "should",
        "i", "you", "he", "she", "we", "they", "my", "your", "our", "all"
    }

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and clean
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        raw_tokens = [w for w in cleaned.split() if len(w) > 1 and w not in self.STOPWORDS]
        # Include subword prefixes/suffixes (3 to 6 chars) for semantic morphological matching
        features = list(raw_tokens)
        for t in raw_tokens:
            if len(t) > 3:
                features.append(t[:4])
                features.append(t[-4:])
        return features

    def get_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            text = "empty"

        tokens = self._tokenize(text)
        vec = [0.0] * self.dimension

        if not tokens:
            return [1.0 / math.sqrt(self.dimension)] * self.dimension

        for token in tokens:
            # Hash token into dimension index and signed weight
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 16) & 1) else -1.0
            vec[idx] += sign

        # L2 Normalization to unit sphere
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        else:
            vec = [1.0 / math.sqrt(self.dimension)] * self.dimension

        return vec

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(t) for t in texts]


def get_embedder(provider: Optional[str] = None) -> BaseEmbedder:
    """
    Factory to retrieve the appropriate embedder.
    Default in Cloud: CloudSemanticEmbedder (zero-cost, runs on Streamlit Cloud CPU).
    Default in Local: OllamaEmbedder if Ollama is running.
    """
    selected = (provider or get_secret("EMBEDDING_PROVIDER", "auto")).lower()

    if selected == "cloud" or selected == "self_contained":
        return CloudSemanticEmbedder(dimension=768)
    elif selected == "ollama":
        return OllamaEmbedder(model_name="nomic-embed-text")
    else:
        # Auto-detect: if Ollama is active, use Ollama; otherwise CloudSemanticEmbedder
        ollama_embedder = OllamaEmbedder()
        if ollama_embedder.is_available():
            return ollama_embedder
        return CloudSemanticEmbedder(dimension=768)
