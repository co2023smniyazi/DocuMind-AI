"""
Embedding Generator Module (RAG-8, Cloud-Native Semantic Embeddings)
Primary: FastEmbed ONNX (BAAI/bge-small-en-v1.5, 384-dim, no API key, runs in Streamlit Cloud 1GB RAM)
Secondary: Google Gemini text-embedding-004 (768-dim, Free Tier via google-genai)
Local Fallback: Ollama nomic-embed-text (768-dim)
Test-Only: TestDeterministicEmbedder (Exclusively for unit tests)
"""

import os
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

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        pass


class FastEmbedEmbedder(BaseEmbedder):
    """
    Production Cloud Embedding Service powered by FastEmbed ONNX runtime.
    Model: BAAI/bge-small-en-v1.5 (384 dimensions, cosine distance).
    Zero API key required; runs locally or inside Streamlit Community Cloud (within 1GB RAM).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.abspath("./fastembed_cache")
        self._model = None
        self._dim = 384

    @property
    def dimension(self) -> int:
        return self._dim

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._model

    def is_available(self) -> bool:
        try:
            self._get_model()
            return True
        except Exception:
            return False

    def get_embedding(self, text: str) -> List[float]:
        cleaned = text.replace("\n", " ").strip() if text else "empty"
        model = self._get_model()
        embs = list(model.embed([cleaned]))
        return [float(x) for x in embs[0]]

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        cleaned = [t.replace("\n", " ").strip() or "empty" for t in texts]
        model = self._get_model()
        embs = list(model.embed(cleaned))
        return [[float(x) for x in e] for e in embs]


class GeminiEmbedder(BaseEmbedder):
    """
    Cloud embedding engine using Google Gemini API (text-embedding-004, 768 dimensions).
    Uses Google AI Studio free tier (verified 1,500 RPD, 15 RPM).
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "text-embedding-004"):
        self.api_key = api_key or get_secret("GEMINI_API_KEY")
        self.model_name = model_name
        self._client = None
        self._dim = 768

    @property
    def dimension(self) -> int:
        return self._dim

    def _get_client(self):
        if self._client is None and self.api_key:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def get_embedding(self, text: str) -> List[float]:
        if not self.is_available():
            raise RuntimeError("Gemini API key is not configured for text-embedding-004.")
        client = self._get_client()
        cleaned = text.replace("\n", " ").strip() if text else "empty"
        res = client.models.embed_content(model=self.model_name, contents=cleaned)
        return [float(x) for x in res.embedding.values]

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(t) for t in texts]


class OllamaEmbedder(BaseEmbedder):
    """
    Local embedding generator using Ollama nomic-embed-text (768 dimensions).
    """

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self._dim = 768

    @property
    def dimension(self) -> int:
        return self._dim

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
            return [float(x) for x in response["embeddings"][0]]
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate embedding with Ollama model '{self.model_name}': {e}."
            ) from e

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        import ollama
        cleaned_texts = [t.replace("\n", " ").strip() or "empty" for t in texts]
        try:
            response = ollama.embed(model=self.model_name, input=cleaned_texts)
            return [[float(x) for x in e] for e in response["embeddings"]]
        except Exception:
            return [self.get_embedding(t) for t in cleaned_texts]


class TestDeterministicEmbedder(BaseEmbedder):
    """
    DETERMINISTIC TEST-ONLY EMBEDDER.
    Never used in production.
    Used exclusively in automated unit tests for instantaneous, offline, reproducible test execution.
    """
    __test__ = False

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def is_available(self) -> bool:
        return True

    def get_embedding(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        clean = (text or "test").lower()
        for idx, ch in enumerate(clean):
            pos = (ord(ch) * 31 + idx) % self._dim
            vec[pos] += 1.0

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 6) for x in vec]

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(t) for t in texts]


def get_embedder(provider: Optional[str] = None) -> BaseEmbedder:
    """
    Factory function for embedding services.
    Default: FastEmbedEmbedder (BAAI/bge-small-en-v1.5, 384-dim, no API key).
    """
    selected = (provider or get_secret("EMBEDDING_PROVIDER", "fastembed")).lower()

    if selected == "fastembed" or selected == "cloud":
        return FastEmbedEmbedder()
    elif selected == "gemini":
        gemini = GeminiEmbedder()
        if gemini.is_available():
            return gemini
        return FastEmbedEmbedder()
    elif selected == "ollama":
        return OllamaEmbedder()
    elif selected == "test":
        return TestDeterministicEmbedder()
    else:
        return FastEmbedEmbedder()


# Deprecated test alias for backward compatibility with test runners
TestOnlyHashEmbedder = TestDeterministicEmbedder


