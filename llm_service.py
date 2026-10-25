"""
LLM Service Module (RAG-11, RAG-12, RAG-14, RAG-22)
Provides unified interface for both Anthropic Claude (Cloud-native) and local Ollama.
Supports dynamic provider switching, secure secret handling, and anti-hallucination enforcement.
"""

import os
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

# Helper to read from environment or Streamlit secrets safely
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    # Check OS environment first
    val = os.environ.get(key)
    if val:
        return val
    # Check Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default


UNKNOWN_ANSWER_PHRASE = "Information not found in the available documents."

SYSTEM_PROMPT = """You are a college information assistant.
Answer the user's question using ONLY the context provided below.
Do not use external knowledge or invent facts.
If the question is paraphrased or asks about a related policy (e.g., how many classes can be missed, passing requirements, or placement rules), explain what the documents explicitly state about that topic (such as the 75% attendance requirement or detention rules below 65%).
If the information or topic is completely absent from the context, reply strictly:
"Information not found in the available documents."
Give a concise, clear, and direct answer.
Always cite the source document name and page number if available."""


class BaseLLMService(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the provider is configured and reachable."""
        pass

    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates an answer from prompt."""
        pass

    def build_prompt(self, question: str, context: str) -> str:
        """Formats the context and question prompt."""
        return f"""CONTEXT:
{context}

QUESTION:
{question}"""


class ClaudeLLMService(BaseLLMService):
    """
    Cloud LLM service powered by Anthropic Claude via official Python SDK.
    Supports Claude 3.5 Sonnet, Claude 3.5 Haiku, and Claude 3 Haiku.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        self.api_key = api_key or get_secret("ANTHROPIC_API_KEY")
        self.model_name = model_name or get_secret("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None and self.api_key:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        """Verifies if Anthropic API key is provided."""
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Calls Anthropic Claude Messages API with robust error handling.
        """
        if not self.is_available():
            raise RuntimeError(
                "Anthropic API key is not configured. "
                "Please set ANTHROPIC_API_KEY in your environment variables or Streamlit Secrets (share.streamlit.io). "
                "Alternatively, select another provider in settings."
            )

        import anthropic
        client = self._get_client()
        sys_instruction = system_prompt or SYSTEM_PROMPT

        try:
            message = client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=sys_instruction,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            # Extract generated response text
            if message and message.content:
                text_blocks = [b.text for b in message.content if getattr(b, "type", "") == "text" or hasattr(b, "text")]
                return "".join(text_blocks).strip()
            return UNKNOWN_ANSWER_PHRASE

        except anthropic.AuthenticationError as ae:
            raise RuntimeError(
                f"Anthropic API Authentication Failed: Invalid API key ({ae}). "
                "Verify your ANTHROPIC_API_KEY in Streamlit Secrets or .env file."
            ) from ae
        except anthropic.RateLimitError as rle:
            raise RuntimeError(
                f"Anthropic API Rate Limit Exceeded: {rle}. Please wait a moment and try again."
            ) from rle
        except anthropic.APITimeoutError as te:
            raise RuntimeError(
                f"Anthropic API request timed out: {te}. Please retry."
            ) from te
        except anthropic.APIConnectionError as ace:
            raise RuntimeError(
                f"Anthropic API connection error: {ace}. Check network connectivity."
            ) from ace
        except anthropic.APIStatusError as ase:
            raise RuntimeError(
                f"Anthropic API status error {ase.status_code}: {ase.message}"
            ) from ase
        except Exception as e:
            raise RuntimeError(f"Anthropic Claude API Error: {e}") from e


class OllamaLLMService(BaseLLMService):
    """
    Local fallback LLM service using locally installed Ollama.
    """

    def __init__(self, model_name: str = "llama3.2:3b", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature

    def is_available(self) -> bool:
        try:
            import ollama
            res = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "ping"}]
            )
            return bool(res and "message" in res)
        except Exception:
            return False

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        import ollama
        sys_instruction = system_prompt or SYSTEM_PROMPT
        full_content = f"{sys_instruction}\n\n{prompt}"

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": full_content}],
                options={"temperature": self.temperature}
            )
            return response["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(
                f"Error communicating with local LLM '{self.model_name}': {e}. "
                "Ensure Ollama is running ('ollama serve') or switch to Claude in cloud settings."
            ) from e


class MockLLMService(BaseLLMService):
    """
    Deterministic mock service for unit testing and offline CI verification.
    """

    def __init__(self, canned_response: Optional[str] = None):
        self.canned_response = canned_response

    def is_available(self) -> bool:
        return True

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self.canned_response:
            return self.canned_response
        lower_p = prompt.lower()
        if any(w in lower_p for w in ["hostel", "swimming", "chief guest", "cafeteria", "bus route", "parking", "laundry", "alumni", "flight"]):
            return UNKNOWN_ANSWER_PHRASE
        return "According to the college academic regulations, students must maintain at least 75% attendance. Source: academic_regulations.pdf"


def get_llm_service(provider: Optional[str] = None, **kwargs) -> BaseLLMService:
    """
    Factory function to instantiate the selected LLM provider.
    Priority:
    1. Explicit provider parameter
    2. Environment / Secrets variable LLM_PROVIDER
    3. Auto: if ANTHROPIC_API_KEY is present -> Claude; else -> Ollama
    """
    selected_provider = (provider or get_secret("LLM_PROVIDER", "auto")).lower()

    if selected_provider == "claude":
        return ClaudeLLMService(**kwargs)
    elif selected_provider == "ollama":
        return OllamaLLMService(**kwargs)
    elif selected_provider == "mock":
        return MockLLMService(**kwargs)
    else:
        # Auto-detect
        api_key = get_secret("ANTHROPIC_API_KEY")
        if api_key and len(api_key.strip()) > 10:
            return ClaudeLLMService(**kwargs)
        else:
            # Fall back to Ollama or Mock
            ollama_service = OllamaLLMService(**kwargs)
            return ollama_service


# Backwards compatibility alias
LLMService = ClaudeLLMService
