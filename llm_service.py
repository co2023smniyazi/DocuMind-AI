"""
LLM Service Module (RAG-11, RAG-12, RAG-14, Cloud-Native Free Providers)
Primary Cloud: Google Gemini (Free Tier via google-genai SDK, Flash / Flash-Lite)
Secondary Cloud: Groq (Developer Free Tier via groq SDK, Llama-3.3-70B / 3.1-8B)
Optional: Anthropic Claude (Active ONLY if ENABLE_CLAUDE=true and API key exists)
Local Fallback: Ollama (llama3.2:3b)
Test-Only: MockLLMService (For deterministic offline CI tests)
"""

import os
import time
from typing import Dict, Any, Optional
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
        pass

    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass

    def build_prompt(self, question: str, context: str) -> str:
        return f"""CONTEXT:
{context}

QUESTION:
{question}"""


class GeminiLLMService(BaseLLMService):
    """
    Primary Cloud LLM Service using Google AI Studio Free Tier (google-genai SDK).
    Default Model: gemini-2.0-flash (Verified Free Tier: 15 RPM, 1,000,000 TPM, 1,500 RPD).
    Official limits: https://ai.google.dev/pricing
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1
    ):
        self.api_key = api_key or get_secret("GEMINI_API_KEY")
        self.model_name = model_name or get_secret("GEMINI_MODEL", "gemini-2.0-flash")
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None and self.api_key:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.is_available():
            raise RuntimeError(
                "Google Gemini API key is not configured. "
                "Please add GEMINI_API_KEY in your Streamlit Secrets (share.streamlit.io) or local .env file. "
                "Get a free key (Rs 0, no credit card required) at https://aistudio.google.com/apikey"
            )

        from google import genai
        from google.genai import types
        from google.genai import errors

        client = self._get_client()
        sys_instruction = system_prompt or SYSTEM_PROMPT

        # Exponential backoff for 429 rate limit
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_instruction,
                        temperature=self.temperature,
                        max_output_tokens=1024,
                    )
                )
                text = response.text
                return text.strip() if text else UNKNOWN_ANSWER_PHRASE
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = (
                    "429" in err_str
                    or "quota" in err_str
                    or "resource_exhausted" in err_str
                    or getattr(e, "code", None) == 429
                )
                if is_rate_limit and attempt < max_retries:
                    sleep_time = (attempt + 1) * 2.5
                    time.sleep(sleep_time)
                    continue

                if is_rate_limit:
                    raise RuntimeError(
                        "Google Gemini Free-Tier rate limit reached (15 RPM / 1,500 RPD). "
                        "Please wait a few seconds before asking another question. "
                        "Official limits: https://ai.google.dev/pricing"
                    ) from e
                raise RuntimeError(f"Google Gemini API error: {e}") from e


class GroqLLMService(BaseLLMService):
    """
    Secondary Free Cloud LLM Service using Groq Cloud Free Tier.
    Default Model: llama-3.3-70b-versatile or llama-3.1-8b-instant.
    Official limits: https://console.groq.com/docs/rate-limits (30 RPM, free of charge).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1
    ):
        self.api_key = api_key or get_secret("GROQ_API_KEY")
        self.model_name = model_name or get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None and self.api_key:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.is_available():
            raise RuntimeError(
                "Groq API key is not configured. "
                "Please add GROQ_API_KEY in your Streamlit Secrets or local .env file. "
                "Get a free key at https://console.groq.com/keys"
            )

        client = self._get_client()
        sys_instruction = system_prompt or SYSTEM_PROMPT

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": sys_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=1024
                )
                content = chat_completion.choices[0].message.content
                return content.strip() if content else UNKNOWN_ANSWER_PHRASE
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str
                if is_rate_limit and attempt < max_retries:
                    time.sleep((attempt + 1) * 2)
                    continue
                if is_rate_limit:
                    raise RuntimeError(
                        "Groq Free Tier rate limit reached (30 RPM). "
                        "Please wait a few seconds and try again. "
                        "Official limits: https://console.groq.com/docs/rate-limits"
                    ) from e
                raise RuntimeError(f"Groq API error: {e}") from e


class ClaudeLLMService(BaseLLMService):
    """
    Anthropic Claude Service (OPTIONAL, PAID API, OFF BY DEFAULT).
    Hard rule: Active ONLY when ENABLE_CLAUDE=true and ANTHROPIC_API_KEY is provided.
    Pricing: Paid pay-as-you-go per token (Official: https://www.anthropic.com/pricing).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        is_enabled: Optional[bool] = None
    ):
        self.api_key = api_key if api_key is not None else get_secret("ANTHROPIC_API_KEY")
        self.model_name = model_name or get_secret("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._is_enabled_override = is_enabled
        self._client = None

    def is_enabled(self) -> bool:
        if self._is_enabled_override is not None:
            return self._is_enabled_override
        val = (get_secret("ENABLE_CLAUDE", "false") or "").strip().lower()
        return val in ("true", "1", "yes")

    def is_available(self) -> bool:
        return bool(self.is_enabled() and self.api_key and len(self.api_key.strip()) > 10)

    def _get_client(self):
        if self._client is None and self.api_key:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.is_enabled():
            raise RuntimeError(
                "Anthropic Claude is disabled by default to prevent unexpected charges (Rs 0 budget policy). "
                "To enable it explicitly, set ENABLE_CLAUDE=true in secrets. "
                "Official pricing: https://www.anthropic.com/pricing"
            )
        if not self.api_key or len(self.api_key.strip()) == 0:
            raise RuntimeError("Anthropic API key is not configured.")

        import anthropic
        client = self._get_client()
        sys_instruction = system_prompt or SYSTEM_PROMPT

        try:
            message = client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=sys_instruction,
                messages=[{"role": "user", "content": prompt}]
            )
            if message and message.content:
                text_blocks = [b.text for b in message.content if getattr(b, "type", "") == "text" or hasattr(b, "text")]
                return "".join(text_blocks).strip()
            return UNKNOWN_ANSWER_PHRASE
        except Exception as e:
            raise RuntimeError(f"Anthropic Claude API Error: {e}") from e


class OllamaLLMService(BaseLLMService):
    """
    Local fallback LLM service using locally installed Ollama (llama3.2:3b).
    Not available in Streamlit Community Cloud.
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
            raise RuntimeError(f"Local Ollama error: {e}. Note: Ollama cannot run on Streamlit Cloud.") from e


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
    Factory function for LLM services.
    Default: GeminiLLMService (Free-tier Google GenAI).
    """
    selected = (provider or get_secret("LLM_PROVIDER", "gemini")).lower()

    if selected == "gemini":
        return GeminiLLMService(**kwargs)
    elif selected == "groq":
        return GroqLLMService(**kwargs)
    elif selected == "claude":
        return ClaudeLLMService(**kwargs)
    elif selected == "ollama":
        return OllamaLLMService(**kwargs)
    elif selected == "mock" or selected == "test":
        return MockLLMService(**kwargs)
    else:
        # Auto-fallback: check if Gemini key is present, else Groq, else Mock
        if get_secret("GEMINI_API_KEY"):
            return GeminiLLMService(**kwargs)
        if get_secret("GROQ_API_KEY"):
            return GroqLLMService(**kwargs)
        # Default to GeminiLLMService
        return GeminiLLMService(**kwargs)


# Backwards compatibility alias
LLMService = GeminiLLMService

