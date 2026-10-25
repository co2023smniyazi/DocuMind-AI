"""
Unit and Integration Tests for Anthropic Claude AI Service (Requirement 2)
Verifies Claude integration, prompt structure, error handling, and secret management.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from llm_service import ClaudeLLMService, get_llm_service, UNKNOWN_ANSWER_PHRASE


def test_claude_service_missing_api_key():
    # When no key is provided, is_available must be False
    service = ClaudeLLMService(api_key="")
    assert service.is_available() is False

    with pytest.raises(RuntimeError) as exc_info:
        service.generate_response("Test prompt")
    assert "Anthropic API key is not configured" in str(exc_info.value)


def test_claude_service_custom_model():
    service = ClaudeLLMService(api_key="sk-ant-test-key-mock", model_name="claude-3-5-sonnet-20241022")
    assert service.model_name == "claude-3-5-sonnet-20241022"
    assert service.is_available() is True


def test_claude_service_mocked_success():
    service = ClaudeLLMService(api_key="sk-ant-test-key-mock")

    # Mock the anthropic client and messages.create
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = "According to academic_regulations.pdf, attendance must be at least 75%."

    mock_response = MagicMock()
    mock_response.content = [mock_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    service._client = mock_client

    result = service.generate_response("What is attendance?")
    assert "75%" in result
    assert "academic_regulations.pdf" in result
    mock_client.messages.create.assert_called_once()


def test_claude_service_mocked_refusal():
    service = ClaudeLLMService(api_key="sk-ant-test-key-mock")

    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = UNKNOWN_ANSWER_PHRASE

    mock_response = MagicMock()
    mock_response.content = [mock_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    service._client = mock_client

    result = service.generate_response("What is the swimming pool fee?")
    assert UNKNOWN_ANSWER_PHRASE in result


def test_claude_service_mocked_rate_limit_error():
    service = ClaudeLLMService(api_key="sk-ant-test-key-mock")

    import anthropic
    mock_client = MagicMock()
    # Create mock response object for RateLimitError
    fake_response = MagicMock()
    fake_response.status_code = 429
    fake_response.headers = {}
    mock_client.messages.create.side_effect = anthropic.RateLimitError(
        message="Rate limit exceeded",
        response=fake_response,
        body={"error": {"message": "Rate limit exceeded"}}
    )
    service._client = mock_client

    with pytest.raises(RuntimeError) as exc_info:
        service.generate_response("Test prompt")
    assert "Rate Limit Exceeded" in str(exc_info.value)


def test_factory_get_llm_service():
    service_claude = get_llm_service(provider="claude", api_key="sk-ant-dummy-key")
    assert isinstance(service_claude, ClaudeLLMService)

    service_mock = get_llm_service(provider="mock")
    assert service_mock.is_available() is True
    ans = service_mock.generate_response("any prompt")
    assert len(ans) > 0

