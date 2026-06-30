"""Provider tests — all HTTP calls are mocked. No real network traffic."""
import json
import pytest
from unittest.mock import MagicMock, patch

from harness.providers.ollama_provider import OllamaProvider
from harness.providers.groq_provider import GroqProvider, RateLimitError
from harness.providers.base import GenerationResult


# ── Ollama ────────────────────────────────────────────────────────────────────

OLLAMA_RESPONSE = {
    "model": "llama3.2:3b",
    "response": "  Patient has a fever.  ",
    "prompt_eval_count": 55,
    "eval_count": 8,
    "done": True,
}


def _mock_ollama_response(status=200, body=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = body or OLLAMA_RESPONSE
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


@patch("harness.providers.ollama_provider.requests.post")
def test_ollama_builds_correct_request(mock_post):
    mock_post.return_value = _mock_ollama_response()
    provider = OllamaProvider(model="llama3.2:3b", host="http://localhost:11434")
    provider.generate("Summarize this.", max_tokens=256, temperature=0.0)

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["model"] == "llama3.2:3b"
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == 256
    assert payload["options"]["temperature"] == 0.0


@patch("harness.providers.ollama_provider.requests.post")
def test_ollama_parses_response_correctly(mock_post):
    mock_post.return_value = _mock_ollama_response()
    provider = OllamaProvider()
    result = provider.generate("Summarize this.")

    assert isinstance(result, GenerationResult)
    assert result.text == "Patient has a fever."  # stripped
    assert result.provider == "ollama"
    assert result.input_tokens == 55
    assert result.output_tokens == 8
    assert result.cost_usd == 0.0
    assert result.latency_ms >= 0


@patch("harness.providers.ollama_provider.requests.post")
def test_ollama_missing_token_counts_gives_none(mock_post):
    body = {**OLLAMA_RESPONSE}
    del body["prompt_eval_count"]
    del body["eval_count"]
    mock_post.return_value = _mock_ollama_response(body=body)
    provider = OllamaProvider()
    result = provider.generate("x")
    assert result.input_tokens is None
    assert result.output_tokens is None


# ── Groq ──────────────────────────────────────────────────────────────────────

GROQ_RESPONSE = {
    "choices": [
        {"message": {"content": "  Diagnosis: pneumonia.  "}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
    "model": "llama-3.1-8b-instant",
}


def _mock_groq_response(status=200, body=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = body or GROQ_RESPONSE

    if status == 429:
        mock_resp.text = "Rate limit exceeded"
        mock_resp.raise_for_status.side_effect = None  # 429 handled before raise_for_status
    else:
        mock_resp.raise_for_status = MagicMock()

    return mock_resp


@patch("harness.providers.groq_provider.requests.post")
def test_groq_builds_chat_request(mock_post):
    mock_post.return_value = _mock_groq_response()
    provider = GroqProvider(api_key="test-key")
    provider.generate("Classify this.", max_tokens=10, temperature=0.0)

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["model"] == "llama-3.1-8b-instant"
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Classify this."
    assert payload["max_tokens"] == 10


@patch("harness.providers.groq_provider.requests.post")
def test_groq_parses_response(mock_post):
    mock_post.return_value = _mock_groq_response()
    provider = GroqProvider(api_key="test-key")
    result = provider.generate("Test prompt.")

    assert isinstance(result, GenerationResult)
    assert result.text == "Diagnosis: pneumonia."  # stripped
    assert result.provider == "groq"
    assert result.input_tokens == 100
    assert result.output_tokens == 5
    assert result.cost_usd > 0  # computed from token counts


@patch("harness.providers.groq_provider.requests.post")
def test_groq_raises_rate_limit_error_on_429(mock_post):
    mock_post.return_value = _mock_groq_response(status=429)
    provider = GroqProvider(api_key="test-key")

    # with_retry wraps _call; patch it so retry doesn't sleep
    with patch("harness.retry.time.sleep"):
        with pytest.raises(RateLimitError):
            provider.generate("Test prompt.")


def test_groq_raises_on_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        GroqProvider()


@patch("harness.providers.groq_provider.requests.post")
def test_groq_cost_computed_from_tokens(mock_post):
    mock_post.return_value = _mock_groq_response()
    provider = GroqProvider(api_key="test-key")
    result = provider.generate("Test.")
    # 100 input tokens * $0.05/M + 5 output tokens * $0.08/M
    expected = (100 / 1_000_000) * 0.05 + (5 / 1_000_000) * 0.08
    assert abs(result.cost_usd - expected) < 1e-10
