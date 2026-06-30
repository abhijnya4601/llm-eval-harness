import os
import time
import requests

from .base import BaseProvider, GenerationResult
from ..retry import with_retry

# Groq published rates as of 2025; free tier bills $0 but we track cost for at-scale projections
_INPUT_COST_PER_M = 0.05
_OUTPUT_COST_PER_M = 0.08

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(BaseProvider):
    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str | None = None):
        self._model = model
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Export it before running or add it to .env."
            )
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResult:
        return with_retry(self._call)(prompt, max_tokens, temperature)

    def _call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        t0 = time.perf_counter()
        response = requests.post(_GROQ_URL, json=payload, headers=self._headers, timeout=30)
        latency_ms = (time.perf_counter() - t0) * 1000

        if response.status_code == 429:
            raise RateLimitError(f"Groq rate limit: {response.text}", response.text)

        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        text = choice["message"]["content"].strip()
        usage = data.get("usage", {})
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")

        cost = 0.0
        if in_tok is not None and out_tok is not None:
            cost = (in_tok / 1_000_000) * _INPUT_COST_PER_M + (out_tok / 1_000_000) * _OUTPUT_COST_PER_M

        return GenerationResult(
            text=text,
            model=self._model,
            provider="groq",
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )


class RateLimitError(Exception):
    def __init__(self, message: str, body: str = ""):
        super().__init__(message)
        self.retry_after: float | None = _parse_retry_after(body)


def _parse_retry_after(body: str) -> float | None:
    import re
    match = re.search(r"try again in ([\d.]+)s", body)
    if match:
        return float(match.group(1)) + 0.5  # pad slightly so the window actually resets
    return None
