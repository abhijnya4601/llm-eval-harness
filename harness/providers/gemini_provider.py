import os
import time
import requests

from .base import BaseProvider, GenerationResult
from ..retry import with_retry, RateLimitError

# Gemini 1.5 Flash paid-tier rates (USD per million tokens) for at-scale projections.
# Free tier bills $0.
_INPUT_COST_PER_M = 0.075
_OUTPUT_COST_PER_M = 0.30

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(BaseProvider):
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        self._model = model
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Get a free key at aistudio.google.com and add it to .env."
            )

    @property
    def provider_name(self) -> str:
        return "gemini"

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
        url = f"{_GEMINI_BASE}/{self._model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

        t0 = time.perf_counter()
        response = requests.post(
            url,
            json=payload,
            params={"key": self._api_key},
            timeout=30,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        if response.status_code == 429:
            raise RateLimitError(f"Gemini rate limit: {response.text}", response.text)

        if not response.ok:
            raise requests.HTTPError(
                f"Gemini {response.status_code}: {response.text}", response=response
            )
        data = response.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        usage = data.get("usageMetadata", {})
        in_tok = usage.get("promptTokenCount")
        out_tok = usage.get("candidatesTokenCount")

        cost = 0.0
        if in_tok is not None and out_tok is not None:
            cost = (in_tok / 1_000_000) * _INPUT_COST_PER_M + (out_tok / 1_000_000) * _OUTPUT_COST_PER_M

        return GenerationResult(
            text=text,
            model=self._model,
            provider="gemini",
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )
