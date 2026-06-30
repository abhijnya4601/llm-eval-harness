import time
import os
import requests

from .base import BaseProvider, GenerationResult


class OllamaProvider(BaseProvider):
    def __init__(self, model: str = "llama3.2:3b", host: str | None = None):
        self._model = model
        self._host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._url = f"{self._host.rstrip('/')}/api/generate"

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResult:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        t0 = time.perf_counter()
        response = requests.post(self._url, json=payload, timeout=120)
        latency_ms = (time.perf_counter() - t0) * 1000

        response.raise_for_status()
        data = response.json()

        return GenerationResult(
            text=data.get("response", "").strip(),
            model=self._model,
            provider="ollama",
            latency_ms=latency_ms,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            cost_usd=0.0,
        )
