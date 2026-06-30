from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    model: str
    provider: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float  # 0.0 for local; computed-at-scale for Groq free tier


class BaseProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResult:
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
