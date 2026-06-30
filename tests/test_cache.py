import pytest
from pathlib import Path
from harness.cache import DiskCache
from harness.providers.base import GenerationResult


@pytest.fixture
def tmp_cache(tmp_path):
    db = tmp_path / "test_cache.db"
    cache = DiskCache(db_path=db)
    yield cache
    cache.close()


def _make_result(**overrides) -> GenerationResult:
    defaults = dict(
        text="Patient has hypertension.",
        model="llama3.2:3b",
        provider="ollama",
        latency_ms=312.5,
        input_tokens=42,
        output_tokens=10,
        cost_usd=0.0,
    )
    defaults.update(overrides)
    return GenerationResult(**defaults)


def test_roundtrip(tmp_cache):
    key = DiskCache.make_key("ollama", "llama3.2:3b", "Summarize this.", 512, 0.0)
    result = _make_result()
    tmp_cache.set(key, result)
    retrieved = tmp_cache.get(key)
    assert retrieved is not None
    assert retrieved.text == result.text
    assert retrieved.latency_ms == result.latency_ms
    assert retrieved.cost_usd == result.cost_usd


def test_cache_miss_returns_none(tmp_cache):
    key = DiskCache.make_key("ollama", "llama3.2:3b", "Does not exist.", 512, 0.0)
    assert tmp_cache.get(key) is None


def test_key_different_for_different_params(tmp_cache):
    key_a = DiskCache.make_key("ollama", "llama3.2:3b", "Prompt A", 512, 0.0)
    key_b = DiskCache.make_key("ollama", "llama3.2:3b", "Prompt B", 512, 0.0)
    assert key_a != key_b


def test_key_different_for_different_providers(tmp_cache):
    key_ollama = DiskCache.make_key("ollama", "llama3.2:3b", "Same prompt", 512, 0.0)
    key_groq = DiskCache.make_key("groq", "llama3.2:3b", "Same prompt", 512, 0.0)
    assert key_ollama != key_groq


def test_key_different_for_different_temperature(tmp_cache):
    key_zero = DiskCache.make_key("ollama", "llama3.2:3b", "Same", 512, 0.0)
    key_one = DiskCache.make_key("ollama", "llama3.2:3b", "Same", 512, 1.0)
    assert key_zero != key_one


def test_overwrite_on_duplicate_key(tmp_cache):
    key = DiskCache.make_key("ollama", "m", "p", 512, 0.0)
    r1 = _make_result(text="First response.")
    r2 = _make_result(text="Second response.")
    tmp_cache.set(key, r1)
    tmp_cache.set(key, r2)
    retrieved = tmp_cache.get(key)
    assert retrieved.text == "Second response."


def test_clear(tmp_cache):
    key = DiskCache.make_key("ollama", "m", "p", 512, 0.0)
    tmp_cache.set(key, _make_result())
    tmp_cache.clear()
    assert tmp_cache.get(key) is None


def test_none_tokens_roundtrip(tmp_cache):
    key = DiskCache.make_key("ollama", "m", "p", 512, 0.0)
    result = _make_result(input_tokens=None, output_tokens=None)
    tmp_cache.set(key, result)
    retrieved = tmp_cache.get(key)
    assert retrieved.input_tokens is None
    assert retrieved.output_tokens is None
