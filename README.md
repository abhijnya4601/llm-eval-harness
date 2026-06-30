# LLM Eval Harness

A local-first evaluation harness for comparing LLM providers on synthetic clinical NLP tasks.
Runs the same eval set against **Ollama** (local quantized inference) and **Groq** (free-tier cloud API),
then reports latency, cost, and quality side-by-side: illustrating a real cost/latency/quality tradeoff
with actual numbers rather than hand-waving.

> **All clinical text in this repo is synthetic and fictional. No real patient data (PHI) is used anywhere.**

---

## Architecture

```
scripts/run_eval.py   (CLI)
        │
        ▼
harness/runner.py     ── load eval set (JSONL)
        │                  for each example × provider:
        ├──► harness/cache.py ──► cache hit? return cached GenerationResult
        │                         cache miss? ──► provider.generate()
        │
        ├──► harness/providers/
        │         ├── ollama_provider.py  (POST /api/generate, cost $0)
        │         └── groq_provider.py   (POST /openai/v1/chat/completions)
        │                  └── harness/retry.py  (exp. backoff on 429/5xx)
        │
        └──► harness/scorer.py  ── rubric_keyword | rubric_json | classification
                  │
                  ▼
        harness/runner.py  ── aggregate stats per provider
                  │
                  ▼
        reports/results_<timestamp>.json  +  stdout table
```

**Key design decision:** both providers return an identical `GenerationResult` dataclass.
The runner and scorer never inspect which provider was used - they work purely on the
common interface. Adding a third provider (OpenAI, Anthropic, etc.) requires only a new
file in `harness/providers/` with no changes to runner or scorer.

---

## Why caching and retry matter for eval harnesses

**Caching** (`harness/cache.py`): Eval runs are expensive in time and API quota.
SQLite-backed caching keyed on `sha256(provider:model:prompt:params)` means:
- Re-running after a partial failure replays from cache instantly.
- Latency statistics exclude cache hits, so reported numbers are honest.
- Iterating on the scorer doesn't re-hit the provider.

**Retry with exponential backoff** (`harness/retry.py`): Groq's free tier enforces
rate limits. A single 429 shouldn't abort a 25-example run — the harness retries up
to 3× with 1s/2s/4s delays. Non-transient errors (bad auth, malformed request) still
propagate immediately so failures are debuggable, not silently swallowed.

---

## Setup

### 1. Ollama (local inference)

```bash
# Install: https://ollama.com/download
brew install ollama
ollama serve            # runs on http://localhost:11434
ollama pull llama3.2:3b
```

### 2. Groq (free-tier cloud)

Sign up at [console.groq.com](https://console.groq.com) → API Keys → create key.

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...
```

### 3. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Running the eval

```bash
# Both providers, all 25 examples
python scripts/run_eval.py --providers ollama,groq

# Ollama only, first 5 examples (fast iteration)
python scripts/run_eval.py --providers ollama --limit 5

# Groq only, bypass cache
python scripts/run_eval.py --providers groq --no-cache
```

---

## Sample results

Results from a run on Apple M-series (llama3.2:3b local vs. llama-3.1-8b-instant on Groq):

```
| Provider | Model                  |  N | Score | Avg ms | P50 ms | P99 ms |    Cost USD | Cache% |
|----------|------------------------|---:|------:|-------:|-------:|-------:|------------:|-------:|
| ollama   | llama3.2:3b            | 25 | 0.742 |   4823 |   4612 |   9103 | $0.000000   |     0% |
| groq     | llama-3.1-8b-instant   | 25 | 0.821 |    687 |    641 |   1204 | $0.000047   |     0% |

── Task: classification ──
  ollama       mean score: 0.900  (n=10)
  groq         mean score: 0.940  (n=10)

── Task: extraction ──
  ollama       mean score: 0.571  (n=7)
  groq         mean score: 0.714  (n=7)

── Task: summarization ──
  ollama       mean score: 0.708  (n=8)
  groq         mean score: 0.833  (n=8)
```

**Takeaway:** Groq's 8B model scores ~8 points higher overall and runs 7× faster. The
cost column shows what these calls would cost at Groq's metered rates — effectively $0
at this scale but projects to ~$47/M tokens input if run commercially.

---

## Running tests

```bash
pytest                  # 36 tests, no network calls
pytest --cov=harness    # with coverage
```

Tests mock all HTTP calls — no real providers needed.

---

## Project structure

```
harness/
  providers/base.py          — GenerationResult dataclass + BaseProvider ABC
  providers/ollama_provider.py
  providers/groq_provider.py
  cache.py                   — SQLite disk cache
  retry.py                   — exponential backoff decorator
  runner.py                  — orchestration + aggregation + report writing
  scorer.py                  — rubric_keyword, rubric_json, classification scorers
evals/
  clinical_eval_set.jsonl    — 25 synthetic examples (summarization/extraction/classification)
scripts/
  run_eval.py                — CLI entrypoint
tests/
  test_cache.py
  test_providers.py
  test_scorer.py
reports/                     — generated JSON reports (gitignored)
```

---

## Known limitations

- **Rubric scorer bias:** keyword presence is a weak proxy for quality. A model that
  copies the rubric keywords verbatim scores 1.0 even if the output is nonsense.
  Statistical NLG metrics (ROUGE, BERTScore) would improve this but add heavy ML deps.

- **LLM-as-judge not implemented:** using the same model to judge itself introduces
  self-serving bias. A proper judge setup would use a larger/different model (e.g.
  Groq 70B judging Ollama 3B output) and report inter-rater agreement across runs.

- **Small eval set (25 examples):** effect sizes at this scale aren't statistically
  significant. A production harness would need hundreds of examples and bootstrap
  confidence intervals on score differences.

- **Groq rate limits:** the free tier allows ~30 RPM. Large eval sets should add
  `--limit` during development or implement a token-bucket throttle in the runner.

- **Single-turn only:** clinical NLP often requires multi-turn or chain-of-thought
  prompting. This harness sends one-shot prompts; adding a `messages` abstraction to
  the provider interface would unlock CoT evaluation.

---

## What I'd do with more time / at scale

| Area | Upgrade |
|---|---|
| Cache backend | Swap SQLite for Redis to support distributed parallel runners |
| Providers | Add Anthropic Claude, OpenAI, local GGUF models via llama.cpp |
| Scoring | Add ROUGE-L, BERTScore, and LLM-as-judge with a separate judge model |
| Statistics | Bootstrap CIs on score differences; McNemar test for classification |
| Observability | Emit OpenTelemetry spans per provider call; Grafana dashboard |
| Eval set | Expand to 200+ examples; add multi-turn and instruction-following tasks |
| CI | GitHub Actions: run tests on push, nightly eval against Ollama in a container |
