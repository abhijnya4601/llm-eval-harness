# LLM Eval Harness

> **All clinical text in this repo is synthetic and fictional. No real patient data or PHI is used anywhere.**

---

## What this is

This is a benchmarking harness that runs the same set of NLP tasks against two different LLM providers and produces a side-by-side comparison of quality, latency, and cost. It is designed to be run entirely for free:

- **Ollama** runs a quantized model locally on your machine. No internet required, no cost, no rate limits. Slower because it runs on CPU/GPU locally.
- **Groq** is a cloud inference API with a free tier. Much faster than local inference because their hardware is purpose-built for it, but subject to rate limits on the free plan.

The eval set covers synthetic clinical NLP tasks — summarizing clinical notes, extracting structured fields from unstructured text, and classifying clinical statements by urgency. This domain is chosen because the tasks have clearly defined right answers that are easy to score automatically without needing a human or a judge model.

The point of the project is to demonstrate how a rigorous eval harness works: consistent interfaces across providers, caching so you don't waste quota re-running the same prompts, retry logic for transient failures, and deterministic scoring that produces real numbers you can defend.

---

## How it works end to end

```
scripts/run_eval.py        ← you run this
        │
        ▼
harness/runner.py          ← loads the eval set, loops over examples × providers
        │
        ├── harness/cache.py      ← checks SQLite before calling a provider
        │                            on a hit: returns stored result instantly
        │                            on a miss: calls the provider, stores result
        │
        ├── harness/providers/
        │       ├── ollama_provider.py   ← POST http://localhost:11434/api/generate
        │       └── groq_provider.py    ← POST https://api.groq.com/openai/v1/chat/completions
        │               └── harness/retry.py  ← wraps Groq calls with exponential backoff
        │
        └── harness/scorer.py     ← scores each output against its reference
                │
                ▼
        harness/runner.py  ← aggregates per-provider stats
                │
                ▼
        stdout table  +  reports/results_<timestamp>.json
```

Both providers implement the same `BaseProvider` interface and return the same `GenerationResult` dataclass. The runner and scorer never check which provider is being used — they just call `.generate()` and get back the same shape. This means adding a new provider (OpenAI, a local GGUF model, etc.) only requires a new file in `harness/providers/` with no changes anywhere else.

---

## What each file does

### `harness/providers/base.py`
Defines the interface every provider must implement. Contains `GenerationResult` (a dataclass with `text`, `model`, `provider`, `latency_ms`, `input_tokens`, `output_tokens`, `cost_usd`) and `BaseProvider` (an abstract base class with a single required method: `generate()`).

### `harness/providers/ollama_provider.py`
Calls the local Ollama server at `http://localhost:11434/api/generate`. Reads token counts from Ollama's response fields (`eval_count`, `prompt_eval_count`). Cost is always `$0.00`.

### `harness/providers/groq_provider.py`
Calls the Groq chat completions API. Reads your API key from the `GROQ_API_KEY` environment variable and raises a clear error if it's missing. Computes a projected cost from token counts using Groq's published per-token rates — the free tier charges nothing, but the field shows what the same run would cost at commercial scale.

### `harness/retry.py`
A decorator that wraps any function with exponential backoff retry. Retries on rate limit errors (429) and transient network failures. Passes through 4xx client errors immediately since those aren't transient. When Groq returns a 429, its response body includes a suggested wait time ("Please try again in 5.33s") — the retry reads that and waits the right amount instead of guessing.

### `harness/cache.py`
**Uses SQLite** via Python's built-in `sqlite3` module. Stores cached results in `cache/cache.db`. The cache key is a SHA-256 hash of `provider:model:prompt:max_tokens:temperature` — so the same prompt to the same model with the same settings always resolves to the same key. On a cache hit, the stored `GenerationResult` (including original latency) is returned directly. The runner excludes cache hits from latency statistics so the reported numbers reflect actual inference time, not replay time.

### `harness/runner.py`
Loads the eval set from JSONL, builds a prompt for each example based on task type, checks the cache, calls the provider if needed, scores the output, and aggregates results per provider. Writes a full JSON report to `reports/` after each run.

### `harness/scorer.py`
Three scoring modes:
- **Keyword rubric** (summarization): checks what fraction of required phrases from the rubric appear in the model output. Scores 0.0–1.0 with equal weight per phrase.
- **JSON field matching** (extraction): parses the model's output as JSON (handles raw JSON, markdown code fences, and embedded JSON in prose), then compares each expected field against the reference using case-insensitive substring matching. Partial credit per field.
- **Label classification**: checks if the expected label (urgent / routine / informational) appears in the output. Falls back to checking known paraphrases for 0.8 partial credit.

### `evals/clinical_eval_set.jsonl`
25 examples in JSONL format. Each line has: `id`, `task`, `input` (the synthetic clinical text), `reference` (the correct answer), and `rubric` (scoring instructions). Covers 8 summarization examples, 7 extraction examples, and 10 classification examples. All patient names and data are invented.

### `scripts/run_eval.py`
The CLI entry point. Parses arguments, loads providers, runs the eval, and prints the results table. Supports `--no-cache` to force fresh provider calls (still writes to cache so follow-up runs get hits) and `--out` to save the report to a specific path.

### `scripts/inspect_task.py`
Loads a saved report JSON and prints per-example output, score, and scorer detail for a given task. Useful for diagnosing why scores are high or low on specific examples.

```bash
python scripts/inspect_task.py --task summarization --report reports/results_cold_run.json
```

### `scripts/plot_results.py`
Reads a report JSON and generates a 2×3 bar-chart grid comparing providers across overall score, average latency, projected cost, and per-task scores. Saves a PNG to `reports/`.

---

## Prerequisites

You need Python 3.11+. Check with:

```bash
python3 --version
```

---

## Setup

### Step 1 — Clone and create a Python environment

```bash
git clone <your-repo-url>
cd llm-eval-harness

python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Step 2 — Install and start Ollama

Ollama runs the local model. Download it from [ollama.com/download](https://ollama.com/download) or via Homebrew:

```bash
brew install ollama
```

Start the server (keep this running in a separate terminal):

```bash
ollama serve
```

Pull the model (about 2 GB, one-time download):

```bash
ollama pull llama3.2:3b
```

### Step 3 — Set up Groq (optional, for the comparison)

Skip this step if you only want to run Ollama.

1. Sign up at [console.groq.com](https://console.groq.com)
2. Go to **API Keys** and create a key — it starts with `gsk_`
3. Copy the example env file and add your key:

```bash
cp .env.example .env
```

Open `.env` and set:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

Do not commit `.env` — it is in `.gitignore`.

---

## Running the eval

Make sure your `.venv` is activated (`source .venv/bin/activate`) and Ollama is running (`ollama serve` in another terminal).

```bash
# Ollama only — all 25 examples
python scripts/run_eval.py --providers ollama

# Groq only
python scripts/run_eval.py --providers groq

# Both providers side by side
python scripts/run_eval.py --providers ollama,groq

# First 5 examples only — useful for quickly testing that everything works
python scripts/run_eval.py --providers ollama --limit 5

# Force fresh calls, ignore anything in the cache
python scripts/run_eval.py --providers ollama --no-cache
```

After the run, a JSON report is saved to `reports/results_<timestamp>.json` with the full per-example breakdown.

---

## Running tests

No providers need to be running. All HTTP calls are mocked.

```bash
pytest
pytest --cov=harness    # with line coverage
```

41 tests cover the cache (roundtrip, key collisions, cache miss), providers (request shape, response parsing, 429 handling, missing API key), and scorer (keyword rubric, JSON extraction, classification, JSON parser edge cases).

---

## Sample results

Run on Apple M2, `llama3.2:3b` local vs. `llama-3.1-8b-instant` on Groq. Cold run (all cache
MISS, 47.9s wall-clock) followed by a cached replay (all cache HIT, 0.0s wall-clock):

```
| Provider | Model                |  N | Score | Avg ms | P50 ms | P99 ms |   Cost USD | Cache% |
|----------|----------------------|---:|------:|-------:|-------:|-------:|-----------:|-------:|
| groq     | llama-3.1-8b-instant | 25 | 0.743 |    254 |    268 |    402 | $0.000301  |     0% |
| ollama   | llama3.2:3b          | 25 | 0.730 |   1656 |   1756 |   4315 | $0.000000  |     0% |

── Task: classification ──
  groq         mean score: 0.700  (n=10)
  ollama       mean score: 0.700  (n=10)

── Task: extraction ──
  groq         mean score: 0.607  (n=7)
  ollama       mean score: 0.607  (n=7)

── Task: summarization ──
  groq         mean score: 0.917  (n=8)
  ollama       mean score: 0.875  (n=8)
```

**What the numbers say:**
- Groq is ~6.5× faster (254ms vs 1,656ms average) at a projected cost of $0.000301 for 25 examples — effectively free at this scale
- Classification and extraction scores are identical across both providers for these tasks — both model sizes handle structured responses well
- Groq edges ahead in summarization (0.917 vs 0.875) because the 8B model more consistently includes all three required elements in every summary
- Cache speedup: the cached replay of the same 50 (provider × example) pairs took 0.0s wall-clock vs 47.9s cold — a complete elimination of inference cost on repeated runs
- Groq P99 (402ms) is 1.6× its P50 — a tight distribution driven by the first network request. Ollama P99 (4,315ms) is the cold model-load penalty on S001; steady-state is 1,545–2,388ms for summarization

## Investigation findings

Two findings documents in `evals/` record the root-cause analysis behind the initial numbers:

**[evals/SUMMARIZATION_FINDINGS.md](evals/SUMMARIZATION_FINDINGS.md)** — Why early summarization
scores were 0.2–0.3 instead of 0.8+. Root cause: the rubric scorer was checking for abstract
category labels ("chief complaint", "follow-up plan") while models produce fluent prose that
describes the content without using those labels. Fix: added a clinical paraphrase map so
"presents with" satisfies "chief complaint", "follow up in N weeks" satisfies "follow-up plan", etc.

**[evals/LATENCY_FINDINGS.md](evals/LATENCY_FINDINGS.md)** — What drives Ollama's P99 spike
(4,315ms vs P50 of 1,756ms). Two causes: (1) S001 is the first request — Ollama loads model
weights into GPU memory on the first call, adding ~2,000ms TTFT that does not repeat; (2)
extraction examples with 4-medication JSON outputs (E004, E006) are the next-highest latency
because output length scales inference time. Classification examples (single-word outputs) are
consistently under 325ms.

---

## Project structure

```
harness/
  providers/
    base.py              GenerationResult dataclass + BaseProvider abstract class
    ollama_provider.py   Local Ollama inference via HTTP
    groq_provider.py     Groq cloud API, with rate-limit aware retry
  cache.py               SQLite-backed result cache (cache/cache.db)
  retry.py               Exponential backoff decorator
  runner.py              Eval loop, aggregation, report writer
  scorer.py              Keyword, JSON field, and classification scorers
evals/
  clinical_eval_set.jsonl     25 synthetic clinical NLP examples
  SUMMARIZATION_FINDINGS.md   Root-cause analysis of summarization score calibration
  LATENCY_FINDINGS.md         Investigation of Ollama P99 latency spike
scripts/
  run_eval.py            CLI entry point
  inspect_task.py        Per-example output inspector
  plot_results.py        matplotlib bar-chart report generator
tests/
  test_cache.py
  test_providers.py
  test_scorer.py
reports/                 Generated JSON reports — gitignored, reports/.gitkeep committed
cache/                   SQLite database — gitignored
```

---

## Known limitations

- **Keyword scorer requires rubric calibration.** The scorer matches rubric phrases against model output. If a rubric uses abstract category labels ("chief complaint") while models write fluent prose ("presents with chest tightness"), scores are deflated even when outputs are correct. The harness includes a clinical paraphrase map that handles this for the included eval set, but new rubrics need similar expansion. A model that parrots rubric labels verbatim would get a perfect score; ROUGE or BERTScore would be more robust but add heavy ML dependencies.

- **Small eval set.** 25 examples is enough to see directional differences but not enough for statistical significance. A production harness would need hundreds of examples and confidence intervals on score differences.

- **Groq free tier has a 6,000 TPM limit.** The retry logic handles it, but large runs will spend time waiting. The retry reads Groq's suggested wait time from the response body and sleeps that long rather than guessing.

- **Single-turn prompts only.** Each example is one prompt, one response. Clinical NLP often benefits from multi-turn or chain-of-thought prompting, which would require extending the provider interface to accept a `messages` list.

---

## What I'd add with more time

| Area | What and why |
|---|---|
| Cache backend | Swap SQLite for Redis to support concurrent parallel runners without write contention |
| More providers | OpenAI, local GGUF models via llama.cpp, Anthropic — same provider interface, new file each |
| Better scoring | ROUGE-L for summarization, BERTScore for semantic similarity, LLM-as-judge with a separate model |
| Statistics | Bootstrap confidence intervals on score differences; McNemar test for classification |
| Observability | OpenTelemetry spans per provider call so you can see latency breakdown in a trace |
| CI | GitHub Actions: run the test suite on every push, nightly eval run against Ollama in a container |
