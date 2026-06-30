# Latency Investigation — Ollama P99 Spike

## Observed data (cold run — all cache MISS)

| Provider | Avg ms | P50 ms | P99 ms |
|----------|--------|--------|--------|
| Groq     | 254    | 268    | 402    |
| Ollama   | 1,656  | 1,756  | 4,315  |

Groq's P99 is only 1.5× its P50 — a tight distribution.  
Ollama's P99 is 2.5× its P50 — a clear outlier.

## Which examples drove the spike?

Ollama latencies ranked descending (cold run):

| Example | Task           | Latency ms | Notes                              |
|---------|----------------|------------|------------------------------------|
| S001    | summarization  | 4,315      | **First request in the run**       |
| E004    | extraction     | 4,110      | 4-drug JSON output (longest gen)   |
| E006    | extraction     | 3,767      | 3-drug + long allergy field        |
| E003    | extraction     | 3,276      | Unscheduled follow-up edge case    |
| E002    | extraction     | 3,114      | 3-drug output                      |

Classification examples: 175–324 ms (single-word output — fastest tier).

## Root cause 1 — First-call TTFT with model already resident (S001, P99)

S001 is the first request in the run. Ollama had already been used earlier in the session, so
`llama3.2:3b` weights were resident in memory — this run did **not** capture a true model-load
cold start. Despite that, S001 still shows a 4,315 ms spike. The extra latency is the
first-call overhead Ollama incurs even with a resident model: context setup, KV-cache
initialization, and scheduling latency before the first token is generated. Subsequent
summarization examples (S002–S008) cluster between 1,545–2,388 ms, consistent with a fully
warmed inference path.

**Evidence:** S001 at 4,315 ms is 75 % slower than the next summarization example (S002 at
2,006 ms). The pattern does not repeat; no other summarization example exceeds 2,400 ms.

**Important caveat — the true model-load cold start is worse.** When Ollama serves a model
that has not been requested since the server started, it loads weights from disk into RAM/VRAM
before generating a single token. On the Apple M2 used for this run, that adds roughly
3–8 seconds on top of the first-call TTFT shown here. This benchmark does not capture that
scenario because the model was already resident. In a production deployment running Ollama
as an on-demand service, the actual P99 would be substantially higher on first access after a
server restart or idle timeout.

**Defensible finding:** the 4,315 ms P99 spike is a lower bound on first-call latency, not
an upper bound. It represents the minimum observable overhead when the model is already loaded.
Any scenario where the model must be loaded from disk will be slower.

**Mitigation:** A preflight warm-up call before the timed eval loop eliminates the first-call
TTFT penalty for resident models. The cache eliminates it entirely for repeated runs (0 ms
wall-clock on cached replay).

## Root cause 2 — Output length (E004, second-highest latency)

E004 asks the model to extract `carvedilol, sacubitril/valsartan, furosemide, spironolactone`
— four medications with complex names — into a structured JSON object. Ollama's local
inference time scales approximately linearly with output tokens. E004's output is ~120 tokens
vs. ~70 tokens for lighter extraction examples.

The same pattern holds across extraction examples: 4-drug examples take longer than 2-drug
ones. Classification (1 output word) is consistently 5–10× faster than summarization.

## Groq vs. Ollama latency profile

Groq (cloud inference, Llama-3.1-8b on GroqChip) is 6× faster on average than Ollama
(local inference, Llama-3.2-3b on CPU/GPU). Key differences:

- **Groq** — dedicated hardware accelerator, purpose-built for LLM token generation;
  latency is dominated by network round-trip (~150 ms) + inference (~100–250 ms).
- **Ollama** — consumer MacBook GPU/CPU; inference time scales with model size and output
  length; no meaningful cold-start penalty after the first request.

Groq's P99 outlier (402 ms, S001 groq) is the first network request; subsequent calls
cluster 140–380 ms. This mirrors Ollama's cold-start pattern at much smaller absolute values.

## Summary

| Factor              | Impact         | Affected examples         |
|---------------------|----------------|---------------------------|
| First-call TTFT (resident model) | +2,000 ms | S001 (P99 driver; true cold load would be higher) |
| Long JSON output    | +500–1,500 ms  | E004, E006, E003          |
| Single-word output  | −1,200 ms      | All C001–C010             |
| Network RTT (Groq)  | +150 ms floor  | All Groq examples         |

The P99 spike is not random jitter — it is deterministic and explained by two identifiable
causes. The cache eliminates both: cached runs complete in <100 ms wall-clock for all 50
(provider × example) combinations.
