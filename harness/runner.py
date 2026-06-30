import json
import logging
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .cache import DiskCache
from .providers.base import BaseProvider, GenerationResult
from .scorer import ScoreResult, score as compute_score

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"


@dataclass
class EvalExample:
    id: str
    task: str
    input: str
    reference: str
    rubric: str


@dataclass
class EvalRecord:
    example_id: str
    task: str
    provider: str
    model: str
    prompt: str
    output: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float
    cache_hit: bool
    score: float
    score_method: str
    score_detail: str


@dataclass
class ProviderSummary:
    provider: str
    model: str
    n_examples: int
    mean_score: float
    mean_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    total_cost_usd: float
    cache_hit_rate: float
    total_input_tokens: int
    total_output_tokens: int


def load_eval_set(path: Path) -> list[EvalExample]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            examples.append(EvalExample(**{k: data[k] for k in EvalExample.__dataclass_fields__}))
    return examples


def run(
    eval_path: Path,
    providers: Sequence[BaseProvider],
    cache: DiskCache | None = None,
    limit: int | None = None,
    use_cache: bool = True,
    out_path: Path | None = None,
) -> tuple[list[EvalRecord], list[ProviderSummary]]:
    examples = load_eval_set(eval_path)
    if limit:
        examples = examples[:limit]

    records: list[EvalRecord] = []
    wall_t0 = time.perf_counter()

    for example in examples:
        prompt = _build_prompt(example)
        for provider in providers:
            result, hit = _call_provider(provider, prompt, cache, use_cache)
            score_result = compute_score(example.task, result.text, example.reference, example.rubric)

            record = EvalRecord(
                example_id=example.id,
                task=example.task,
                provider=provider.provider_name,
                model=provider.model_name,
                prompt=prompt,
                output=result.text,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                cache_hit=hit,
                score=score_result.score,
                score_method=score_result.method,
                score_detail=score_result.detail,
            )
            records.append(record)

            logger.info(
                "[%s/%s] provider=%s score=%.2f latency=%.0fms cache=%s",
                example.id, example.task, provider.provider_name,
                score_result.score, result.latency_ms, "HIT" if hit else "MISS",
            )

    wall_seconds = time.perf_counter() - wall_t0
    logger.info("Total wall-clock time: %.1fs", wall_seconds)

    summaries = _aggregate(records)
    _write_report(records, summaries, wall_seconds, out_path)
    return records, summaries


def _build_prompt(example: EvalExample) -> str:
    task_instructions = {
        "summarization": (
            "You are a clinical documentation assistant. "
            "Summarize the following clinical note concisely. "
            "Focus on the chief complaint, current medications, and follow-up plan. "
            "Respond in 2-4 sentences.\n\n"
            f"Clinical Note:\n{example.input}\n\nSummary:"
        ),
        "extraction": (
            "You are a clinical data extraction system. "
            "Extract the following fields from the clinical note and return ONLY a JSON object "
            "with keys: diagnosis, medications, follow_up_date, allergies.\n\n"
            f"Clinical Note:\n{example.input}\n\nJSON:"
        ),
        "classification": (
            "You are a clinical triage assistant. "
            "Classify the following clinical statement as one of: urgent, routine, or informational. "
            "Respond with exactly one word.\n\n"
            f"Statement: {example.input}\n\nClassification:"
        ),
    }
    return task_instructions.get(example.task, f"Task: {example.task}\n\nInput: {example.input}\n\nResponse:")


def _call_provider(
    provider: BaseProvider,
    prompt: str,
    cache: DiskCache | None,
    use_cache: bool,
) -> tuple[GenerationResult, bool]:
    key = DiskCache.make_key(provider.provider_name, provider.model_name, prompt, 512, 0.0) if cache else None

    if cache and use_cache and key:
        cached = cache.get(key)
        if cached is not None:
            return cached, True

    result = provider.generate(prompt, max_tokens=512, temperature=0.0)

    if cache and key:
        cache.set(key, result)

    return result, False


def _aggregate(records: list[EvalRecord]) -> list[ProviderSummary]:
    from itertools import groupby
    summaries = []
    keyfn = lambda r: (r.provider, r.model)
    sorted_records = sorted(records, key=keyfn)

    for (provider, model), group in groupby(sorted_records, key=keyfn):
        grp = list(group)
        live = [r for r in grp if not r.cache_hit]
        latencies = [r.latency_ms for r in live] if live else [r.latency_ms for r in grp]

        summaries.append(ProviderSummary(
            provider=provider,
            model=model,
            n_examples=len(grp),
            mean_score=statistics.mean(r.score for r in grp),
            mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0.0,
            p99_latency_ms=sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0.0,
            total_cost_usd=sum(r.cost_usd for r in grp),
            cache_hit_rate=sum(1 for r in grp if r.cache_hit) / len(grp),
            total_input_tokens=sum(r.input_tokens or 0 for r in grp),
            total_output_tokens=sum(r.output_tokens or 0 for r in grp),
        ))

    return summaries


def _write_report(
    records: list[EvalRecord],
    summaries: list[ProviderSummary],
    wall_seconds: float = 0.0,
    out_path: Path | None = None,
) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_path or (REPORTS_DIR / f"results_{ts}.json")

    report = {
        "generated_at": ts,
        "wall_clock_seconds": round(wall_seconds, 2),
        "summaries": [asdict(s) for s in summaries],
        "records": [asdict(r) for r in records],
    }
    path.write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s  (wall-clock: %.1fs)", path, wall_seconds)
    return path
