#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from harness.cache import DiskCache
from harness.runner import run
from harness.providers.ollama_provider import OllamaProvider
from harness.providers.groq_provider import GroqProvider
from harness.providers.gemini_provider import GeminiProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def build_providers(names: list[str]):
    providers = []
    for name in names:
        name = name.strip().lower()
        if name == "ollama":
            providers.append(OllamaProvider())
        elif name == "groq":
            providers.append(GroqProvider())
        elif name == "gemini":
            providers.append(GeminiProvider())
        else:
            raise ValueError(f"Unknown provider: {name!r}. Choose from: ollama, groq, gemini")
    return providers


def render_table(summaries) -> str:
    try:
        from tabulate import tabulate
        rows = []
        for s in summaries:
            rows.append([
                s.provider,
                s.model,
                s.n_examples,
                f"{s.mean_score:.3f}",
                f"{s.mean_latency_ms:.0f}",
                f"{s.p50_latency_ms:.0f}",
                f"{s.p99_latency_ms:.0f}",
                f"${s.total_cost_usd:.6f}",
                f"{s.cache_hit_rate:.0%}",
            ])
        headers = ["Provider", "Model", "N", "Score", "Avg ms", "P50 ms", "P99 ms", "Cost USD", "Cache%"]
        return tabulate(rows, headers=headers, tablefmt="github")
    except ImportError:
        # Hand-formatted fallback if tabulate is not installed
        lines = [
            f"{'Provider':<10} {'Model':<25} {'N':>3} {'Score':>7} {'Avg ms':>8} {'P50 ms':>8} {'P99 ms':>8} {'Cost USD':>12} {'Cache%':>7}"
        ]
        lines.append("-" * 100)
        for s in summaries:
            lines.append(
                f"{s.provider:<10} {s.model:<25} {s.n_examples:>3} {s.mean_score:>7.3f} "
                f"{s.mean_latency_ms:>8.0f} {s.p50_latency_ms:>8.0f} {s.p99_latency_ms:>8.0f} "
                f"${s.total_cost_usd:>11.6f} {s.cache_hit_rate:>6.0%}"
            )
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run LLM eval harness.")
    parser.add_argument(
        "--eval-set",
        default="evals/clinical_eval_set.jsonl",
        help="Path to JSONL eval set (default: evals/clinical_eval_set.jsonl)",
    )
    parser.add_argument(
        "--providers",
        default="ollama",
        help="Comma-separated list of providers: ollama,groq (default: ollama)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run on first N examples only (useful for fast iteration)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass cache reads — force fresh provider calls (still writes to cache)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Custom output path for the JSON report (default: reports/results_<timestamp>.json)",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        sys.exit(f"Eval set not found: {eval_path}")

    provider_names = args.providers.split(",")
    try:
        providers = build_providers(provider_names)
    except (ValueError, EnvironmentError) as e:
        sys.exit(str(e))

    cache = DiskCache()

    print(f"\nRunning eval: {eval_path.name}")
    print(f"Providers:    {', '.join(p.provider_name for p in providers)}")
    print(f"Cache reads:  {'disabled (--no-cache)' if args.no_cache else 'enabled'}")
    if args.limit:
        print(f"Limit:        first {args.limit} examples")
    print()

    records, summaries = run(
        eval_path=eval_path,
        providers=providers,
        cache=cache,
        limit=args.limit,
        use_cache=not args.no_cache,
        out_path=args.out,
    )

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(render_table(summaries))
    print()

    tasks = sorted({r.task for r in records})
    for task in tasks:
        print(f"\n── Task: {task} ──")
        task_records = [r for r in records if r.task == task]
        for provider_name in {r.provider for r in task_records}:
            pr = [r for r in task_records if r.provider == provider_name]
            mean_score = sum(r.score for r in pr) / len(pr)
            print(f"  {provider_name:<12} mean score: {mean_score:.3f}  (n={len(pr)})")


if __name__ == "__main__":
    main()
