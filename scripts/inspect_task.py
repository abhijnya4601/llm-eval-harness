#!/usr/bin/env python3
"""Inspect per-example results for a specific task from a saved report."""
import argparse
import json
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def latest_report() -> Path:
    reports = sorted(REPORTS_DIR.glob("results_*.json"))
    if not reports:
        sys.exit("No reports found in reports/")
    return reports[-1]


def main():
    parser = argparse.ArgumentParser(description="Inspect per-example results for a task.")
    parser.add_argument("--report", type=Path, default=None, help="Path to report JSON")
    parser.add_argument("--task", default="summarization", help="Task to inspect (default: summarization)")
    parser.add_argument("--provider", default=None, help="Filter by provider name")
    args = parser.parse_args()

    report_path = args.report or latest_report()
    data = json.loads(report_path.read_text())

    print(f"Report: {report_path.name}")
    print(f"Generated: {data.get('generated_at', 'unknown')}")
    print(f"Wall-clock: {data.get('wall_clock_seconds', '?')}s")
    print()

    records = [
        r for r in data["records"]
        if r["task"] == args.task and (args.provider is None or r["provider"] == args.provider)
    ]

    if not records:
        sys.exit(f"No records found for task={args.task!r}" + (f" provider={args.provider!r}" if args.provider else ""))

    records.sort(key=lambda r: (r["example_id"], r["provider"]))

    for rec in records:
        hit = rec.get("cache_hit", False)
        print(f"{'=' * 72}")
        print(f"  {rec['example_id']}  provider={rec['provider']}  score={rec['score']:.2f}  latency={rec['latency_ms']:.0f}ms  cache={'HIT' if hit else 'MISS'}")
        print(f"  method: {rec['score_method']}")
        print(f"  detail: {rec['score_detail']}")
        print()
        print(f"  OUTPUT:")
        for line in rec["output"].strip().splitlines():
            print(f"    {line}")
        print()


if __name__ == "__main__":
    main()
