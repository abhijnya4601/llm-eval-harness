#!/usr/bin/env python3
"""Generate bar charts from the most recent eval report in reports/.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --report reports/results_20260630T090420Z.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — writes files, no display needed
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    sys.exit("matplotlib is not installed. Run: pip install matplotlib")

REPORTS_DIR = Path(__file__).parent.parent / "reports"

PROVIDER_COLORS = {
    "ollama": "#4C72B0",
    "groq":   "#DD8452",
    "gemini": "#55A868",
}


def load_report(path: Path | None) -> dict:
    if path:
        return json.loads(path.read_text())
    candidates = sorted(REPORTS_DIR.glob("results_*.json"), reverse=True)
    if not candidates:
        sys.exit("No reports found in reports/. Run the eval first.")
    return json.loads(candidates[0].read_text())


def bar_chart(
    ax,
    providers: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    fmt: str = "{:.3f}",
    colors: list[str] | None = None,
):
    colors = colors or [PROVIDER_COLORS.get(p, "#888888") for p in providers]
    bars = ax.bar(providers, values, color=colors, width=0.5, zorder=2)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            fmt.format(val),
            ha="center", va="bottom", fontsize=9,
        )


def plot(report: dict, out_path: Path):
    summaries = report["summaries"]
    providers = [s["provider"] for s in summaries]
    records = report["records"]
    tasks = sorted({r["task"] for r in records})

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("LLM Eval Harness — Provider Comparison", fontsize=13, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── Row 1: overall metrics ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bar_chart(ax1, providers,
              [s["mean_score"] for s in summaries],
              "Overall Score", "Score (0–1)", "{:.3f}")
    ax1.set_ylim(0, 1.1)

    ax2 = fig.add_subplot(gs[0, 1])
    bar_chart(ax2, providers,
              [s["mean_latency_ms"] for s in summaries],
              "Avg Latency (live calls)", "Milliseconds", "{:.0f} ms")

    ax3 = fig.add_subplot(gs[0, 2])
    costs = [s["total_cost_usd"] for s in summaries]
    bar_chart(ax3, providers, costs,
              "Projected Cost (25 examples)", "USD", "${:.6f}")

    # ── Row 2: per-task score breakdown ──────────────────────────────────────
    for col, task in enumerate(tasks):
        ax = fig.add_subplot(gs[1, col])
        task_scores = []
        for provider in providers:
            task_records = [r for r in records if r["task"] == task and r["provider"] == provider]
            mean = sum(r["score"] for r in task_records) / len(task_records) if task_records else 0.0
            task_scores.append(mean)
        bar_chart(ax, providers, task_scores,
                  f"Score — {task.capitalize()}", "Score (0–1)", "{:.3f}")
        ax.set_ylim(0, 1.1)

    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot eval results.")
    parser.add_argument("--report", type=Path, default=None,
                        help="Path to a specific results JSON (default: most recent)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: reports/chart_<timestamp>.png)")
    args = parser.parse_args()

    report = load_report(args.report)
    ts = report.get("generated_at", "unknown")
    out = args.out or (REPORTS_DIR / f"chart_{ts}.png")
    plot(report, out)


if __name__ == "__main__":
    main()
