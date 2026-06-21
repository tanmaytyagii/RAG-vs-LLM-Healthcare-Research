"""
visualize.py
============
Generates the static chart images required under "Visualizations":
  - Accuracy comparison chart
  - Hallucination reduction chart
  - Latency comparison chart
  - Safety score comparison chart

Reads aggregated results from results/metrics.csv (produced by
evaluation.py or app.py) and writes PNGs to results/plots/.

Usage:
    python visualize.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless backend, safe for servers/CI
import matplotlib.pyplot as plt
import pandas as pd

import config
from evaluation import aggregate_by_system
from logging_utils import get_logger

logger = get_logger(__name__)

# A calm, accessible two-color palette used consistently across all charts.
COLOR_LLM = "#5B8DEF"
COLOR_RAG = "#2EB872"


def _bar_chart(
    agg: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    lower_is_better: bool = False,
) -> Optional[Path]:
    """Render a single LLM-vs-RAG bar chart for one metric and save it."""
    if metric not in agg.columns or agg[metric].isna().all():
        logger.warning("Metric '%s' not present in aggregated data; skipping chart.", metric)
        return None

    systems = [s for s in ("LLM", "RAG") if s in agg.index]
    values = [agg.loc[s, metric] for s in systems]
    colors = [COLOR_LLM if s == "LLM" else COLOR_RAG for s in systems]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(systems, values, color=colors, width=0.5, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    suffix = "\n(lower is better)" if lower_is_better else ""
    ax.set_title(title + suffix, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    out_path = config.PLOTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved chart: %s", out_path)
    return out_path


def generate_all_charts(metrics_csv: Optional[Path] = None) -> list[Path]:
    """Generate all required comparison charts from results/metrics.csv.

    Args:
        metrics_csv: Optional override path; defaults to config.METRICS_CSV.

    Returns:
        List of paths to the generated PNG files (charts that were
        skipped due to missing data are omitted).
    """
    csv_path = metrics_csv or config.METRICS_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run evaluation.py or the Streamlit app to "
            "generate evaluation records first."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty; nothing to plot.")

    agg = aggregate_by_system(df)
    logger.info("Generating charts from %d records across systems: %s", len(df), list(agg.index))

    generated = []
    generated.append(
        _bar_chart(
            agg,
            "factual_accuracy",
            "Factual Accuracy: LLM vs. RAG",
            "Factual Accuracy (0-1)",
            "accuracy_comparison.png",
        )
    )
    generated.append(
        _bar_chart(
            agg,
            "hallucination_rate",
            "Hallucination Rate: LLM vs. RAG",
            "Hallucination Rate (0-1)",
            "hallucination_reduction.png",
            lower_is_better=True,
        )
    )
    generated.append(
        _bar_chart(
            agg,
            "latency_seconds",
            "Response Latency: LLM vs. RAG",
            "Latency (seconds)",
            "latency_comparison.png",
            lower_is_better=True,
        )
    )
    generated.append(
        _bar_chart(
            agg,
            "clinical_safety_score",
            "Clinical Safety Score: LLM vs. RAG",
            "Clinical Safety Score (0-1)",
            "safety_score_comparison.png",
        )
    )

    return [p for p in generated if p is not None]


if __name__ == "__main__":
    try:
        paths = generate_all_charts()
        print(f"Generated {len(paths)} chart(s):")
        for p in paths:
            print(f"  - {p}")
    except (FileNotFoundError, ValueError) as e:
        print(f"[Skipped] {e}")
