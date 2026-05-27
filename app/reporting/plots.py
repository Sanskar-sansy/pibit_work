"""
Visualization plots for optimization results.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def plot_score_curve(
    trajectory_entries: list[dict],
    output_path: str,
    title: str = "Optimization Score Curve",
) -> Optional[str]:
    """Plot F1 score over optimization iterations."""
    if not _MPL_AVAILABLE:
        return None

    iterations = [e["iteration"] for e in trajectory_entries]
    scores = [e["score"] for e in trajectory_entries]
    accepted = [e["accepted"] for e in trajectory_entries]

    # Running best
    running_best = []
    best = 0.0
    for s in scores:
        best = max(best, s)
        running_best.append(best)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if a else "#e74c3c" for a in accepted]
    ax.scatter(iterations, scores, c=colors, zorder=5, s=60, alpha=0.8)
    ax.plot(iterations, running_best, color="#3498db", linewidth=2, label="Running Best")
    ax.plot(iterations, scores, color="#95a5a6", linewidth=1, alpha=0.5, linestyle="--")

    accept_patch = mpatches.Patch(color="#2ecc71", label="Accepted")
    reject_patch = mpatches.Patch(color="#e74c3c", label="Rejected")
    ax.legend(handles=[accept_patch, reject_patch, plt.Line2D([0], [0], color="#3498db", linewidth=2, label="Running Best")])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_per_field_scores(
    per_field_before: dict[str, float],
    per_field_after: dict[str, float],
    output_path: str,
) -> Optional[str]:
    """Bar chart comparing per-field scores before and after optimization."""
    if not _MPL_AVAILABLE:
        return None

    fields = list(per_field_before.keys())
    before_vals = [per_field_before.get(f, 0.0) for f in fields]
    after_vals = [per_field_after.get(f, 0.0) for f in fields]

    x = range(len(fields))
    fig, ax = plt.subplots(figsize=(max(8, len(fields) * 1.5), 5))
    width = 0.35
    ax.bar([i - width/2 for i in x], before_vals, width, label="Seed Prompt", color="#e74c3c", alpha=0.8)
    ax.bar([i + width/2 for i in x], after_vals, width, label="Optimized Prompt", color="#2ecc71", alpha=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(fields, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Per-Field Score: Before vs After Optimization")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_mutation_strategy_performance(
    trajectory_entries: list[dict],
    output_path: str,
) -> Optional[str]:
    """Bar chart of average score per mutation strategy."""
    if not _MPL_AVAILABLE:
        return None

    strategy_scores: dict[str, list[float]] = {}
    for e in trajectory_entries:
        strat = e.get("mutation_strategy") or "seed"
        strategy_scores.setdefault(strat, []).append(e["score"])

    strategies = list(strategy_scores.keys())
    avg_scores = [sum(v) / len(v) for v in strategy_scores.values()]
    colors = ["#3498db" if s != "seed" else "#95a5a6" for s in strategies]

    fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 1.5), 5))
    bars = ax.bar(strategies, avg_scores, color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("Mutation Strategy")
    ax.set_ylabel("Avg F1 Score")
    ax.set_title("Average Score per Mutation Strategy")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=30, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path
