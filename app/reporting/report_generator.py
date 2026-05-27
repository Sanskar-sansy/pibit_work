"""
Final experiment report generator.
Produces a Markdown report with scores, diffs, and plot links.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.reporting.diff_viewer import summarize_diff
from app.reporting.plots import (
    plot_mutation_strategy_performance,
    plot_per_field_scores,
    plot_score_curve,
)
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generates a comprehensive Markdown report for an optimization run."""

    def __init__(self, reports_dir: str = "./reports") -> None:
        self._dir = Path(reports_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        experiment_name: str,
        seed_prompt: str,
        best_prompt: str,
        trajectory_entries: list[dict],
        seed_scores: dict[str, Any],
        best_scores: dict[str, Any],
        test_scores: Optional[dict[str, Any]] = None,
        config_snapshot: Optional[dict] = None,
    ) -> str:
        """
        Generate and save a full Markdown report.

        Returns:
            Path to the generated report file.
        """
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_name = f"{experiment_name}_{ts}"
        plots_dir = self._dir / report_name / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots
        curve_path = plot_score_curve(
            trajectory_entries,
            str(plots_dir / "score_curve.png"),
            title=f"Score Curve — {experiment_name}",
        )
        field_path = plot_per_field_scores(
            seed_scores.get("per_field", {}),
            best_scores.get("per_field", {}),
            str(plots_dir / "per_field_scores.png"),
        )
        strat_path = plot_mutation_strategy_performance(
            trajectory_entries,
            str(plots_dir / "strategy_performance.png"),
        )

        diff_summary = summarize_diff(seed_prompt, best_prompt)
        n_accepted = sum(1 for e in trajectory_entries if e.get("accepted", False))
        n_total = len(trajectory_entries)

        lines = [
            f"# Prompt Optimization Report",
            f"",
            f"**Experiment:** `{experiment_name}`  ",
            f"**Generated:** {datetime.utcnow().isoformat()}  ",
            f"",
            f"---",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Seed Prompt | Best Prompt | Δ |",
            f"|--------|------------|-------------|---|",
        ]

        for metric in ("mean_f1", "mean_precision", "mean_recall", "mean_aggregate"):
            seed_val = seed_scores.get(metric, 0.0)
            best_val = best_scores.get(metric, 0.0)
            delta = best_val - seed_val
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| {metric.replace('_', ' ').title()} "
                f"| {seed_val:.4f} | {best_val:.4f} | {sign}{delta:.4f} |"
            )

        if test_scores:
            lines += [
                f"",
                f"### Final Test Set Scores",
                f"",
                f"| Metric | Score |",
                f"|--------|-------|",
            ]
            for k, v in test_scores.items():
                if isinstance(v, float):
                    lines.append(f"| {k} | {v:.4f} |")

        lines += [
            f"",
            f"---",
            f"",
            f"## Optimization Statistics",
            f"",
            f"- **Total iterations:** {n_total}",
            f"- **Accepted mutations:** {n_accepted} / {n_total}",
            f"- **Acceptance rate:** {n_accepted/n_total:.1%}" if n_total else "- **Acceptance rate:** N/A",
            f"",
            f"---",
            f"",
            f"## Prompt Diff",
            f"",
            f"| | Before | After |",
            f"|-|--------|-------|",
            f"| Words | {diff_summary['words_before']} | {diff_summary['words_after']} |",
            f"| Chars | {diff_summary['chars_before']} | {diff_summary['chars_after']} |",
            f"| Lines added | — | +{diff_summary['lines_added']} |",
            f"| Lines removed | {diff_summary['lines_removed']} | — |",
            f"",
            f"```diff",
            diff_summary["diff"] or "(no changes)",
            f"```",
            f"",
            f"---",
            f"",
            f"## Per-Field Scores",
            f"",
            f"| Field | Seed | Best | Δ |",
            f"|-------|------|------|---|",
        ]

        for field_name in seed_scores.get("per_field", {}).keys():
            sv = seed_scores["per_field"].get(field_name, 0.0)
            bv = best_scores.get("per_field", {}).get(field_name, 0.0)
            d = bv - sv
            sign = "+" if d >= 0 else ""
            lines.append(f"| {field_name} | {sv:.4f} | {bv:.4f} | {sign}{d:.4f} |")

        lines += [
            f"",
            f"---",
            f"",
            f"## Plots",
            f"",
            f"![Score Curve](plots/score_curve.png)",
            f"",
            f"![Per-Field Scores](plots/per_field_scores.png)",
            f"",
            f"![Strategy Performance](plots/strategy_performance.png)",
            f"",
            f"---",
            f"",
            f"## Best Prompt",
            f"",
            f"```",
            best_prompt,
            f"```",
            f"",
            f"---",
            f"",
            f"## Seed Prompt",
            f"",
            f"```",
            seed_prompt,
            f"```",
        ]

        report_text = "\n".join(lines)
        report_path = self._dir / report_name / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")

        logger.info(f"Report saved to: {report_path}")
        return str(report_path)
