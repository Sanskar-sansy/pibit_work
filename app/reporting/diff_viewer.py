"""
Prompt diff viewer: computes and formats diffs between prompt versions.
"""
from __future__ import annotations
import difflib
from typing import Optional


def compute_diff(prompt_a: str, prompt_b: str, context_lines: int = 3) -> str:
    """Return a unified diff string between two prompts."""
    lines_a = prompt_a.splitlines(keepends=True)
    lines_b = prompt_b.splitlines(keepends=True)
    diff = difflib.unified_diff(
        lines_a, lines_b,
        fromfile="before",
        tofile="after",
        n=context_lines,
    )
    return "".join(diff)


def summarize_diff(prompt_a: str, prompt_b: str) -> dict:
    """Return a summary of changes between two prompts."""
    words_a = len(prompt_a.split())
    words_b = len(prompt_b.split())
    chars_a = len(prompt_a)
    chars_b = len(prompt_b)
    diff = compute_diff(prompt_a, prompt_b)
    added = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    return {
        "words_before": words_a,
        "words_after": words_b,
        "word_delta": words_b - words_a,
        "chars_before": chars_a,
        "chars_after": chars_b,
        "lines_added": added,
        "lines_removed": removed,
        "diff": diff,
    }
