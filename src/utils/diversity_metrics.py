from __future__ import annotations

from collections import Counter
from statistics import mean
import re
from typing import Iterable


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for repetition and template-collapse checks."""
    return WHITESPACE_RE.sub(" ", text.strip().lower())


def prefix_signature(text: str, max_chars: int = 120) -> str:
    """Create a lightweight signature from the beginning of a text."""
    normalized = normalize_text(text)
    return normalized[:max_chars]


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_diversity_report(
    texts: Iterable[str],
    *,
    prefix_chars: int = 120,
    top_k: int = 5,
) -> dict:
    """
    Compute simple diversity and template-collapse diagnostics.

    Inspired by FinePhrase's observation that overly uniform generations can
    hurt downstream performance despite looking cleaner.
    """
    text_list = [text for text in texts if text and text.strip()]
    total = len(text_list)
    if total == 0:
        return {
            "total_texts": 0,
            "avg_chars": 0.0,
            "min_chars": 0,
            "max_chars": 0,
            "unique_full_text_rate": 0.0,
            "unique_prefix_rate": 0.0,
            "template_collapse_rate": 0.0,
            "most_common_prefix_count": 0,
            "most_common_prefix_rate": 0.0,
            "most_common_prefix_examples": [],
            "duplicate_full_text_count": 0,
        }

    normalized = [normalize_text(text) for text in text_list]
    prefixes = [prefix_signature(text, max_chars=prefix_chars) for text in text_list]
    lengths = [len(text) for text in text_list]

    full_counter = Counter(normalized)
    prefix_counter = Counter(prefixes)
    duplicate_full_text_count = sum(count - 1 for count in full_counter.values() if count > 1)
    repeated_prefix_total = sum(count for count in prefix_counter.values() if count > 1)
    most_common_prefix = prefix_counter.most_common(top_k)

    return {
        "total_texts": total,
        "avg_chars": round(mean(lengths), 1),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "unique_full_text_rate": round(_safe_rate(len(full_counter), total), 4),
        "unique_prefix_rate": round(_safe_rate(len(prefix_counter), total), 4),
        "template_collapse_rate": round(_safe_rate(repeated_prefix_total, total), 4),
        "most_common_prefix_count": most_common_prefix[0][1],
        "most_common_prefix_rate": round(_safe_rate(most_common_prefix[0][1], total), 4),
        "most_common_prefix_examples": [
            {"prefix": prefix, "count": count}
            for prefix, count in most_common_prefix
        ],
        "duplicate_full_text_count": duplicate_full_text_count,
    }