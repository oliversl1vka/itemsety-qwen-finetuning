#!/usr/bin/env python3
"""
Shared utilities for training data generation.

Contains the compact system prompt, CSV loading, CoT generation,
and ground truth formatting used across SFT, DPO, and GRPO data pipelines.

v3 Council Update (2026-03-09):
  - Concise CoT format: column-grouped, counts only in <think>, evidence in JSON
  - RESULT SUMMARY termination signal to prevent repetition loops
  - Token budget calculation per dataset
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ── Compact system prompt for training (~150 tokens) ─────────────────────────
# Much shorter than the 318-line extractor_system_prompt.md.
# Optimized for small models: clear rules, minimal fluff.
SYSTEM_PROMPT = (
    "You are a frequent itemset extractor. Given CSV transaction data and a "
    "minimum support count, identify all itemsets whose items co-occur in at "
    "least that many rows.\n\n"
    "Rules:\n"
    "1. Scan single items, pairs, and triples (up to size 3)\n"
    "2. Count = number of distinct rows containing ALL items in the itemset\n"
    "3. Only report itemsets with count >= min_support\n"
    "4. Canonicalize items: lowercase, trimmed, sorted alphabetically\n"
    "5. Row references: \"Row N\" format, 1-based indexing\n\n"
    "Think step by step inside <think> tags, then output ONLY a JSON array:\n"
    '[{"itemset": ["item1", "item2"], "count": N, "rows": ["Row 1", "Row 3"]}]'
)


def load_csv_as_prompt(csv_path: str) -> Tuple[str, int, int]:
    """
    Load CSV and format as user-readable text.

    Returns: (formatted_text, n_rows, n_cols)
    """
    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape

    rows_text = []
    for idx, row in df.iterrows():
        items = [str(v).strip() for v in row.values if pd.notna(v) and str(v).strip()]
        rows_text.append(f"Row {idx + 1}: {', '.join(items)}")

    return "\n".join(rows_text), n_rows, n_cols


def load_json_file(path: str) -> Any:
    """Load and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_cot(
    apriori_itemsets: List[Dict],
    n_rows: int,
    n_cols: int,
    min_support: int,
    max_items_in_cot: int = 40,
) -> str:
    """
    Generate CONCISE Chain-of-Thought reasoning from Apriori ground truth.

    v3 Council format (2026-03-09):
    - Column-grouped singles: one line per column, not per value
    - Compact notation: col:val=count(R1,R2,R3)✓
    - FREQUENT SINGLES/PAIRS summary lines as natural termination signals
    - RESULT SUMMARY termination marker
    - No verbose evidence_rows in <think> — only in final JSON

    This format is ~60% shorter than v2, drastically reducing repetition loops.
    """
    # Group by size
    by_size: Dict[int, List[Dict]] = {}
    for item in apriori_itemsets:
        size = item.get("size", len(item["itemset"]))
        by_size.setdefault(size, []).append(item)

    lines = [f"Dataset: {n_rows} rows, {n_cols} cols. min_support={min_support}."]
    lines.append("")

    # ── Singles: column-grouped compact format ─────────────────────────────
    singles = by_size.get(1, [])
    if singles:
        lines.append("SINGLES SCAN:")
        # Group singles by column prefix (e.g., "age:" → ["age:young", "age:old"])
        col_groups: Dict[str, List[Dict]] = {}
        for s in singles:
            item_name = s["itemset"][0] if s["itemset"] else ""
            col = item_name.split(":")[0] if ":" in item_name else item_name
            col_groups.setdefault(col, []).append(s)

        for col, items in col_groups.items():
            parts = []
            for it in items:
                item_name = it["itemset"][0]
                rows = it.get("unique_rows", it.get("rows", []))
                count = it.get("unique_row_count", it.get("count", len(rows)))
                # Compact row refs: R1,R2,R3 (max 5 shown)
                row_nums = []
                for r in rows[:5]:
                    if isinstance(r, str) and r.startswith("Row "):
                        row_nums.append(f"R{r[4:]}")
                    else:
                        row_nums.append(str(r))
                row_str = ", ".join(row_nums)
                if len(rows) > 5:
                    row_str += f" +{len(rows) - 5}more"
                mark = "✓" if count >= min_support else "✗"
                parts.append(f"{item_name}={count}({row_str}){mark}")
            lines.append(f"  {' | '.join(parts)}")

        frequent_singles = [
            s["itemset"][0] for s in singles
            if (s.get("unique_row_count", s.get("count", 0)) >= min_support)
        ]
        lines.append(f"\nFREQUENT SINGLES [{len(frequent_singles)}]: {', '.join(frequent_singles[:20])}")
        if len(frequent_singles) > 20:
            lines[-1] += f" (+{len(frequent_singles) - 20} more)"
    else:
        lines.append("SINGLES SCAN: none meet threshold")
        frequent_singles = []

    # ── Pairs: compact format ──────────────────────────────────────────────
    pairs = by_size.get(2, [])
    if pairs:
        lines.append("\nPAIRS SCAN:")
        frequent_pairs = []
        # Sort by count descending
        pairs_sorted = sorted(pairs, key=lambda x: x.get("count", 0), reverse=True)
        for p in pairs_sorted:
            items_str = ",".join(p["itemset"])
            rows = p.get("unique_rows", p.get("rows", []))
            count = p.get("unique_row_count", p.get("count", len(rows)))
            row_nums = []
            for r in rows[:4]:
                if isinstance(r, str) and r.startswith("Row "):
                    row_nums.append(f"R{r[4:]}")
                else:
                    row_nums.append(str(r))
            row_str = ", ".join(row_nums)
            if len(rows) > 4:
                row_str += f" +{len(rows) - 4}more"
            lines.append(f"  ({items_str})={count}({row_str})✓")
            frequent_pairs.append(f"{{{items_str}}}")

        lines.append(f"\nFREQUENT PAIRS [{len(pairs)}]: {', '.join(frequent_pairs[:15])}")
        if len(frequent_pairs) > 15:
            lines[-1] += f" (+{len(frequent_pairs) - 15} more)"
    else:
        lines.append("\nPAIRS SCAN: none meet threshold")

    # ── Triples: compact format ────────────────────────────────────────────
    triples = by_size.get(3, [])
    if triples:
        lines.append("\nTRIPLES SCAN:")
        triples_sorted = sorted(triples, key=lambda x: x.get("count", 0), reverse=True)
        for t in triples_sorted:
            items_str = ",".join(t["itemset"])
            rows = t.get("unique_rows", t.get("rows", []))
            count = t.get("unique_row_count", t.get("count", len(rows)))
            row_nums = []
            for r in rows[:4]:
                if isinstance(r, str) and r.startswith("Row "):
                    row_nums.append(f"R{r[4:]}")
                else:
                    row_nums.append(str(r))
            row_str = ", ".join(row_nums)
            if len(rows) > 4:
                row_str += f" +{len(rows) - 4}more"
            lines.append(f"  ({items_str})={count}({row_str})✓")
    elif len(frequent_singles) >= 3:
        lines.append("\nTRIPLES: none meet threshold")

    # ── RESULT SUMMARY — critical termination signal ───────────────────────
    n_singles = len(by_size.get(1, []))
    n_pairs = len(by_size.get(2, []))
    n_triples = len(by_size.get(3, []))
    total = len(apriori_itemsets)
    lines.append(
        f"\nRESULT SUMMARY: {n_singles} singles + {n_pairs} pairs + "
        f"{n_triples} triples = {total} frequent itemsets"
    )

    return "\n".join(lines)


# Keep the legacy verbose format available for comparison
def generate_cot_legacy(
    apriori_itemsets: List[Dict],
    n_rows: int,
    n_cols: int,
    min_support: int,
    max_items_in_cot: int = 40,
) -> str:
    """
    Legacy v2 CoT format (verbose, with evidence_rows in <think>).
    Kept for comparison — DO NOT USE for v3 training (causes repetition loops).
    """
    by_size: Dict[int, List[Dict]] = {}
    for item in apriori_itemsets:
        size = item.get("size", len(item["itemset"]))
        by_size.setdefault(size, []).append(item)

    lines = [f"Dataset: {n_rows} rows × {n_cols} columns, min_support={min_support}"]
    lines.append("")

    total = len(apriori_itemsets)
    abbreviated = total > max_items_in_cot

    if abbreviated:
        lines.append(
            f"Scanning for frequent items and combinations... "
            f"({total} total found)"
        )
        lines.append("")

    size_labels = {1: "Singles", 2: "Pairs", 3: "Triples"}
    items_shown = 0

    for size in sorted(by_size.keys()):
        items = by_size[size]
        label = size_labels.get(size, f"Size-{size}")
        items_sorted = sorted(items, key=lambda x: x.get("count", 0), reverse=True)

        if abbreviated:
            budget = max(3, (max_items_in_cot - items_shown) // max(1, len(by_size) - size + 1))
            show_items = items_sorted[:budget]
            lines.append(f"{label} ({len(items)} found, showing top {len(show_items)}):")
        else:
            show_items = items_sorted
            lines.append(f"{label} (count≥{min_support}):")

        for it in show_items:
            itemset_str = json.dumps(it["itemset"], ensure_ascii=False)
            rows = it.get("unique_rows", it.get("rows", []))
            count = it.get("unique_row_count", it.get("count", len(rows)))
            row_refs = ", ".join(rows[:15])
            if len(rows) > 15:
                row_refs += f" (+{len(rows) - 15} more)"
            lines.append(f"- {itemset_str}: {row_refs} → {count} ✓")
            items_shown += 1

        if abbreviated and len(items) > len(show_items):
            lines.append(f"  ... and {len(items) - len(show_items)} more {label.lower()}")
        lines.append("")

    lines.append(f"Found {total} frequent itemsets total.")
    return "\n".join(lines)


def calculate_token_budget(n_rows: int, n_cols: int, n_unique_vals: int = 0) -> dict:
    """
    Calculate dataset-specific token budgets for generation.

    Council advice: dynamically limit max_new_tokens based on expected output size.
    Prevents infinite loops by hard-capping generation length.

    Returns dict with separate budgets for think phase and JSON phase.
    """
    if n_unique_vals == 0:
        n_unique_vals = n_cols * 3  # rough estimate: 3 values per column

    # Estimate frequent itemsets (heuristic: ~30% of possible meet threshold)
    singles_estimate = n_unique_vals
    pairs_estimate = min(singles_estimate * (singles_estimate - 1) // 4, 50)
    triples_estimate = min(pairs_estimate // 3, 20)

    # Think phase: concise format ~20 tokens per single, ~25 per pair, ~30 per triple
    think_tokens = (
        singles_estimate * 20 +
        pairs_estimate * 25 +
        triples_estimate * 30 +
        100  # headers + summary
    )

    # JSON phase: ~60 tokens per itemset
    total_itemsets = singles_estimate + pairs_estimate + triples_estimate
    json_tokens = total_itemsets * 60

    # Apply 1.5x safety margin
    think_budget = int(think_tokens * 1.5)
    json_budget = int(json_tokens * 1.5)
    total_budget = think_budget + json_budget

    return {
        "think_tokens": max(300, min(think_budget, 3000)),
        "json_tokens": max(300, min(json_budget, 3000)),
        "total_tokens": max(512, min(total_budget, 6000)),
        "estimated_itemsets": total_itemsets,
    }


def format_ground_truth_json(apriori_itemsets: List[Dict]) -> str:
    """
    Format Apriori output as clean JSON for assistant response.
    Strips internal fields (support, size, unique_rows) to match expected output format.
    """
    clean = []
    for it in apriori_itemsets:
        rows = it.get("unique_rows", it.get("rows", []))
        count = it.get("unique_row_count", it.get("count", len(rows)))
        clean.append({
            "itemset": it["itemset"],
            "count": count,
            "rows": rows,
        })
    return json.dumps(clean, ensure_ascii=False)


def normalize_llm_output(llm_items: List[Dict]) -> str:
    """
    Normalize LLM extractor output to a consistent JSON format.
    Handles different key names (evidence_rows vs rows, etc.)
    """
    clean = []
    for it in llm_items:
        itemset = it.get("itemset", it.get("items", []))
        count = it.get("count", 0)
        rows = it.get("rows", it.get("evidence_rows", it.get("evidence_transactions", [])))

        # Normalize row format: ensure "Row N" strings
        normalized_rows = []
        for r in rows:
            if isinstance(r, int):
                normalized_rows.append(f"Row {r}")
            elif isinstance(r, str):
                normalized_rows.append(r)
            else:
                normalized_rows.append(str(r))

        clean.append({
            "itemset": itemset,
            "count": count,
            "rows": normalized_rows,
        })
    return json.dumps(clean, ensure_ascii=False)


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate (~4 chars per token for English/JSON mixed content).
    Conservative — real tokenizers usually produce fewer tokens.
    """
    return len(text) // 4


def build_user_message(csv_text: str, min_support: int) -> str:
    """Build the user message for itemset extraction."""
    return (
        f"Find all frequent itemsets with minimum support count = {min_support} "
        f"in this dataset:\n\n{csv_text}"
    )


def resolve_data_path(path_str: str) -> Path:
    """
    Resolve a DB-stored data_path to an actual file on disk.

    Handles directory renames (e.g., datasets_v2_batch500 → datasets_v2)
    and /tmp paths from ephemeral batch runs.
    """
    p = Path(path_str)
    if p.exists():
        return p

    # Try known directory mappings
    filename = p.name
    candidate_dirs = [
        "data/datasets_v2",
        "data/batch50_gpt4o",
    ]
    for d in candidate_dirs:
        candidate = Path(d) / filename
        if candidate.exists():
            return candidate

    return p  # Return original; caller checks .exists()
