#!/usr/bin/env python3
"""
Generate Chain-of-Thought SFT training data from Apriori ground truth.

Key improvements over export_training_data.py:
1. Generates CoT reasoning traces (model learns HOW to find itemsets, not just WHAT)
2. Uses ALL 500 datasets (Apriori is deterministic — no need for validation_passed=1)
3. Compact system prompt (~150 tokens vs 6000 tokens)
4. Token-aware filtering (skips examples that won't fit in max_seq_length)

Usage:
    python src/training/generate_cot_sft_data.py --db runs.db --output data/sft_cot_v2.json
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict
import argparse
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.training_utils import (
    SYSTEM_PROMPT,
    load_csv_as_prompt,
    load_json_file,
    generate_cot,
    format_ground_truth_json,
    estimate_tokens,
    build_user_message,
    resolve_data_path,
    calculate_token_budget,
)
from src.utils.diversity_metrics import compute_diversity_report


def get_all_datasets_with_apriori(db_path: str) -> List[Dict]:
    """
    Get one Apriori output per unique dataset from the DB.
    Since Apriori is deterministic, any model's run works — we just need the file paths.
    This gives us ALL 500 datasets, not just the 307 that passed validation.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get one run per dataset (any model, just need apriori path + csv path)
    cursor.execute("""
        SELECT
            dataset_id,
            dataset_name,
            data_path,
            apriori_output_path,
            min_support,
            dataset_size_rows
        FROM runs
        WHERE apriori_output_path IS NOT NULL
        GROUP BY dataset_id
        ORDER BY dataset_id
    """)

    datasets = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return datasets


def generate_sft_examples(
    db_path: str = "runs.db",
    output_path: str = "data/sft_cot_v3.json",
    max_tokens: int = 1800,
    max_cot_items: int = 40,
    report_path: str | None = None,
) -> None:
    """
    Generate SFT training examples with Chain-of-Thought reasoning.

    For each dataset:
    1. Load CSV → user prompt
    2. Load Apriori output → ground truth
    3. Generate CoT trace from Apriori evidence
    4. Format as ChatML messages

    Args:
        db_path: Path to runs.db
        output_path: Where to save the JSON output
        max_tokens: Maximum estimated tokens per example (filters out too-long examples)
        max_cot_items: Max itemsets to show in CoT trace (abbreviates large outputs)
    """
    datasets = get_all_datasets_with_apriori(db_path)
    print(f"📂 Found {len(datasets)} unique datasets in DB")

    examples = []
    skipped_missing = 0
    skipped_tokens = 0
    skipped_empty = 0

    for ds in datasets:
        dataset_id = ds["dataset_id"]
        data_path = ds["data_path"]
        apriori_path = ds["apriori_output_path"]
        min_support = ds["min_support"] or 3

        # Resolve actual file paths (handles directory renames)
        resolved_data = resolve_data_path(data_path)
        resolved_apriori = Path(apriori_path)

        if not resolved_data.exists():
            skipped_missing += 1
            continue
        if not resolved_apriori.exists():
            skipped_missing += 1
            continue

        # Load data
        try:
            csv_text, n_rows, n_cols = load_csv_as_prompt(str(resolved_data))
            apriori_items = load_json_file(str(resolved_apriori))
        except Exception as e:
            print(f"  ⚠️ Error loading {dataset_id}: {e}")
            skipped_missing += 1
            continue

        if not apriori_items:
            skipped_empty += 1
            continue

        # Generate CoT reasoning from Apriori trace
        cot = generate_cot(apriori_items, n_rows, n_cols, min_support, max_cot_items)
        ground_truth_json = format_ground_truth_json(apriori_items)

        # Build messages
        user_content = build_user_message(csv_text, min_support)
        assistant_content = f"<think>\n{cot}\n</think>\n{ground_truth_json}"

        # Token budget check
        total_text = SYSTEM_PROMPT + user_content + assistant_content
        est_tokens = estimate_tokens(total_text)

        if est_tokens > max_tokens:
            skipped_tokens += 1
            continue

        # Create SFT example
        example = {
            "dataset_id": dataset_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            # Ground truth stored separately (used by GRPO later)
            "ground_truth": ground_truth_json,
            "metadata": {
                "n_rows": n_rows,
                "n_cols": n_cols,
                "n_itemsets": len(apriori_items),
                "min_support": min_support,
                "est_tokens": est_tokens,
            },
        }
        examples.append(example)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    # Report
    print(f"\n✅ Generated {len(examples)} SFT-CoT examples")
    print(f"   Skipped (missing files): {skipped_missing}")
    print(f"   Skipped (empty Apriori): {skipped_empty}")
    print(f"   Skipped (too long >{max_tokens} tokens): {skipped_tokens}")
    print(f"   Saved to: {output_path}")

    if examples:
        tokens = [e["metadata"]["est_tokens"] for e in examples]
        itemsets = [e["metadata"]["n_itemsets"] for e in examples]
        print(f"\n📊 Stats:")
        print(f"   Token range: {min(tokens)} – {max(tokens)} (avg {sum(tokens)//len(tokens)})")
        print(f"   Itemset range: {min(itemsets)} – {max(itemsets)} (avg {sum(itemsets)//len(itemsets)})")

        report = {
            "output_path": output_path,
            "examples": len(examples),
            "token_stats": {
                "min": min(tokens),
                "max": max(tokens),
                "avg": round(sum(tokens) / len(tokens), 1),
            },
            "itemset_stats": {
                "min": min(itemsets),
                "max": max(itemsets),
                "avg": round(sum(itemsets) / len(itemsets), 1),
            },
            "assistant_diversity": compute_diversity_report(
                [e["messages"][-1]["content"] for e in examples]
            ),
        }
        report_target = Path(report_path) if report_path else Path(output_path).with_suffix(".report.json")
        report_target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   Diversity report: {report_target}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Chain-of-Thought SFT training data from Apriori ground truth"
    )
    parser.add_argument("--db", default="runs.db", help="Path to runs.db")
    parser.add_argument("--output", default="data/sft_cot_v3.json", help="Output JSON path")
    parser.add_argument(
        "--max-tokens", type=int, default=3500,
        help="Max estimated tokens per example (default: 3500 for 4096 seq length)"
    )
    parser.add_argument(
        "--max-cot-items", type=int, default=40,
        help="Max itemsets to show in CoT trace (abbreviates larger outputs)"
    )
    parser.add_argument(
        "--report-path", default=None,
        help="Optional path for a diversity/report sidecar JSON"
    )
    args = parser.parse_args()

    generate_sft_examples(args.db, args.output, args.max_tokens, args.max_cot_items, args.report_path)


if __name__ == "__main__":
    main()
