#!/usr/bin/env python3
"""
Export DPO training data using REAL LLM failures as rejected responses.

Key improvements over export_rlhf_training_data.py:
1. Rejected = REAL LLM failures (1050 available), not synthetic corruptions
2. Chosen = Apriori + CoT (teaches reasoning, not just memorization)
3. Uses ALL datasets (Apriori is deterministic — no need for validation_passed=1)
4. Multiple rejected variants per dataset (different models failed differently)

Why real failures are better than synthetic:
- 99.5% of real errors are item_missing_in_row (hallucinated evidence)
- Synthetic corruptions create random noise that doesn't match real error distributions
- Model learns to avoid the EXACT mistakes LLMs actually make

Usage:
    python src/training/export_real_dpo_data.py --db runs.db --output data/dpo_real_v2.json
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.training_utils import (
    SYSTEM_PROMPT,
    load_csv_as_prompt,
    load_json_file,
    generate_cot,
    format_ground_truth_json,
    normalize_llm_output,
    estimate_tokens,
    build_user_message,
    resolve_data_path,
)
from src.utils.diversity_metrics import compute_diversity_report


def get_dataset_apriori_map(db_path: str) -> Dict[str, Dict]:
    """Get Apriori output info for each unique dataset."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT dataset_id, data_path, apriori_output_path, min_support,
               dataset_size_rows
        FROM runs
        WHERE apriori_output_path IS NOT NULL
        GROUP BY dataset_id
        ORDER BY dataset_id
    """)

    result = {row["dataset_id"]: dict(row) for row in cursor.fetchall()}
    conn.close()
    return result


def get_failed_llm_runs(db_path: str) -> List[Dict]:
    """Get all failed LLM runs that have actual output (llm_itemset_count > 0)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT dataset_id, llm_output_path, llm_model, llm_itemset_count
        FROM runs
        WHERE validation_passed = 0
          AND llm_itemset_count > 0
          AND llm_output_path IS NOT NULL
        ORDER BY dataset_id, llm_model
    """)

    result = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return result


def export_real_dpo(
    db_path: str = "runs.db",
    output_path: str = "data/dpo_real_v2.json",
    max_tokens: int = 3800,
    max_cot_items: int = 40,
    max_rejected_per_dataset: int = 3,
    report_path: str | None = None,
) -> None:
    """
    Export DPO training data with real LLM failures as rejected responses.

    For each failed LLM run:
    - chosen = Apriori ground truth with CoT reasoning
    - rejected = real LLM output (the actual mistakes the model made)

    Args:
        db_path: Path to runs.db
        output_path: Where to save the JSON output
        max_tokens: Maximum estimated tokens per example
        max_cot_items: Max itemsets to show in CoT trace
        max_rejected_per_dataset: Max rejected variants per dataset (avoid over-representation)
    """
    # Load data
    dataset_map = get_dataset_apriori_map(db_path)
    failed_runs = get_failed_llm_runs(db_path)
    print(f"📂 Found {len(dataset_map)} datasets, {len(failed_runs)} failed LLM runs")

    # Group failed runs by dataset
    fails_by_dataset: Dict[str, List[Dict]] = {}
    for frun in failed_runs:
        ds_id = frun["dataset_id"]
        fails_by_dataset.setdefault(ds_id, []).append(frun)

    dpo_pairs = []
    skipped_missing = 0
    skipped_tokens = 0
    skipped_limit = 0
    dataset_counts: Dict[str, int] = {}

    for ds_id, failed_list in fails_by_dataset.items():
        if ds_id not in dataset_map:
            skipped_missing += 1
            continue

        ds = dataset_map[ds_id]
        min_support = ds["min_support"] or 3

        # Resolve actual file paths (handles directory renames)
        resolved_data = resolve_data_path(ds["data_path"])
        resolved_apriori = Path(ds["apriori_output_path"])

        if not resolved_data.exists() or not resolved_apriori.exists():
            skipped_missing += 1
            continue

        # Load Apriori + CSV (once per dataset)
        try:
            csv_text, n_rows, n_cols = load_csv_as_prompt(str(resolved_data))
            apriori_items = load_json_file(str(resolved_apriori))
        except Exception:
            skipped_missing += 1
            continue

        if not apriori_items:
            continue

        # Build chosen response (Apriori + CoT)
        cot = generate_cot(apriori_items, n_rows, n_cols, min_support, max_cot_items)
        gt_json = format_ground_truth_json(apriori_items)
        chosen_content = f"<think>\n{cot}\n</think>\n{gt_json}"

        # Build user message
        user_content = build_user_message(csv_text, min_support)

        # Build prompt messages
        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Process each failed LLM run for this dataset
        count_for_ds = 0
        for frun in failed_list:
            if count_for_ds >= max_rejected_per_dataset:
                skipped_limit += 1
                continue

            llm_path = frun["llm_output_path"]
            if not Path(llm_path).exists():
                skipped_missing += 1
                continue

            try:
                llm_items = load_json_file(llm_path)
            except Exception:
                skipped_missing += 1
                continue

            if not llm_items:
                continue

            # Normalize LLM output to consistent format
            rejected_json = normalize_llm_output(llm_items)

            # Token check: DPOTrainer uses max(prompt+chosen, prompt+rejected)
            # not the sum of all three
            prompt_tokens = estimate_tokens(SYSTEM_PROMPT + user_content)
            chosen_tokens = estimate_tokens(chosen_content)
            rejected_tokens = estimate_tokens(rejected_json)
            max_seq_tokens = prompt_tokens + max(chosen_tokens, rejected_tokens)

            if max_seq_tokens > max_tokens:
                skipped_tokens += 1
                continue

            # Create DPO pair
            dpo_pairs.append({
                "prompt": prompt_msgs,
                "chosen": [{"role": "assistant", "content": chosen_content}],
                "rejected": [{"role": "assistant", "content": rejected_json}],
                "metadata": {
                    "dataset_id": ds_id,
                    "rejected_model": frun["llm_model"],
                    "rejected_type": "real_failure",
                    "n_itemsets_gt": len(apriori_items),
                    "n_itemsets_rejected": frun["llm_itemset_count"],
                },
            })
            count_for_ds += 1

        dataset_counts[ds_id] = count_for_ds

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dpo_pairs, f, indent=2, ensure_ascii=False)

    # Report
    unique_datasets = len([c for c in dataset_counts.values() if c > 0])
    print(f"\n✅ Generated {len(dpo_pairs)} real DPO pairs")
    print(f"   Unique datasets used: {unique_datasets}")
    print(f"   Avg pairs/dataset: {len(dpo_pairs) / max(1, unique_datasets):.1f}")
    print(f"   Skipped (missing files): {skipped_missing}")
    print(f"   Skipped (too long): {skipped_tokens}")
    print(f"   Skipped (per-dataset limit): {skipped_limit}")
    print(f"   Saved to: {output_path}")

    # Model distribution
    model_counts: Dict[str, int] = {}
    for pair in dpo_pairs:
        m = pair["metadata"]["rejected_model"]
        model_counts[m] = model_counts.get(m, 0) + 1
    print(f"\n📊 Rejected model distribution:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"   {model}: {count} ({count / len(dpo_pairs) * 100:.1f}%)")

    if dpo_pairs:
        report = {
            "output_path": output_path,
            "pairs": len(dpo_pairs),
            "unique_datasets": unique_datasets,
            "avg_pairs_per_dataset": round(len(dpo_pairs) / max(1, unique_datasets), 2),
            "rejected_model_distribution": model_counts,
            "chosen_diversity": compute_diversity_report(
                [pair["chosen"][0]["content"] for pair in dpo_pairs]
            ),
            "rejected_diversity": compute_diversity_report(
                [pair["rejected"][0]["content"] for pair in dpo_pairs]
            ),
        }
        report_target = Path(report_path) if report_path else Path(output_path).with_suffix(".report.json")
        report_target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"📝 Diversity report: {report_target}")


def main():
    parser = argparse.ArgumentParser(
        description="Export DPO data with real LLM failures as rejected"
    )
    parser.add_argument("--db", default="runs.db", help="Path to runs.db")
    parser.add_argument("--output", default="data/dpo_real_v2.json", help="Output path")
    parser.add_argument("--max-tokens", type=int, default=3800)
    parser.add_argument("--max-cot-items", type=int, default=40)
    parser.add_argument(
        "--max-rejected", type=int, default=3,
        help="Max rejected variants per dataset"
    )
    parser.add_argument(
        "--report-path", default=None,
        help="Optional path for a diversity/report sidecar JSON"
    )
    args = parser.parse_args()

    export_real_dpo(
        args.db, args.output, args.max_tokens, args.max_cot_items, args.max_rejected, args.report_path
    )


if __name__ == "__main__":
    main()
