#!/usr/bin/env python3
"""
Export RLHF training data from runs.db with preference pairs (chosen/rejected).

For RLHF, we need:
1. Prompt (CSV context + task)
2. Chosen response (Apriori ground truth - HIGH quality)
3. Rejected response (Synthetic mistakes - LOW quality)

Based on: https://github.com/opendilab/awesome-RLHF#dataset
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
import random
import copy


def load_csv_sample(csv_path: str, max_rows: int = 50) -> str:
    """Load CSV and format as text (limited rows to avoid token overflow)"""
    try:
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        sample_rows = min(max_rows, total_rows)

        # Format as readable transaction list
        rows_text = []
        for idx, row in df.head(sample_rows).iterrows():
            items = [str(v) for v in row.values if pd.notna(v) and str(v).strip()]
            rows_text.append(f"Row {idx+1}: {', '.join(items)}")

        header = f"Dataset: {Path(csv_path).name}\n"
        header += f"Dimensions: {total_rows} rows × {len(df.columns)} columns\n\n"

        if total_rows > sample_rows:
            footer = f"\n... ({total_rows - sample_rows} more rows not shown)"
        else:
            footer = ""

        return header + "\n".join(rows_text) + footer

    except Exception as e:
        return f"Error loading CSV: {e}"


def load_ground_truth(apriori_path: str) -> List[Dict[str, Any]]:
    """Load Apriori output (ground truth itemsets)"""
    try:
        with open(apriori_path, "r", encoding="utf-8") as f:
            itemsets = json.load(f)

        # Clean format: keep only itemset, count, rows
        cleaned = []
        for item in itemsets:
            cleaned.append(
                {
                    "itemset": item["itemset"],
                    "count": item["count"],
                    "rows": item["rows"],
                }
            )

        return cleaned

    except Exception as e:
        print(f"Warning: Failed to load {apriori_path}: {e}")
        return []


def generate_rejected_responses(
    ground_truth: List[Dict[str, Any]], 
    min_support: int,
    num_variants: int = 3
) -> List[Tuple[List[Dict[str, Any]], str]]:
    """
    Generate rejected (low-quality) responses by introducing common mistakes:
    
    1. **Hallucinations**: Add non-existent itemsets
    2. **Missing itemsets**: Remove some valid itemsets
    3. **Wrong counts**: Incorrect support counts
    4. **Wrong evidence**: Incorrect row references
    5. **Supersets/Subsets confusion**: Include redundant or incomplete sets
    6. **Format errors**: Malformed JSON structure
    
    Returns: List of (rejected_response, error_type) tuples
    """
    rejected_variants = []
    
    if not ground_truth:
        return rejected_variants
    
    # Get unique items from ground truth
    all_items = set()
    for gt in ground_truth:
        all_items.update(gt["itemset"])
    all_items = list(all_items)
    
    # Variant 1: HALLUCINATIONS - Add fake itemsets
    if num_variants >= 1:
        hallucinated = copy.deepcopy(ground_truth)
        # Add 1-3 fake itemsets
        num_fake = random.randint(1, min(3, len(all_items)))
        for _ in range(num_fake):
            fake_size = random.randint(2, 3)
            fake_itemset = random.sample(all_items, min(fake_size, len(all_items)))
            fake_count = random.randint(min_support, min_support + 5)
            fake_rows = [f"Row {random.randint(1, 30)}" for _ in range(fake_count)]
            
            hallucinated.append({
                "itemset": fake_itemset,
                "count": fake_count,
                "rows": fake_rows
            })
        
        rejected_variants.append((hallucinated, "hallucination"))
    
    # Variant 2: MISSING ITEMSETS - Remove valid itemsets
    if num_variants >= 2 and len(ground_truth) > 2:
        missing = copy.deepcopy(ground_truth)
        # Remove 20-40% of itemsets
        num_to_remove = max(1, int(len(missing) * random.uniform(0.2, 0.4)))
        indices_to_remove = random.sample(range(len(missing)), num_to_remove)
        missing = [item for i, item in enumerate(missing) if i not in indices_to_remove]
        
        rejected_variants.append((missing, "missing_itemsets"))
    
    # Variant 3: WRONG COUNTS - Incorrect support counts
    if num_variants >= 3:
        wrong_counts = copy.deepcopy(ground_truth)
        # Corrupt 30-50% of counts
        num_to_corrupt = max(1, int(len(wrong_counts) * random.uniform(0.3, 0.5)))
        indices = random.sample(range(len(wrong_counts)), num_to_corrupt)
        
        for idx in indices:
            # Either increase or decrease count incorrectly
            original_count = wrong_counts[idx]["count"]
            if random.random() < 0.5:
                # Underestimate
                wrong_counts[idx]["count"] = max(1, original_count - random.randint(1, 3))
            else:
                # Overestimate
                wrong_counts[idx]["count"] = original_count + random.randint(1, 5)
        
        rejected_variants.append((wrong_counts, "wrong_counts"))
    
    # Variant 4: WRONG EVIDENCE - Incorrect row references
    if num_variants >= 4:
        wrong_evidence = copy.deepcopy(ground_truth)
        # Corrupt 40-60% of evidence
        num_to_corrupt = max(1, int(len(wrong_evidence) * random.uniform(0.4, 0.6)))
        indices = random.sample(range(len(wrong_evidence)), num_to_corrupt)
        
        for idx in indices:
            count = wrong_evidence[idx]["count"]
            # Generate random wrong row numbers
            wrong_evidence[idx]["rows"] = [f"Row {random.randint(1, 50)}" for _ in range(count)]
        
        rejected_variants.append((wrong_evidence, "wrong_evidence"))
    
    # Variant 5: SUBSET/SUPERSET CONFUSION - Mix specific and general sets
    if num_variants >= 5 and len(ground_truth) > 1:
        confused = copy.deepcopy(ground_truth)
        
        # Add supersets of existing itemsets (should be filtered but aren't)
        if len(confused) > 0:
            base_itemset = random.choice(confused)
            if len(all_items) > len(base_itemset["itemset"]):
                extra_items = [item for item in all_items if item not in base_itemset["itemset"]]
                if extra_items:
                    superset = base_itemset["itemset"] + [random.choice(extra_items)]
                    confused.append({
                        "itemset": superset,
                        "count": base_itemset["count"] - 1,  # Slightly lower count
                        "rows": base_itemset["rows"][:-1] if len(base_itemset["rows"]) > 1 else base_itemset["rows"]
                    })
        
        rejected_variants.append((confused, "subset_superset_confusion"))
    
    # Variant 6: BELOW MIN SUPPORT - Include itemsets below threshold
    if num_variants >= 6:
        below_support = copy.deepcopy(ground_truth)
        # Add 1-2 itemsets with support < min_support
        num_fake = random.randint(1, 2)
        for _ in range(num_fake):
            fake_size = random.randint(2, 3)
            fake_itemset = random.sample(all_items, min(fake_size, len(all_items)))
            # Count below minimum
            fake_count = random.randint(1, min_support - 1)
            fake_rows = [f"Row {random.randint(1, 30)}" for _ in range(fake_count)]
            
            below_support.append({
                "itemset": fake_itemset,
                "count": fake_count,
                "rows": fake_rows
            })
        
        rejected_variants.append((below_support, "below_min_support"))
    
    return rejected_variants[:num_variants]


def export_rlhf_training_examples(
    db_path: str = "runs.db",
    output_dir: str = "data/rlhf_training_v1",
    max_csv_rows: int = 50,
    model_filter: str = "gpt_4_1",
    num_rejected_per_example: int = 3,
) -> None:
    """
    Export RLHF training data with preference pairs.
    
    Format follows HH-RLHF and Stanford SHP datasets:
    {
        "prompt": "CSV context + task description",
        "chosen": "Apriori ground truth (high quality)",
        "rejected": "Synthetic error (low quality)",
        "error_type": "hallucination|missing_itemsets|wrong_counts|...",
        "metadata": {...}
    }
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Query validated runs
    cursor.execute(
        """
        SELECT 
            dataset_id,
            dataset_name,
            data_path,
            apriori_output_path,
            llm_output_path,
            min_support,
            dataset_size_rows,
            dataset_hash
        FROM runs
        WHERE validation_passed = 1
        AND llm_model = ?
        ORDER BY dataset_id
    """,
        (model_filter,),
    )

    rlhf_examples = []
    skipped = 0

    for row in cursor.fetchall():
        dataset_id = row["dataset_id"]

        # Load CSV context
        csv_context = load_csv_sample(row["data_path"], max_csv_rows)

        # Load ground truth (Apriori output)
        ground_truth = load_ground_truth(row["apriori_output_path"])

        if not ground_truth:
            print(f"⚠️  Skipping {dataset_id}: No valid ground truth")
            skipped += 1
            continue

        # Create prompt (same for all variants)
        prompt = f"{csv_context}\n\n"
        prompt += (
            f"Find all frequent itemsets with minimum support count = "
            f"{row['min_support']}. "
            f"Return a JSON array with itemset, count, and rows fields."
        )

        # Chosen response (ground truth)
        chosen = json.dumps(ground_truth, ensure_ascii=False)

        # Generate rejected responses
        rejected_variants = generate_rejected_responses(
            ground_truth, 
            row['min_support'],
            num_rejected_per_example
        )

        # Create multiple RLHF pairs for this dataset
        for rejected_response, error_type in rejected_variants:
            rejected = json.dumps(rejected_response, ensure_ascii=False)
            
            rlhf_example = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "error_type": error_type,
                "metadata": {
                    "dataset_id": dataset_id,
                    "dataset_name": row["dataset_name"],
                    "dataset_hash": row["dataset_hash"],
                    "min_support": row["min_support"],
                    "total_rows": row["dataset_size_rows"],
                    "ground_truth_count": len(ground_truth),
                    "rejected_count": len(rejected_response),
                },
            }
            
            rlhf_examples.append(rlhf_example)

        # Save individual dataset file
        dataset_file = output_path / f"{dataset_id}_rlhf.json"
        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump({
                "prompt": prompt,
                "chosen": chosen,
                "rejected_variants": [
                    {"response": json.dumps(resp, ensure_ascii=False), "error_type": err}
                    for resp, err in rejected_variants
                ],
                "metadata": {
                    "dataset_id": dataset_id,
                    "dataset_name": row["dataset_name"],
                    "ground_truth_count": len(ground_truth),
                }
            }, f, indent=2, ensure_ascii=False)

    conn.close()

    # Save combined file
    combined_file = output_path / "all_rlhf_pairs.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(rlhf_examples, f, indent=2, ensure_ascii=False)

    # Analyze error type distribution
    error_type_counts = {}
    for ex in rlhf_examples:
        err_type = ex["error_type"]
        error_type_counts[err_type] = error_type_counts.get(err_type, 0) + 1

    # Save summary
    summary = {
        "total_rlhf_pairs": len(rlhf_examples),
        "unique_datasets": len(set(ex["metadata"]["dataset_id"] for ex in rlhf_examples)),
        "skipped": skipped,
        "avg_pairs_per_dataset": len(rlhf_examples) / max(1, len(set(ex["metadata"]["dataset_id"] for ex in rlhf_examples))),
        "error_type_distribution": error_type_counts,
        "model_filter": model_filter,
        "max_csv_rows": max_csv_rows,
        "output_directory": str(output_path),
    }

    summary_file = output_path / "rlhf_export_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ RLHF Export Complete!")
    print(f"   Total RLHF pairs: {len(rlhf_examples)}")
    print(f"   Unique datasets: {summary['unique_datasets']}")
    print(f"   Avg pairs/dataset: {summary['avg_pairs_per_dataset']:.1f}")
    print(f"   Skipped: {skipped}")
    print(f"   Output: {output_path}")
    print(f"\nError type distribution:")
    for err_type, count in error_type_counts.items():
        print(f"   - {err_type}: {count} pairs ({count/len(rlhf_examples)*100:.1f}%)")
    print(f"\nFiles created:")
    print(f"   - {summary['unique_datasets']} individual JSON files (per dataset)")
    print(f"   - all_rlhf_pairs.json (all pairs combined)")
    print(f"   - rlhf_export_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Export RLHF training data from runs.db")
    parser.add_argument("--db", default="runs.db", help="Path to SQLite database")
    parser.add_argument("--output", default="data/rlhf_training_v1", help="Output directory")
    parser.add_argument(
        "--max-rows", type=int, default=50, help="Max CSV rows to include in prompt"
    )
    parser.add_argument("--model", default="gpt_4_1", help="Model to filter by")
    parser.add_argument(
        "--num-rejected", 
        type=int, 
        default=3, 
        help="Number of rejected variants per example"
    )

    args = parser.parse_args()

    export_rlhf_training_examples(
        db_path=args.db,
        output_dir=args.output,
        max_csv_rows=args.max_rows,
        model_filter=args.model,
        num_rejected_per_example=args.num_rejected,
    )


if __name__ == "__main__":
    main()
