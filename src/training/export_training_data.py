#!/usr/bin/env python3
"""
Export validated runs from runs.db for HuggingFace training.
Creates training-ready JSON files with CSV context + ground truth.
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import argparse


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


def export_training_examples(
    db_path: str = "runs.db",
    output_dir: str = "training_data",
    max_csv_rows: int = 50,
    model_filter: str = "gpt_4_1",  # Use one model to avoid duplicates
) -> None:
    """Export all validated runs as training examples"""

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

    examples = []
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

        # Create training example
        example = {
            "id": dataset_id,
            "dataset_name": row["dataset_name"],
            "dataset_hash": row["dataset_hash"],
            "csv_context": csv_context,
            "min_support": row["min_support"],
            "ground_truth": ground_truth,
            "metadata": {
                "total_rows": row["dataset_size_rows"],
                "itemset_count": len(ground_truth),
                "data_path": row["data_path"],
                "apriori_path": row["apriori_output_path"],
            },
        }

        examples.append(example)

        # Save individual file
        example_file = output_path / f"{dataset_id}_training.json"
        with open(example_file, "w", encoding="utf-8") as f:
            json.dump(example, f, indent=2, ensure_ascii=False)

    conn.close()

    # Save combined file
    combined_file = output_path / "all_training_examples.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "total_examples": len(examples),
        "skipped": skipped,
        "model_filter": model_filter,
        "max_csv_rows": max_csv_rows,
        "output_directory": str(output_path),
    }

    summary_file = output_path / "export_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Export Complete!")
    print(f"   Examples: {len(examples)}")
    print(f"   Skipped: {skipped}")
    print(f"   Output: {output_path}")
    print(f"\nFiles created:")
    print(f"   - {len(examples)} individual JSON files")
    print(f"   - all_training_examples.json (combined)")
    print(f"   - export_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Export training data from runs.db")
    parser.add_argument("--db", default="runs.db", help="Path to SQLite database")
    parser.add_argument("--output", default="training_data", help="Output directory")
    parser.add_argument(
        "--max-rows", type=int, default=50, help="Max CSV rows to include"
    )
    parser.add_argument("--model", default="gpt_4_1", help="Model to filter by")

    args = parser.parse_args()

    export_training_examples(
        db_path=args.db,
        output_dir=args.output,
        max_csv_rows=args.max_rows,
        model_filter=args.model,
    )


if __name__ == "__main__":
    main()
