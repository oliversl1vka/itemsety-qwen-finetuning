#!/usr/bin/env python3
"""
Push evaluation datasets to HuggingFace Hub as a separate dataset.

Creates a HF dataset with:
  - csv_text: the formatted "Row N: item1, item2" text (model input)
  - ground_truth: Apriori itemsets as JSON string
  - metadata: n_rows, n_cols, n_itemsets, source, min_support

Usage:
  python scripts/push_eval_datasets_to_hf.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load HF token
for env_file in ["hf.env", ".env"]:
    if Path(env_file).exists():
        load_dotenv(env_file)
        break

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    print("❌ No HF_TOKEN found. Set it in hf.env or environment.")
    sys.exit(1)

HF_REPO = "OliverSlivka/itemset-eval-v2"
EVAL_DIR = Path("data/eval_datasets_v2")
METADATA_FILE = EVAL_DIR / "eval_full_metadata.json"


def main():
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    print("═" * 60)
    print("  PUSH EVAL DATASETS TO HUGGINGFACE")
    print("═" * 60)

    if not METADATA_FILE.exists():
        print(f"❌ Metadata not found: {METADATA_FILE}")
        print("   Run: python scripts/generate_eval_datasets.py first")
        sys.exit(1)

    with open(METADATA_FILE) as f:
        all_meta = json.load(f)

    print(f"📄 Loaded {len(all_meta)} eval datasets from metadata")

    # Build HF dataset rows
    rows = []
    for entry in all_meta:
        rows.append({
            "id": entry["id"],
            "filename": entry["filename"],
            "csv_text": entry["csv_text"],
            "ground_truth": json.dumps(entry["ground_truth"], ensure_ascii=False),
            "n_rows": entry["n_rows"],
            "n_cols": entry["n_cols"],
            "n_itemsets": entry["n_ground_truth_itemsets"],
            "source": entry["source"],
            "min_support": entry["min_support"],
            "max_size": entry["max_size"],
            "size_category": entry["size_category"],
            "hash": entry["hash"],
        })

    ds = Dataset.from_list(rows)
    print(f"\n📊 Dataset: {ds}")
    print(f"   Columns: {ds.column_names}")
    print(f"   Rows: {len(ds)}")

    # Push to Hub
    print(f"\n📤 Pushing to {HF_REPO}...")
    ds.push_to_hub(
        HF_REPO,
        token=HF_TOKEN,
        private=False,
    )

    print(f"\n✅ Pushed to https://huggingface.co/datasets/{HF_REPO}")
    print(f"   {len(rows)} eval datasets with ground truth")

    # Also upload the raw CSV files as supplementary
    api = HfApi(token=HF_TOKEN)
    csv_files = sorted(EVAL_DIR.glob("eval_*.csv"))
    if csv_files:
        print(f"\n📤 Uploading {len(csv_files)} CSV files...")
        api.upload_folder(
            folder_path=str(EVAL_DIR),
            repo_id=HF_REPO,
            repo_type="dataset",
            path_in_repo="csv_files",
            allow_patterns="eval_*.csv",
        )
        print(f"   ✅ CSVs uploaded to {HF_REPO}/csv_files/")

    print(f"\n{'═' * 60}")
    print(f"  ✅ DONE — Eval dataset available at:")
    print(f"     https://huggingface.co/datasets/{HF_REPO}")
    print(f"═" * 60)


if __name__ == "__main__":
    main()
