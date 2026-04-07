#!/usr/bin/env python3
"""
Generate evaluation datasets for the fine-tuned itemset extraction model.

These datasets are:
- Generated from the SAME real-world sources as training (same format: col:value)
- But with a DIFFERENT random seed → completely different row/column subsamples
- NEVER overlap with training data (verified via hash)
- Pushed to HuggingFace as a separate dataset for fair evaluation

Output: data/eval_datasets_v2/ with 30 CSV files + metadata JSON
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════
EVAL_SEED = 99999  # Different from training seed (42)
N_EVAL_DATASETS = 30
OUTPUT_DIR = Path("data/eval_datasets_v2")
REAL_DATASETS_DIR = Path("real_datasets")

# Size ranges (same as training to match what model expects)
SIZE_DISTRIBUTION = {
    "small":  {"rows": (5, 7),   "cols": (4, 8),   "weight": 0.30},
    "medium": {"rows": (8, 10),  "cols": (6, 10),  "weight": 0.40},
    "large":  {"rows": (11, 15), "cols": (8, 12),  "weight": 0.30},
}

MIN_SUPPORT = 3
MAX_SIZE = 3


# ═════════════════════════════════════════════════════════════════════════════
# Apriori (for ground truth)
# ═════════════════════════════════════════════════════════════════════════════
def apriori(transactions: list[list[str]], min_support: int = 3, max_size: int = 3) -> list[dict]:
    if not transactions:
        return []

    row_labels = [f"Row {i+1}" for i in range(len(transactions))]
    counts: dict[tuple, dict] = {}

    for idx, trans in enumerate(transactions):
        seen: set[str] = set()
        for item in trans:
            item_n = str(item).strip().lower()
            if not item_n or item_n in seen:
                continue
            seen.add(item_n)
            k = (item_n,)
            if k not in counts:
                counts[k] = {"count": 0, "rows": []}
            counts[k]["count"] += 1
            counts[k]["rows"].append(row_labels[idx])

    def prune(d):
        return {k: v for k, v in d.items() if v["count"] >= min_support}

    L1 = prune(counts)
    if not L1:
        return []

    freq_levels = [L1]
    current = L1
    k = 2
    while k <= max_size and current:
        prev_keys = sorted(current.keys())
        candidates: set[tuple] = set()
        for i in range(len(prev_keys)):
            for j in range(i + 1, len(prev_keys)):
                a, b = prev_keys[i], prev_keys[j]
                if a[:k-2] == b[:k-2]:
                    merged = tuple(sorted(set(a) | set(b)))
                    if len(merged) == k:
                        if all(tuple(sorted(sub)) in current
                               for sub in itertools.combinations(merged, k-1)):
                            candidates.add(merged)
        if not candidates:
            break
        cand_counts = {c: {"count": 0, "rows": []} for c in candidates}
        for idx, trans in enumerate(transactions):
            tset = {str(x).strip().lower() for x in trans}
            for cand in candidates:
                if set(cand).issubset(tset):
                    cand_counts[cand]["count"] += 1
                    cand_counts[cand]["rows"].append(row_labels[idx])
        current = prune(cand_counts)
        if current:
            freq_levels.append(current)
        k += 1

    out = []
    for level in freq_levels:
        for itemset, info in level.items():
            out.append({"itemset": list(itemset), "count": info["count"], "rows": info["rows"]})
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Source dataset loading
# ═════════════════════════════════════════════════════════════════════════════
def load_source(path: Path) -> dict | None:
    """Load a real-world CSV and identify categorical columns."""
    try:
        df = pd.read_csv(path, low_memory=False)
        cat_cols = []
        for col in df.columns:
            if df[col].dtype == "object":
                if 2 <= df[col].nunique() <= 50:
                    cat_cols.append(col)
            elif df[col].dtype in ["int64", "float64"]:
                if 2 <= df[col].nunique() <= 10:
                    cat_cols.append(col)

        if len(cat_cols) < 4:
            return None

        return {"path": path, "df": df, "cat_cols": cat_cols}
    except Exception as e:
        print(f"  ⚠️  Skip {path.name}: {e}")
        return None


def extract_transactions(df: pd.DataFrame, cat_cols: list[str]) -> list[list[str]]:
    """Extract col:value items from selected categorical columns."""
    transactions = []
    for row_dict in df[cat_cols].to_dict("records"):
        items = []
        for col in cat_cols:
            val = row_dict.get(col)
            if pd.notna(val):
                item = str(val).strip().lower()
                if item and item != "nan":
                    items.append(f"{col.lower()}:{item}")
        if items:
            transactions.append(items)
    return transactions


# ═════════════════════════════════════════════════════════════════════════════
# Dataset generation
# ═════════════════════════════════════════════════════════════════════════════
def generate_one_dataset(
    source: dict, target_rows: int, target_cols: int
) -> tuple[list[list[str]], list[str]] | None:
    """Generate one eval dataset from a source."""
    # Pick random subset of categorical columns
    cols = source["cat_cols"]
    n_cols = min(target_cols, len(cols))
    selected_cols = random.sample(cols, n_cols)

    # Extract transactions from full source
    all_trans = extract_transactions(source["df"], selected_cols)
    if len(all_trans) < target_rows:
        return None

    # Pick random rows
    indices = random.sample(range(len(all_trans)), target_rows)
    transactions = [all_trans[i] for i in sorted(indices)]

    return transactions, selected_cols


def transactions_to_csv_file(transactions: list[list[str]], path: Path) -> None:
    """Write transactions as a proper CSV file."""
    max_items = max(len(t) for t in transactions)
    header = [f"item_{i+1}" for i in range(max_items)]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for trans in transactions:
            padded = trans + [""] * (max_items - len(trans))
            writer.writerow(padded)


def format_csv_text(transactions: list[list[str]]) -> str:
    """Format as 'Row N: item1, item2, ...' (matches training format)."""
    lines = []
    for idx, trans in enumerate(transactions):
        lines.append(f"Row {idx + 1}: {', '.join(trans)}")
    return "\n".join(lines)


def dataset_hash(csv_text: str) -> str:
    return hashlib.sha256(csv_text.encode()).hexdigest()[:12]


# ═════════════════════════════════════════════════════════════════════════════
# Get training dataset hashes to ensure no overlap
# ═════════════════════════════════════════════════════════════════════════════
def get_training_hashes() -> set[str]:
    """Collect all dataset_id hashes from training data to avoid overlap."""
    hashes = set()
    for json_path in [Path("data/sft_cot_v2.json"), Path("data/dpo_real_v2.json")]:
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        for entry in data:
            ds_id = entry.get("dataset_id", "")
            # Extract hash part: "ds_0001_7x12_85aed5f8.csv:85aed5f80c90" → "85aed5f80c90"
            if ":" in ds_id:
                hashes.add(ds_id.split(":")[-1])
    return hashes


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)

    print("═" * 60)
    print("  EVALUATION DATASET GENERATOR")
    print("═" * 60)

    # Load real-world source datasets
    if not REAL_DATASETS_DIR.is_dir():
        print(f"❌ Source directory not found: {REAL_DATASETS_DIR}")
        sys.exit(1)

    sources = []
    for csv_path in sorted(REAL_DATASETS_DIR.glob("*.csv")):
        src = load_source(csv_path)
        if src:
            sources.append(src)
            print(f"  ✅ {csv_path.name}: {len(src['cat_cols'])} cat cols, {len(src['df'])} rows")

    if not sources:
        print("❌ No usable source datasets")
        sys.exit(1)

    print(f"\n📚 Loaded {len(sources)} source datasets")

    # Get training hashes to avoid overlap
    training_hashes = get_training_hashes()
    print(f"🔒 {len(training_hashes)} training dataset hashes loaded for exclusion")

    # Generate eval datasets
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    attempts = 0
    max_attempts = N_EVAL_DATASETS * 10

    # Build size buckets
    size_weights = [v["weight"] for v in SIZE_DISTRIBUTION.values()]
    size_keys = list(SIZE_DISTRIBUTION.keys())

    while len(generated) < N_EVAL_DATASETS and attempts < max_attempts:
        attempts += 1

        # Pick random source
        source = random.choice(sources)

        # Pick size
        size_key = random.choices(size_keys, weights=size_weights, k=1)[0]
        cfg = SIZE_DISTRIBUTION[size_key]
        target_rows = random.randint(*cfg["rows"])
        target_cols = random.randint(*cfg["cols"])

        # Generate
        result = generate_one_dataset(source, target_rows, target_cols)
        if result is None:
            continue

        transactions, cols_used = result

        # Check it has enough frequent itemsets (at least 3)
        ground_truth = apriori(transactions, MIN_SUPPORT, MAX_SIZE)
        if len(ground_truth) < 3:
            continue

        # Check hash doesn't overlap with training
        csv_text = format_csv_text(transactions)
        h = dataset_hash(csv_text)
        if h in training_hashes:
            continue

        # Also check we haven't already generated this exact dataset
        if any(g["hash"] == h for g in generated):
            continue

        # Save
        idx = len(generated) + 1
        n_rows = len(transactions)
        n_cols = max(len(t) for t in transactions)
        filename = f"eval_{idx:03d}_{n_rows}x{n_cols}_{h}.csv"
        filepath = OUTPUT_DIR / filename

        transactions_to_csv_file(transactions, filepath)

        meta = {
            "id": idx,
            "filename": filename,
            "hash": h,
            "source": source["path"].name,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "size_category": size_key,
            "n_ground_truth_itemsets": len(ground_truth),
            "min_support": MIN_SUPPORT,
            "max_size": MAX_SIZE,
            "csv_text": csv_text,
            "ground_truth": ground_truth,
        }
        generated.append(meta)
        print(
            f"  [{idx:2d}/{N_EVAL_DATASETS}] {filename}  "
            f"({size_key}, {n_rows}r×{n_cols}c, "
            f"{len(ground_truth)} itemsets, src={source['path'].name})"
        )

    # Save metadata
    metadata_path = OUTPUT_DIR / "eval_metadata.json"
    # Save a lightweight version (without full csv_text and ground_truth for the JSON)
    metadata_light = []
    for g in generated:
        metadata_light.append({
            k: v for k, v in g.items() if k not in ("csv_text", "ground_truth")
        })

    metadata_path.write_text(
        json.dumps(metadata_light, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save full metadata (with ground truth — for the HF dataset)
    full_path = OUTPUT_DIR / "eval_full_metadata.json"
    full_path.write_text(
        json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  ✅ Generated {len(generated)} eval datasets")
    print(f"  📁 Output: {OUTPUT_DIR}/")
    print(f"  📄 Metadata: {metadata_path}")
    print(f"  🔒 Zero overlap with {len(training_hashes)} training hashes")
    print(f"  📊 Ground truth itemsets: {sum(g['n_ground_truth_itemsets'] for g in generated)} total")

    # Size distribution
    for sk in size_keys:
        count = sum(1 for g in generated if g["size_category"] == sk)
        print(f"     {sk}: {count} datasets")

    # Source distribution
    src_counts: dict[str, int] = {}
    for g in generated:
        src_counts[g["source"]] = src_counts.get(g["source"], 0) + 1
    print(f"\n  Source distribution:")
    for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"     {src}: {cnt}")

    print(f"\n  Next steps:")
    print(f"    python scripts/push_eval_datasets_to_hf.py")
    print(f"═" * 60)


if __name__ == "__main__":
    main()
