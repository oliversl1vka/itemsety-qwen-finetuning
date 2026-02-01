"""
Training Data Generator V2 - From Real Datasets to 500 Augmented Examples

This script takes 50 real-world frequent itemset mining datasets and generates
10 variations of each, resulting in 500 training examples with:
- Realistic item names (not synthetic)
- Preserved co-occurrence patterns
- Chain-of-thought reasoning
- Apriori ground truth

Usage:
    python generate_training_v2.py --input-dir real_datasets/ --output-dir training_v2/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, UTC
from collections import Counter
from itertools import combinations

import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Augmentation settings
    "variations_per_dataset": 10,
    "min_support_options": [2, 3, 4, 5],  # Different thresholds to train on
    
    # Subsampling
    "row_sample_range": (0.6, 0.95),  # Keep 60-95% of rows
    "col_sample_range": (0.7, 1.0),   # Keep 70-100% of columns
    
    # Noise injection
    "noise_row_prob": 0.3,  # 30% chance to add noise rows
    "max_noise_rows": 3,
    
    # Size limits for training
    "min_rows": 5,
    "max_rows": 30,
    "min_items_per_row": 2,
    "max_items_per_row": 20,
    
    # Output
    "train_split": 0.9,
    "random_seed": 42,
}


# =============================================================================
# DATASET ANALYSIS
# =============================================================================

def detect_dataset_format(df: pd.DataFrame) -> str:
    """
    Detect the format of a dataset:
    - 'transaction_list': Two columns (transaction_id, item)
    - 'basket_per_row': Multiple item columns per row
    - 'one_hot': Binary presence columns
    - 'single_column': Items in one column (comma/space separated)
    """
    if len(df.columns) == 2:
        # Could be transaction list format
        col1, col2 = df.columns
        if df[col1].dtype == 'object' or df[col1].dtype == 'int64':
            # Check if first column looks like IDs
            if len(df[col1].unique()) < len(df):
                return 'transaction_list'
    
    # Check for one-hot encoding
    binary_cols = sum(1 for c in df.columns if set(df[c].dropna().unique()).issubset({0, 1, '0', '1', True, False}))
    if binary_cols > len(df.columns) * 0.7:
        return 'one_hot'
    
    # Check if single column with delimited items
    if len(df.columns) == 1:
        return 'single_column'
    
    # Default: basket per row (most common)
    return 'basket_per_row'


def extract_transactions(df: pd.DataFrame, format_type: str) -> List[List[str]]:
    """
    Extract transactions (list of lists of items) from any format.
    """
    transactions = []
    
    if format_type == 'transaction_list':
        # Group items by transaction ID
        id_col, item_col = df.columns[:2]
        grouped = df.groupby(id_col)[item_col].apply(list).tolist()
        transactions = [[str(item).strip().lower() for item in items] for items in grouped]
    
    elif format_type == 'one_hot':
        # Each row is a transaction, column names are items where value is truthy
        for _, row in df.iterrows():
            items = [str(col).strip().lower() for col in df.columns if row[col] in [1, '1', True, 'True', 'true']]
            if items:
                transactions.append(items)
    
    elif format_type == 'single_column':
        # Split by common delimiters
        col = df.columns[0]
        for val in df[col].dropna():
            items = re.split(r'[,;\s]+', str(val))
            items = [item.strip().lower() for item in items if item.strip()]
            if items:
                transactions.append(items)
    
    else:  # basket_per_row
        # Each cell value is an item
        for _, row in df.iterrows():
            items = []
            for val in row:
                if pd.notna(val) and str(val).strip():
                    # Skip numeric-only values (likely not items)
                    val_str = str(val).strip()
                    if not re.match(r'^-?\d+\.?\d*$', val_str):
                        items.append(val_str.lower())
            if items:
                transactions.append(items)
    
    return transactions


def extract_vocabulary(transactions: List[List[str]]) -> Dict[str, int]:
    """Extract item vocabulary with frequencies."""
    freq = Counter()
    for trans in transactions:
        freq.update(trans)
    return dict(freq)


# =============================================================================
# APRIORI IMPLEMENTATION (Simple, for ground truth)
# =============================================================================

def apriori_frequent_itemsets(
    transactions: List[List[str]],
    min_support: int = 3,
    max_size: int = 4
) -> List[Dict[str, Any]]:
    """
    Simple Apriori implementation for ground truth generation.
    Returns itemsets with count >= min_support.
    """
    results = []
    n_trans = len(transactions)
    
    if n_trans == 0:
        return []
    
    # Convert transactions to sets for faster lookup
    trans_sets = [set(t) for t in transactions]
    
    # Get all unique items
    all_items = set()
    for t in transactions:
        all_items.update(t)
    
    # Level 1: Singletons
    candidates = [{item} for item in all_items]
    
    for size in range(1, max_size + 1):
        frequent_at_level = []
        
        for candidate in candidates:
            # Count support
            count = 0
            evidence_rows = []
            for i, ts in enumerate(trans_sets):
                if candidate.issubset(ts):
                    count += 1
                    evidence_rows.append(i + 1)  # 1-based indexing
            
            if count >= min_support:
                itemset = sorted(list(candidate))
                frequent_at_level.append({
                    "itemset": itemset,
                    "count": count,
                    "evidence_rows": evidence_rows,
                    "support": round(count / n_trans, 4)
                })
        
        results.extend(frequent_at_level)
        
        if size >= max_size:
            break
        
        # Generate next level candidates (Apriori pruning)
        frequent_items = [set(f["itemset"]) for f in frequent_at_level]
        candidates = []
        for i, s1 in enumerate(frequent_items):
            for s2 in frequent_items[i+1:]:
                union = s1 | s2
                if len(union) == size + 1:
                    # Check all subsets are frequent
                    all_subsets_frequent = True
                    for item in union:
                        subset = union - {item}
                        if subset not in frequent_items:
                            all_subsets_frequent = False
                            break
                    if all_subsets_frequent and union not in candidates:
                        candidates.append(union)
    
    # Sort by size, then by count (descending)
    results.sort(key=lambda x: (len(x["itemset"]), -x["count"], x["itemset"]))
    
    return results


# =============================================================================
# CHAIN-OF-THOUGHT GENERATION
# =============================================================================

def generate_chain_of_thought(
    csv_content: str,
    transactions: List[List[str]],
    itemsets: List[Dict[str, Any]],
    min_support: int
) -> str:
    """
    Generate chain-of-thought reasoning for the itemset extraction.
    """
    cot_parts = []
    
    # Step 1: Parse CSV
    num_rows = len(transactions)
    all_items = set()
    for t in transactions:
        all_items.update(t)
    
    cot_parts.append(f"## Step 1: Parse CSV")
    cot_parts.append(f"Total rows/transactions: {num_rows}")
    cot_parts.append(f"Unique items found: {len(all_items)}")
    
    # Step 2: Show transactions (first 5)
    cot_parts.append(f"\n## Step 2: Extract items per row")
    for i, trans in enumerate(transactions[:5], 1):
        cot_parts.append(f"Row {i}: {trans}")
    if len(transactions) > 5:
        cot_parts.append(f"... ({num_rows - 5} more rows)")
    
    # Step 3: Count occurrences (top items)
    freq = Counter()
    for t in transactions:
        freq.update(t)
    
    cot_parts.append(f"\n## Step 3: Count item occurrences (top 10)")
    for item, count in freq.most_common(10):
        rows = [i+1 for i, t in enumerate(transactions) if item in t]
        rows_str = str(rows[:5]) + ("..." if len(rows) > 5 else "")
        status = "✓" if count >= min_support else "✗"
        cot_parts.append(f"- '{item}': {count} rows {rows_str} {status}")
    
    # Step 4: Itemsets found
    cot_parts.append(f"\n## Step 4: Find frequent itemsets (support ≥ {min_support})")
    
    # Group by size
    by_size = {}
    for iset in itemsets:
        size = len(iset["itemset"])
        by_size.setdefault(size, []).append(iset)
    
    for size in sorted(by_size.keys()):
        cot_parts.append(f"\n### {size}-itemsets:")
        for iset in by_size[size][:5]:  # Show top 5 per size
            items_str = ", ".join(iset["itemset"])
            evidence_str = str(iset["evidence_rows"][:5])
            if len(iset["evidence_rows"]) > 5:
                evidence_str += "..."
            cot_parts.append(f"- {{{items_str}}}: count={iset['count']}, rows={evidence_str}")
        if len(by_size[size]) > 5:
            cot_parts.append(f"  ... ({len(by_size[size]) - 5} more)")
    
    # Step 5: Summary
    cot_parts.append(f"\n## Step 5: Summary")
    cot_parts.append(f"Total itemsets with support ≥ {min_support}: {len(itemsets)}")
    size_counts = {s: len(items) for s, items in by_size.items()}
    cot_parts.append(f"By size: {size_counts}")
    
    return "\n".join(cot_parts)


def format_output_json(itemsets: List[Dict[str, Any]], limit: int = 20) -> str:
    """Format itemsets as JSON output, limiting to top N for training."""
    # Take top itemsets (prioritize 2-3 itemsets over singletons)
    sorted_itemsets = sorted(
        itemsets,
        key=lambda x: (1 if len(x["itemset"]) in [2, 3] else 0, -x["count"]),
        reverse=True
    )[:limit]
    
    output = []
    for iset in sorted_itemsets:
        output.append({
            "itemset": iset["itemset"],
            "count": iset["count"],
            "evidence_rows": iset["evidence_rows"][:10],  # Limit evidence for training
            "explanation": f"Co-occurs in {iset['count']} transactions"
        })
    
    return json.dumps(output, indent=2, ensure_ascii=False)


# =============================================================================
# AUGMENTATION FUNCTIONS
# =============================================================================

def augment_subsample_rows(transactions: List[List[str]], ratio: float) -> List[List[str]]:
    """Randomly sample a subset of rows."""
    n = max(CONFIG["min_rows"], int(len(transactions) * ratio))
    if n >= len(transactions):
        return transactions.copy()
    indices = random.sample(range(len(transactions)), n)
    return [transactions[i] for i in sorted(indices)]


def augment_subsample_items(transactions: List[List[str]], ratio: float) -> List[List[str]]:
    """Randomly remove some items from transactions."""
    result = []
    for trans in transactions:
        if len(trans) <= CONFIG["min_items_per_row"]:
            result.append(trans.copy())
        else:
            n = max(CONFIG["min_items_per_row"], int(len(trans) * ratio))
            result.append(random.sample(trans, n))
    return result


def augment_shuffle_rows(transactions: List[List[str]]) -> List[List[str]]:
    """Shuffle the order of transactions."""
    result = transactions.copy()
    random.shuffle(result)
    return result


def augment_add_noise_rows(transactions: List[List[str]], vocab: List[str]) -> List[List[str]]:
    """Add a few random noise rows."""
    result = transactions.copy()
    n_noise = random.randint(1, CONFIG["max_noise_rows"])
    for _ in range(n_noise):
        noise_size = random.randint(2, 5)
        noise_row = random.sample(vocab, min(noise_size, len(vocab)))
        pos = random.randint(0, len(result))
        result.insert(pos, noise_row)
    return result


def augment_item_synonyms(transactions: List[List[str]], synonym_map: Dict[str, str]) -> List[List[str]]:
    """Replace some items with synonyms."""
    result = []
    for trans in transactions:
        new_trans = []
        for item in trans:
            if item in synonym_map and random.random() < 0.3:
                new_trans.append(synonym_map[item])
            else:
                new_trans.append(item)
        result.append(new_trans)
    return result


def transactions_to_csv(transactions: List[List[str]]) -> str:
    """Convert transactions to CSV format."""
    if not transactions:
        return ""
    
    # Determine max items per row
    max_items = max(len(t) for t in transactions)
    
    # Create header
    header = ",".join([f"item_{i+1}" for i in range(max_items)])
    
    # Create rows
    rows = [header]
    for trans in transactions:
        row = trans + [""] * (max_items - len(trans))
        rows.append(",".join(row))
    
    return "\n".join(rows)


# =============================================================================
# MAIN TRAINING DATA GENERATION
# =============================================================================

def generate_variations(
    base_transactions: List[List[str]],
    vocabulary: List[str],
    num_variations: int = 10
) -> List[Tuple[List[List[str]], str]]:
    """
    Generate N variations of a base transaction set.
    Returns list of (transactions, augmentation_description) tuples.
    """
    variations = []
    
    # Original (shuffled only)
    variations.append((augment_shuffle_rows(base_transactions), "original_shuffled"))
    
    # Row subsamples
    for i in range(2):
        ratio = random.uniform(*CONFIG["row_sample_range"])
        aug = augment_subsample_rows(base_transactions, ratio)
        aug = augment_shuffle_rows(aug)
        variations.append((aug, f"row_subsample_{ratio:.2f}"))
    
    # Item subsamples
    for i in range(2):
        ratio = random.uniform(*CONFIG["col_sample_range"])
        aug = augment_subsample_items(base_transactions, ratio)
        aug = augment_shuffle_rows(aug)
        variations.append((aug, f"item_subsample_{ratio:.2f}"))
    
    # Combined subsampling
    for i in range(2):
        row_ratio = random.uniform(*CONFIG["row_sample_range"])
        col_ratio = random.uniform(*CONFIG["col_sample_range"])
        aug = augment_subsample_rows(base_transactions, row_ratio)
        aug = augment_subsample_items(aug, col_ratio)
        aug = augment_shuffle_rows(aug)
        variations.append((aug, f"combined_{row_ratio:.2f}_{col_ratio:.2f}"))
    
    # With noise
    if len(vocabulary) >= 5:
        for i in range(2):
            aug = augment_add_noise_rows(base_transactions.copy(), vocabulary)
            aug = augment_shuffle_rows(aug)
            variations.append((aug, f"with_noise_{i}"))
    else:
        # Fallback: more subsampling
        for i in range(2):
            ratio = random.uniform(0.7, 0.9)
            aug = augment_subsample_rows(base_transactions, ratio)
            variations.append((aug, f"extra_subsample_{ratio:.2f}"))
    
    # Trim to exact number
    return variations[:num_variations]


def process_single_dataset(
    file_path: Path,
    output_dir: Path,
    dataset_idx: int
) -> List[Dict[str, Any]]:
    """
    Process a single real dataset and generate training examples.
    """
    examples = []
    
    try:
        # Load dataset
        df = pd.read_csv(file_path, low_memory=False)
        
        # Detect format and extract transactions
        format_type = detect_dataset_format(df)
        transactions = extract_transactions(df, format_type)
        
        if len(transactions) < CONFIG["min_rows"]:
            print(f"  ⚠️  Skipping {file_path.name}: too few transactions ({len(transactions)})")
            return []
        
        # Limit to max_rows for training
        if len(transactions) > CONFIG["max_rows"]:
            transactions = random.sample(transactions, CONFIG["max_rows"])
        
        # Extract vocabulary
        vocab = list(extract_vocabulary(transactions).keys())
        
        print(f"  📊 Format: {format_type}, Transactions: {len(transactions)}, Items: {len(vocab)}")
        
        # Generate variations
        variations = generate_variations(
            transactions, vocab, CONFIG["variations_per_dataset"]
        )
        
        for var_idx, (var_transactions, var_desc) in enumerate(variations):
            # Pick random min_support
            min_support = random.choice(CONFIG["min_support_options"])
            
            # Convert to CSV
            csv_content = transactions_to_csv(var_transactions)
            
            # Run Apriori for ground truth
            itemsets = apriori_frequent_itemsets(var_transactions, min_support=min_support)
            
            # Generate chain-of-thought
            cot = generate_chain_of_thought(csv_content, var_transactions, itemsets, min_support)
            
            # Format output
            json_output = format_output_json(itemsets)
            
            # Create training example
            example = {
                "id": f"ds_{dataset_idx:04d}_var_{var_idx:02d}",
                "source_file": file_path.name,
                "variation": var_desc,
                "min_support": min_support,
                "num_transactions": len(var_transactions),
                "num_items": len(vocab),
                "num_itemsets": len(itemsets),
                "input": csv_content,
                "chain_of_thought": cot,
                "output": json_output,
                # Full example for training (input -> CoT + output)
                "full_output": f"{cot}\n\n## Final Output:\n{json_output}"
            }
            
            examples.append(example)
        
    except Exception as e:
        print(f"  ❌ Error processing {file_path.name}: {e}")
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate training data V2 from real datasets")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with real CSV datasets")
    parser.add_argument("--output-dir", type=str, default="training_v2", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--variations", type=int, default=10, help="Variations per dataset")
    args = parser.parse_args()
    
    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    CONFIG["variations_per_dataset"] = args.variations
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TRAINING DATA GENERATOR V2")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Variations per dataset: {CONFIG['variations_per_dataset']}")
    print()
    
    # Find all CSV files
    csv_files = sorted(input_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    print()
    
    if not csv_files:
        print("❌ No CSV files found!")
        return
    
    # Process each dataset
    all_examples = []
    
    for idx, csv_file in enumerate(csv_files, 1):
        print(f"[{idx}/{len(csv_files)}] Processing: {csv_file.name}")
        examples = process_single_dataset(csv_file, output_dir, idx)
        all_examples.extend(examples)
        print(f"  ✅ Generated {len(examples)} examples")
        print()
    
    # Split into train/val
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * CONFIG["train_split"])
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    # Save outputs
    print("=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)
    
    # Save full examples
    with open(output_dir / "all_examples.json", "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved all examples: {output_dir / 'all_examples.json'}")
    
    # Save train/val splits
    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_examples, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved train set: {len(train_examples)} examples")
    
    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_examples, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved val set: {len(val_examples)} examples")
    
    # Save HuggingFace format (for direct training)
    hf_train = []
    hf_val = []
    
    for ex in train_examples:
        hf_train.append({
            "messages": [
                {"role": "system", "content": "You are a frequent itemset extractor. Parse the CSV, count item occurrences, and find itemsets with support >= threshold. Output valid JSON."},
                {"role": "user", "content": ex["input"]},
                {"role": "assistant", "content": ex["full_output"]}
            ]
        })
    
    for ex in val_examples:
        hf_val.append({
            "messages": [
                {"role": "system", "content": "You are a frequent itemset extractor. Parse the CSV, count item occurrences, and find itemsets with support >= threshold. Output valid JSON."},
                {"role": "user", "content": ex["input"]},
                {"role": "assistant", "content": ex["full_output"]}
            ]
        })
    
    with open(output_dir / "train_hf.jsonl", "w", encoding="utf-8") as f:
        for ex in hf_train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"✅ Saved HF train: {output_dir / 'train_hf.jsonl'}")
    
    with open(output_dir / "val_hf.jsonl", "w", encoding="utf-8") as f:
        for ex in hf_val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"✅ Saved HF val: {output_dir / 'val_hf.jsonl'}")
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"📊 Total datasets processed: {len(csv_files)}")
    print(f"📊 Total examples generated: {len(all_examples)}")
    print(f"📊 Train examples: {len(train_examples)}")
    print(f"📊 Val examples: {len(val_examples)}")
    print()
    
    # Stats
    avg_transactions = sum(ex["num_transactions"] for ex in all_examples) / len(all_examples)
    avg_itemsets = sum(ex["num_itemsets"] for ex in all_examples) / len(all_examples)
    empty_count = sum(1 for ex in all_examples if ex["num_itemsets"] == 0)
    
    print(f"📈 Avg transactions per example: {avg_transactions:.1f}")
    print(f"📈 Avg itemsets per example: {avg_itemsets:.1f}")
    print(f"📈 Empty outputs (no itemsets): {empty_count} ({100*empty_count/len(all_examples):.1f}%)")
    print()
    print("✅ DONE! Ready for fine-tuning.")


if __name__ == "__main__":
    main()
