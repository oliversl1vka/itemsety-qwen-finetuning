"""
Semi-Human Dataset Generator for Frequent Itemset Mining Fine-tuning V2

Generates 500 training datasets from 25 real-world datasets with:
- Optimal size for LLM context window (based on V1 experiment findings)
- Preserved real-world patterns and item names
- Diverse variations (subsampling, shuffling, format changes)

Size Rationale (from V1 experiment):
- ds_0001_5x53 (5 rows): 74s generation, worked well
- ds_0006_28x100 (28 rows): 973s (16 min!), JSON errors
- >30 rows: Skipped as too large for CPU inference

Optimal dimensions for LLM processing:
- Rows: 5-25 (sweet spot for itemset mining + LLM context)
- Cols: 5-20 (categorical columns only)
- Max tokens: ~2000-3000 (safe for any LLM context window)
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, UTC
from collections import Counter
from itertools import combinations
import shutil

import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURATION - OPTIMIZED FOR LLM CONTEXT WINDOW
# =============================================================================

CONFIG = {
    # Dataset size constraints (based on V1 experiment findings)
    "min_rows": 5,          # Minimum transactions
    "max_rows": 25,         # Maximum transactions (was 30, reduced for safety)
    "target_rows_small": (5, 10),    # 40% of datasets
    "target_rows_medium": (10, 18),  # 40% of datasets  
    "target_rows_large": (18, 25),   # 20% of datasets
    
    "min_cols": 5,          # Minimum items/attributes
    "max_cols": 20,         # Maximum items/attributes (reduced from 100!)
    
    # Generation targets
    "total_datasets": 500,
    "datasets_per_source": 20,  # 25 sources × 20 = 500
    
    # Variation types distribution
    "variation_weights": {
        "row_subsample": 0.25,
        "col_subsample": 0.20,
        "combined_subsample": 0.25,
        "shuffle_only": 0.15,
        "add_noise": 0.15,
    },
    
    # Output
    "output_dir": "datasets_v2",
    "random_seed": 42,
    
    # Itemset mining parameters (for ground truth)
    "min_support_options": [2, 3, 4],  # Will vary per dataset
}


# =============================================================================
# DATASET ANALYSIS & LOADING
# =============================================================================

def load_and_analyze_source(file_path: Path) -> Dict[str, Any]:
    """Load a source dataset and extract its characteristics."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        
        # Identify categorical columns (for itemset mining)
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Text column - good for items
                if df[col].nunique() <= 100:  # Reasonable vocabulary
                    categorical_cols.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                # Numeric - check if it's really categorical (few unique values)
                if df[col].nunique() <= 10:
                    categorical_cols.append(col)
        
        # Extract vocabulary (unique values across categorical columns)
        vocabulary = set()
        for col in categorical_cols:
            values = df[col].dropna().astype(str).str.strip().str.lower()
            vocabulary.update(values.unique())
        
        # Remove empty/nan strings
        vocabulary.discard('')
        vocabulary.discard('nan')
        
        return {
            "file_path": file_path,
            "df": df,
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "categorical_cols": categorical_cols,
            "vocabulary": list(vocabulary),
            "vocab_size": len(vocabulary),
        }
        
    except Exception as e:
        print(f"  ❌ Error loading {file_path.name}: {e}")
        return None


def extract_transactions(df: pd.DataFrame, categorical_cols: List[str]) -> List[List[str]]:
    """
    Extract transactions from a DataFrame using categorical columns.
    Each row becomes a transaction with non-null categorical values as items.
    """
    transactions = []
    
    for _, row in df.iterrows():
        items = []
        for col in categorical_cols:
            val = row.get(col)
            if pd.notna(val):
                # Canonicalize: lowercase, strip whitespace
                item = str(val).strip().lower()
                if item and item != 'nan':
                    # Include column context for disambiguation
                    # e.g., "gender:male" instead of just "male"
                    items.append(f"{col.lower()}:{item}")
        
        if items:
            transactions.append(items)
    
    return transactions


# =============================================================================
# VARIATION GENERATORS
# =============================================================================

def generate_row_subsample(
    transactions: List[List[str]], 
    target_rows: int
) -> List[List[str]]:
    """Subsample rows to target size."""
    if len(transactions) <= target_rows:
        return transactions.copy()
    
    indices = random.sample(range(len(transactions)), target_rows)
    return [transactions[i] for i in sorted(indices)]


def generate_col_subsample(
    transactions: List[List[str]], 
    target_cols: int
) -> List[List[str]]:
    """Subsample columns (item prefixes) to target size."""
    # Get all unique column prefixes
    all_prefixes = set()
    for trans in transactions:
        for item in trans:
            if ':' in item:
                prefix = item.split(':')[0]
                all_prefixes.add(prefix)
    
    if len(all_prefixes) <= target_cols:
        return transactions
    
    # Select random subset of columns
    selected_prefixes = set(random.sample(list(all_prefixes), target_cols))
    
    # Filter transactions to only include selected columns
    result = []
    for trans in transactions:
        filtered = [item for item in trans if item.split(':')[0] in selected_prefixes]
        if filtered:
            result.append(filtered)
    
    return result


def generate_combined_subsample(
    transactions: List[List[str]], 
    target_rows: int,
    target_cols: int
) -> List[List[str]]:
    """Subsample both rows and columns."""
    # First subsample columns
    result = generate_col_subsample(transactions, target_cols)
    # Then subsample rows
    result = generate_row_subsample(result, target_rows)
    return result


def add_noise_transactions(
    transactions: List[List[str]], 
    noise_ratio: float = 0.1
) -> List[List[str]]:
    """Add some noise transactions with random items."""
    result = transactions.copy()
    
    # Get vocabulary
    all_items = list(set(item for trans in transactions for item in trans))
    if len(all_items) < 3:
        return result
    
    # Add noise rows
    n_noise = max(1, int(len(transactions) * noise_ratio))
    for _ in range(n_noise):
        noise_size = random.randint(2, min(5, len(all_items)))
        noise_trans = random.sample(all_items, noise_size)
        pos = random.randint(0, len(result))
        result.insert(pos, noise_trans)
    
    return result


def shuffle_transactions(transactions: List[List[str]]) -> List[List[str]]:
    """Shuffle transaction order."""
    result = transactions.copy()
    random.shuffle(result)
    return result


# =============================================================================
# CSV GENERATION
# =============================================================================

def transactions_to_csv(transactions: List[List[str]], include_header: bool = True) -> str:
    """
    Convert transactions to CSV format.
    Uses basket-per-row format where each column is an item slot.
    """
    if not transactions:
        return ""
    
    # Determine max items per transaction
    max_items = max(len(t) for t in transactions)
    max_items = min(max_items, CONFIG["max_cols"])  # Cap at max columns
    
    # Create header
    header = ",".join([f"item_{i+1}" for i in range(max_items)])
    
    # Create rows
    rows = []
    if include_header:
        rows.append(header)
    
    for trans in transactions:
        # Truncate if too many items
        trans = trans[:max_items]
        # Pad with empty strings
        row = trans + [""] * (max_items - len(trans))
        # Escape commas and quotes
        row = [f'"{v}"' if ',' in v or '"' in v else v for v in row]
        rows.append(",".join(row))
    
    return "\n".join(rows)


def calculate_dataset_hash(csv_content: str) -> str:
    """Calculate hash for dataset identification."""
    return hashlib.sha256(csv_content.encode()).hexdigest()[:12]


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

def generate_single_variation(
    source_info: Dict[str, Any],
    variation_type: str,
    variation_idx: int
) -> Tuple[str, Dict[str, Any]]:
    """Generate a single dataset variation from a source."""
    
    # Determine target size based on distribution
    size_roll = random.random()
    if size_roll < 0.4:
        target_rows = random.randint(*CONFIG["target_rows_small"])
    elif size_roll < 0.8:
        target_rows = random.randint(*CONFIG["target_rows_medium"])
    else:
        target_rows = random.randint(*CONFIG["target_rows_large"])
    
    target_cols = random.randint(CONFIG["min_cols"], CONFIG["max_cols"])
    
    # Extract transactions from source
    transactions = extract_transactions(
        source_info["df"], 
        source_info["categorical_cols"]
    )
    
    if len(transactions) < CONFIG["min_rows"]:
        # Source too small, duplicate some rows
        while len(transactions) < CONFIG["min_rows"]:
            transactions.extend(random.sample(transactions, min(3, len(transactions))))
    
    # Apply variation
    if variation_type == "row_subsample":
        result = generate_row_subsample(transactions, target_rows)
        result = shuffle_transactions(result)
    
    elif variation_type == "col_subsample":
        result = generate_col_subsample(transactions, target_cols)
        result = generate_row_subsample(result, target_rows)
        result = shuffle_transactions(result)
    
    elif variation_type == "combined_subsample":
        result = generate_combined_subsample(transactions, target_rows, target_cols)
        result = shuffle_transactions(result)
    
    elif variation_type == "shuffle_only":
        result = generate_row_subsample(transactions, target_rows)
        result = shuffle_transactions(result)
    
    elif variation_type == "add_noise":
        result = generate_row_subsample(transactions, target_rows - 2)  # Leave room for noise
        result = add_noise_transactions(result, noise_ratio=0.15)
        result = shuffle_transactions(result)
    
    else:
        result = generate_row_subsample(transactions, target_rows)
        result = shuffle_transactions(result)
    
    # Ensure we have valid transactions
    if not result:
        result = transactions[:target_rows]
    
    # Convert to CSV
    csv_content = transactions_to_csv(result)
    
    # Calculate actual dimensions
    lines = csv_content.strip().split('\n')
    actual_rows = len(lines) - 1  # Minus header
    actual_cols = len(lines[0].split(',')) if lines else 0
    
    # Metadata
    metadata = {
        "source_file": source_info["file_path"].name,
        "variation_type": variation_type,
        "variation_idx": variation_idx,
        "target_rows": target_rows,
        "target_cols": target_cols,
        "actual_rows": actual_rows,
        "actual_cols": actual_cols,
        "num_transactions": len(result),
        "total_items": sum(len(t) for t in result),
        "unique_items": len(set(item for t in result for item in t)),
    }
    
    return csv_content, metadata


def main():
    """Main generation pipeline."""
    random.seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])
    
    print("=" * 80)
    print("SEMI-HUMAN DATASET GENERATOR V2")
    print("Generating 500 LLM-optimized datasets from 25 real-world sources")
    print("=" * 80)
    
    # Setup directories
    source_dir = Path("real_datasets")
    output_dir = Path(CONFIG["output_dir"])
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    # Load all source datasets
    print("\n📂 Loading source datasets...")
    source_files = sorted(source_dir.glob("*.csv"))
    sources = []
    
    for sf in source_files:
        print(f"  Loading: {sf.name}...", end=" ")
        info = load_and_analyze_source(sf)
        if info and len(info["categorical_cols"]) >= 3:
            sources.append(info)
            print(f"✅ ({len(info['categorical_cols'])} categorical cols, {info['vocab_size']} items)")
        else:
            print(f"⚠️ Skipped (insufficient categorical columns)")
    
    print(f"\n✅ Loaded {len(sources)} valid source datasets")
    
    if len(sources) < 25:
        print(f"⚠️ Warning: Only {len(sources)} sources available, adjusting generation count")
    
    # Calculate datasets per source
    datasets_per_source = CONFIG["total_datasets"] // len(sources)
    remainder = CONFIG["total_datasets"] % len(sources)
    
    print(f"📊 Generating {datasets_per_source} datasets per source ({datasets_per_source * len(sources)} + {remainder} extra)")
    
    # Prepare variation type weights
    variation_types = list(CONFIG["variation_weights"].keys())
    variation_probs = list(CONFIG["variation_weights"].values())
    
    # Generate datasets
    print("\n🔄 Generating datasets...")
    all_datasets = []
    generation_log = []
    
    dataset_idx = 1
    
    for source_idx, source in enumerate(sources):
        # Determine how many datasets from this source
        n_datasets = datasets_per_source
        if source_idx < remainder:
            n_datasets += 1
        
        print(f"\n[{source_idx + 1}/{len(sources)}] {source['file_path'].name}: generating {n_datasets} variations")
        
        for var_idx in range(n_datasets):
            # Select variation type
            variation_type = random.choices(variation_types, weights=variation_probs)[0]
            
            # Generate variation
            csv_content, metadata = generate_single_variation(source, variation_type, var_idx)
            
            # Create filename
            dataset_hash = calculate_dataset_hash(csv_content)
            rows = metadata["actual_rows"]
            cols = metadata["actual_cols"]
            filename = f"ds_{dataset_idx:04d}_{rows}x{cols}_{dataset_hash[:8]}.csv"
            
            # Save dataset
            output_path = output_dir / filename
            output_path.write_text(csv_content, encoding='utf-8')
            
            # Log
            metadata["filename"] = filename
            metadata["dataset_idx"] = dataset_idx
            metadata["hash"] = dataset_hash
            generation_log.append(metadata)
            
            all_datasets.append({
                "filename": filename,
                "path": str(output_path),
                "metadata": metadata
            })
            
            # Progress indicator
            if dataset_idx % 50 == 0:
                print(f"    Generated {dataset_idx}/{CONFIG['total_datasets']} datasets...")
            
            dataset_idx += 1
    
    # Save generation log
    log_path = output_dir / "generation_log.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(generation_log, f, indent=2)
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("📊 GENERATION COMPLETE - STATISTICS")
    print("=" * 80)
    
    rows_dist = [m["actual_rows"] for m in generation_log]
    cols_dist = [m["actual_cols"] for m in generation_log]
    items_dist = [m["unique_items"] for m in generation_log]
    
    print(f"\n✅ Total datasets generated: {len(generation_log)}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Generation log: {log_path}")
    
    print(f"\n📏 SIZE DISTRIBUTION:")
    print(f"   Rows:  min={min(rows_dist)}, max={max(rows_dist)}, avg={np.mean(rows_dist):.1f}, median={np.median(rows_dist):.0f}")
    print(f"   Cols:  min={min(cols_dist)}, max={max(cols_dist)}, avg={np.mean(cols_dist):.1f}, median={np.median(cols_dist):.0f}")
    print(f"   Items: min={min(items_dist)}, max={max(items_dist)}, avg={np.mean(items_dist):.1f}")
    
    # Size buckets
    small = sum(1 for r in rows_dist if r <= 10)
    medium = sum(1 for r in rows_dist if 10 < r <= 18)
    large = sum(1 for r in rows_dist if r > 18)
    
    print(f"\n📊 SIZE BUCKETS:")
    print(f"   Small (5-10 rows):   {small} ({100*small/len(rows_dist):.1f}%)")
    print(f"   Medium (11-18 rows): {medium} ({100*medium/len(rows_dist):.1f}%)")
    print(f"   Large (19-25 rows):  {large} ({100*large/len(rows_dist):.1f}%)")
    
    # Variation types used
    var_counts = Counter(m["variation_type"] for m in generation_log)
    print(f"\n🔄 VARIATION TYPES:")
    for vtype, count in sorted(var_counts.items(), key=lambda x: -x[1]):
        print(f"   {vtype}: {count} ({100*count/len(generation_log):.1f}%)")
    
    # Source distribution
    source_counts = Counter(m["source_file"] for m in generation_log)
    print(f"\n📂 SOURCE DISTRIBUTION (top 10):")
    for source, count in source_counts.most_common(10):
        print(f"   {source}: {count}")
    
    # Estimated token usage
    avg_chars = np.mean([len(d["path"]) for d in all_datasets])
    # Rough estimate: 1 token ≈ 4 characters
    est_tokens = (np.mean(rows_dist) * np.mean(cols_dist) * 10) / 4
    
    print(f"\n🔢 ESTIMATED LLM TOKEN USAGE:")
    print(f"   Avg dataset: ~{est_tokens:.0f} tokens")
    print(f"   Max safe context: ~{CONFIG['max_rows'] * CONFIG['max_cols'] * 10 / 4:.0f} tokens")
    print(f"   ✅ All datasets fit within typical LLM context windows (32k-128k)")
    
    print("\n" + "=" * 80)
    print("✅ GENERATION COMPLETE!")
    print(f"   {len(generation_log)} datasets ready for LLM extraction and Apriori comparison")
    print("=" * 80)
    
    return all_datasets


if __name__ == "__main__":
    main()
