#!/usr/bin/env python3
"""
Create training data for fine-tuning V2.

This script:
1. Reads validated gpt-4o extraction results from artifacts/
2. Matches them with original CSV datasets from datasets_v2/
3. Creates training examples in ChatML format for HuggingFace
4. Includes Chain-of-Thought reasoning steps (optional)

Based on V1 lessons:
- Use real item names (column:value format)
- Add explicit rules against hallucination
- Include step-by-step reasoning

Output: training_data_v2/train.jsonl
"""

import json
import os
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
import hashlib

# Paths
DATASETS_DIR = Path("datasets_v2")
EXTRACTOR_OUTPUTS_DIR = Path("artifacts/extractor_outputs")
APRIORI_OUTPUTS_DIR = Path("artifacts/apriori_outputs")
OUTPUT_DIR = Path("training_data_v2")

# System prompt - enhanced with explicit anti-hallucination rules
SYSTEM_PROMPT = """You are a frequent itemset mining expert. Your task is to extract frequent itemsets from transaction data.

CRITICAL RULES:
1. ONLY use items that EXACTLY appear in the input CSV - NEVER invent or hallucinate items
2. Each item is in "column:value" format - use the exact format from the CSV
3. Count carefully - verify each count by listing the row numbers
4. A frequent itemset must appear in at least min_support transactions
5. Return ONLY valid JSON array format

OUTPUT FORMAT:
Return a JSON array where each element has:
- "itemset": array of items (use exact strings from CSV)
- "count": integer (number of rows containing ALL items in the itemset)
- "evidence_rows": array of row labels like ["Row 1", "Row 3", "Row 5"]

VALIDATION:
- count must equal len(evidence_rows)
- every item in itemset must exist in every evidence row
- no duplicate items within an itemset"""


def load_csv_content(csv_path: Path) -> str:
    """Load CSV file and return as string."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def parse_csv_transactions(csv_path: Path) -> List[List[str]]:
    """Parse CSV into list of transactions (each row = list of column:value items)."""
    transactions = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items = [f"{col}:{val}" for col, val in row.items() if val and str(val).strip()]
            transactions.append(items)
    return transactions


def create_user_prompt(csv_content: str, min_support: int) -> str:
    """Create the user prompt for extraction task."""
    return f"""Extract frequent itemsets from this CSV data with minimum support count of {min_support}.

CSV DATA:
{csv_content}

Return ONLY a JSON array of itemsets. Each itemset must have:
- "itemset": list of items (exact strings from CSV)
- "count": number of rows containing all items
- "evidence_rows": list of row labels ["Row 1", "Row 2", ...]"""


def create_cot_response(transactions: List[List[str]], llm_output: List[Dict], min_support: int) -> str:
    """Create Chain-of-Thought response with reasoning steps."""
    lines = []
    
    # Step 1: Parse
    lines.append("## Step 1: Parse CSV")
    lines.append(f"Total transactions: {len(transactions)}")
    lines.append("")
    
    # Step 2: Show transactions (first 5 max for brevity)
    lines.append("## Step 2: Transactions overview")
    for i, trans in enumerate(transactions[:5], 1):
        items_str = ", ".join(trans[:5])  # First 5 items
        if len(trans) > 5:
            items_str += f", ... ({len(trans)} items total)"
        lines.append(f"Row {i}: [{items_str}]")
    if len(transactions) > 5:
        lines.append(f"... ({len(transactions)} rows total)")
    lines.append("")
    
    # Step 3: Found itemsets
    lines.append(f"## Step 3: Frequent itemsets (support >= {min_support})")
    for rec in llm_output[:10]:  # Limit to first 10 for brevity
        itemset = rec.get('itemset', [])
        count = rec.get('count', 0)
        rows = rec.get('evidence_rows', rec.get('rows', []))
        itemset_str = ", ".join(itemset)
        rows_str = ", ".join(str(r) for r in rows[:5])
        if len(rows) > 5:
            rows_str += f", ..."
        lines.append(f"- {{{itemset_str}}}: count={count}, rows=[{rows_str}]")
    if len(llm_output) > 10:
        lines.append(f"... ({len(llm_output)} itemsets total)")
    lines.append("")
    
    # Final output
    lines.append("## Final Output:")
    # Convert to standard format (using 'evidence_rows' consistently)
    output_formatted = []
    for rec in llm_output:
        output_formatted.append({
            "itemset": rec.get('itemset', []),
            "count": rec.get('count', 0),
            "evidence_rows": rec.get('evidence_rows', rec.get('rows', []))
        })
    lines.append(json.dumps(output_formatted, ensure_ascii=False, indent=2))
    
    return "\n".join(lines)


def create_simple_response(llm_output: List[Dict]) -> str:
    """Create simple JSON response (no CoT)."""
    output_formatted = []
    for rec in llm_output:
        output_formatted.append({
            "itemset": rec.get('itemset', []),
            "count": rec.get('count', 0),
            "evidence_rows": rec.get('evidence_rows', rec.get('rows', []))
        })
    return json.dumps(output_formatted, ensure_ascii=False, indent=2)


def find_matching_files() -> List[Dict[str, Path]]:
    """Find matching dataset CSV and extractor output files."""
    matches = []
    
    for extractor_file in sorted(EXTRACTOR_OUTPUTS_DIR.glob("gpt-4o_extractor_output_*.json")):
        # Parse filename: gpt-4o_extractor_output_ds_0001_7x3_dd5b6660_c60c507e34bb.json
        parts = extractor_file.stem.split("_")
        # Find ds_NNNN pattern
        ds_idx = None
        for i, part in enumerate(parts):
            if part == "ds":
                ds_idx = i
                break
        
        if ds_idx is None:
            continue
        
        # Reconstruct dataset filename: ds_0001_7x3_dd5b6660.csv
        # The pattern is: ds_NNNN_RxC_hash
        try:
            ds_num = parts[ds_idx + 1]  # e.g., "0001"
            dimensions = parts[ds_idx + 2]  # e.g., "7x3"
            csv_hash = parts[ds_idx + 3]  # e.g., "dd5b6660"
            
            csv_filename = f"ds_{ds_num}_{dimensions}_{csv_hash}.csv"
            csv_path = DATASETS_DIR / csv_filename
            
            if csv_path.exists():
                matches.append({
                    "csv_path": csv_path,
                    "extractor_path": extractor_file,
                    "ds_num": ds_num,
                    "dimensions": dimensions,
                    "hash": csv_hash
                })
        except (IndexError, ValueError):
            continue
    
    return matches


def create_training_example(match: Dict[str, Path], use_cot: bool = False, min_support: int = 2) -> Optional[Dict[str, Any]]:
    """Create a single training example from matched files."""
    csv_path = match["csv_path"]
    extractor_path = match["extractor_path"]
    
    try:
        # Load CSV content
        csv_content = load_csv_content(csv_path)
        transactions = parse_csv_transactions(csv_path)
        
        # Load extractor output
        with open(extractor_path, 'r', encoding='utf-8') as f:
            llm_output = json.load(f)
        
        # Skip empty outputs (no itemsets found)
        if not llm_output:
            return None
        
        # Create messages in ChatML format
        user_prompt = create_user_prompt(csv_content, min_support)
        
        if use_cot:
            assistant_response = create_cot_response(transactions, llm_output, min_support)
        else:
            assistant_response = create_simple_response(llm_output)
        
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ],
            "metadata": {
                "dataset_id": match["ds_num"],
                "dimensions": match["dimensions"],
                "hash": match["hash"],
                "num_transactions": len(transactions),
                "num_itemsets": len(llm_output),
                "source_csv": str(csv_path.name),
                "source_extractor": str(extractor_path.name)
            }
        }
    except Exception as e:
        print(f"Error processing {csv_path.name}: {e}")
        return None


def main():
    print("=" * 70)
    print("TRAINING DATA GENERATOR V2")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find matching files
    print("\n📂 Scanning for matching files...")
    matches = find_matching_files()
    print(f"   Found {len(matches)} matched dataset/extractor pairs")
    
    # Generate training examples
    print("\n📝 Generating training examples...")
    
    # Create two versions: with and without CoT
    examples_simple = []
    examples_cot = []
    
    for i, match in enumerate(matches, 1):
        if i % 50 == 0:
            print(f"   Processing {i}/{len(matches)}...")
        
        # Simple version (just JSON output)
        example_simple = create_training_example(match, use_cot=False)
        if example_simple:
            examples_simple.append(example_simple)
        
        # CoT version (with reasoning steps)
        example_cot = create_training_example(match, use_cot=True)
        if example_cot:
            examples_cot.append(example_cot)
    
    print(f"\n✅ Generated {len(examples_simple)} simple examples")
    print(f"✅ Generated {len(examples_cot)} CoT examples")
    
    # Save JSONL files
    simple_path = OUTPUT_DIR / "train_simple.jsonl"
    cot_path = OUTPUT_DIR / "train_cot.jsonl"
    combined_path = OUTPUT_DIR / "train_combined.jsonl"
    
    # Save simple version
    with open(simple_path, 'w', encoding='utf-8') as f:
        for ex in examples_simple:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\n💾 Saved: {simple_path}")
    
    # Save CoT version
    with open(cot_path, 'w', encoding='utf-8') as f:
        for ex in examples_cot:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"💾 Saved: {cot_path}")
    
    # Save combined version (mix of both for variety)
    combined = []
    for i in range(len(examples_simple)):
        # Alternate between simple and CoT
        if i % 2 == 0:
            combined.append(examples_simple[i])
        else:
            combined.append(examples_cot[i])
    
    with open(combined_path, 'w', encoding='utf-8') as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"💾 Saved: {combined_path}")
    
    # Save metadata
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_examples": len(examples_simple),
        "examples_simple": len(examples_simple),
        "examples_cot": len(examples_cot),
        "source_datasets_dir": str(DATASETS_DIR),
        "source_extractor_dir": str(EXTRACTOR_OUTPUTS_DIR),
        "system_prompt_hash": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12],
        "min_support": 2
    }
    
    metadata_path = OUTPUT_DIR / "training_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved: {metadata_path}")
    
    # Print sample
    print("\n" + "=" * 70)
    print("SAMPLE TRAINING EXAMPLE (Simple)")
    print("=" * 70)
    if examples_simple:
        sample = examples_simple[0]
        print(f"\nSystem prompt: {sample['messages'][0]['content'][:100]}...")
        print(f"\nUser prompt: {sample['messages'][1]['content'][:200]}...")
        print(f"\nAssistant response: {sample['messages'][2]['content'][:300]}...")
        print(f"\nMetadata: {sample['metadata']}")
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print(f"""
📊 SUMMARY:
   - Simple examples: {len(examples_simple)}
   - CoT examples: {len(examples_cot)}
   - Combined: {len(combined)}
   
📁 OUTPUT FILES:
   - {simple_path}
   - {cot_path}
   - {combined_path}
   - {metadata_path}
   
🎯 NEXT STEPS:
   1. Upload to HuggingFace or use with local trainer
   2. Fine-tune Qwen2.5-3B or 7B model
   3. Evaluate on held-out test set
""")


if __name__ == "__main__":
    main()
