#!/usr/bin/env python3
"""Analyze raw model outputs from evaluation to diagnose failure patterns."""

import json
import os
import re
import csv
from pathlib import Path
from collections import Counter

RAW_DIR = Path("src/evaluation/eval_results/raw_20260301_2212")
RESULTS_CSV = Path("src/evaluation/eval_results/results_20260301_2212.csv")
EVAL_META = Path("data/eval_datasets_v2/eval_full_metadata.json")

def main():
    # Load results CSV for context
    with open(RESULTS_CSV) as f:
        results = list(csv.DictReader(f))
    
    # Load eval metadata for ground truth comparison
    with open(EVAL_META) as f:
        eval_meta = json.load(f)
    meta_by_name = {d["filename"]: d for d in eval_meta}
    
    print("=" * 80)
    print("  DETAILED FAILURE PATTERN ANALYSIS")
    print("=" * 80)
    
    pattern_counts = Counter()
    all_issues = []
    
    for fname in sorted(os.listdir(RAW_DIR)):
        filepath = RAW_DIR / fname
        dataset_name = fname.replace(".txt", "")
        
        with open(filepath) as f:
            raw = f.read()
        
        print(f"\n{'─' * 60}")
        print(f"  {dataset_name}")
        print(f"{'─' * 60}")
        
        issues = []
        
        # 1. Check for <think> tags
        has_think = "<think>" in raw and "</think>" in raw
        if not has_think:
            issues.append("NO_THINK: Missing <think>...</think> reasoning")
            pattern_counts["no_think"] += 1
        
        # 2. Parse JSON
        json_text = raw
        if has_think:
            json_text = raw.split("</think>", 1)[1].strip()
        
        parsed = []
        parse_ok = False
        try:
            parsed = json.loads(json_text)
            parse_ok = True
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", json_text, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                    parse_ok = True
                except:
                    pass
        
        if not parse_ok:
            # Try on raw
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                    parse_ok = True
                except:
                    pass
        
        if not parse_ok:
            issues.append("JSON_FAIL: Could not parse JSON at all")
            pattern_counts["json_fail"] += 1
        
        # 3. Check for post-JSON text
        if parse_ok and not has_think:
            # Find where JSON array ends
            last_bracket = raw.rfind("]")
            if last_bracket >= 0 and last_bracket < len(raw) - 5:
                after_json = raw[last_bracket+1:].strip()
                if after_json:
                    issues.append(f"POST_JSON_TEXT: '{after_json[:80]}...'")
                    pattern_counts["post_json_text"] += 1
        
        # 4. Analyze itemset format
        if parsed:
            items_list = parsed if isinstance(parsed, list) else []
            
            n_col_value = 0  # Correct "col:value" format
            n_split = 0       # Split "col" and "value" as separate items
            n_column_only = 0  # Just column name, no value
            n_value_concat = 0 # Wrong value like "rating:21" instead of individual
            n_below_min_support = 0
            n_duplicate_itemsets = 0
            n_halluc_rows = 0
            
            seen_itemsets = set()
            
            for item in items_list:
                if not isinstance(item, dict):
                    continue
                itemset = item.get("itemset", [])
                count = item.get("count", 0)
                rows = item.get("rows", item.get("row", []))
                
                # Check for duplicates
                key = frozenset(str(x).lower() for x in itemset)
                if key in seen_itemsets:
                    n_duplicate_itemsets += 1
                seen_itemsets.add(key)
                
                # Check count vs min_support
                if count < 3:
                    n_below_min_support += 1
                
                # Check row references
                for r in (rows if isinstance(rows, list) else []):
                    r_str = str(r).strip()
                    if not re.match(r"^Row \d+$", r_str):
                        n_halluc_rows += 1
                
                # Analyze item format
                for it in itemset:
                    it_str = str(it).strip()
                    if ":" in it_str:
                        # Has colon - could be correct or concatenated
                        parts = it_str.split(":")
                        if len(parts) == 2:
                            n_col_value += 1
                        else:
                            n_value_concat += 1
                    elif it_str.isdigit() or (it_str.replace(".", "").isdigit()):
                        n_split += 1  # Bare number = split from column name
                    else:
                        n_column_only += 1  # Just column name
            
            total_items = n_col_value + n_split + n_column_only + n_value_concat
            
            print(f"  Itemsets: {len(items_list)} predicted")
            print(f"  Item format breakdown ({total_items} total items):")
            if n_col_value:
                print(f"    ✅ Correct col:value  : {n_col_value}")
            if n_split:
                print(f"    ❌ Split (bare number) : {n_split}")
            if n_column_only:
                print(f"    ❌ Column name only    : {n_column_only}")
            if n_value_concat:
                print(f"    ❌ Value concatenation : {n_value_concat}")
            
            if n_split > 0:
                pattern_counts["item_split"] += 1
            if n_column_only > 0:
                pattern_counts["column_only"] += 1
            if n_col_value > 0 and n_split == 0 and n_column_only == 0:
                pattern_counts["correct_format"] += 1
            
            if n_below_min_support > 0:
                issues.append(f"BELOW_MIN_SUPPORT: {n_below_min_support} itemsets with count < 3")
                pattern_counts["below_min_support"] += 1
            if n_duplicate_itemsets > 0:
                issues.append(f"DUPLICATE_ITEMSETS: {n_duplicate_itemsets} duplicates")
                pattern_counts["duplicate_itemsets"] += 1
            if n_halluc_rows > 0:
                issues.append(f"HALLUC_ROWS: {n_halluc_rows} invalid row references")
                pattern_counts["halluc_rows"] += 1
        
        # 5. Check for repetition loop
        if len(raw) > 1000:
            # If the output is very long, check for repetition
            chunks = [raw[i:i+50] for i in range(0, len(raw), 50)]
            if len(chunks) > 5:
                most_common = Counter(chunks).most_common(1)
                if most_common[0][1] > 3:
                    issues.append(f"REPETITION_LOOP: Same chunk repeated {most_common[0][1]} times")
                    pattern_counts["repetition_loop"] += 1
        
        # Print issues
        for issue in issues:
            print(f"  ⚠️  {issue}")
        
        # Show first 2 raw itemsets for context
        if parsed and isinstance(parsed, list):
            print(f"  Sample output:")
            for item in parsed[:2]:
                if isinstance(item, dict):
                    print(f"    {json.dumps(item)}")
        
        all_issues.append({"name": dataset_name, "issues": issues})
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"  PATTERN SUMMARY (across {len(all_issues)} datasets)")
    print(f"{'=' * 80}")
    for pattern, count in pattern_counts.most_common():
        pct = count / len(all_issues) * 100
        print(f"  {pattern:<25} {count:>3} / {len(all_issues)}  ({pct:.0f}%)")
    
    # Ground truth comparison
    print(f"\n{'=' * 80}")
    print(f"  GROUND TRUTH vs MODEL FORMAT")
    print(f"{'=' * 80}")
    
    # Show what ground truth expects vs what model produces for first few
    for fname in sorted(os.listdir(RAW_DIR))[:5]:
        dataset_name = fname.replace(".txt", "")
        meta = meta_by_name.get(dataset_name)
        if not meta:
            continue
        
        gt = meta["ground_truth"][:2]
        with open(RAW_DIR / fname) as f:
            raw = f.read()
        
        try:
            model_items = json.loads(raw if raw.strip().startswith("[") else re.search(r"\[.*?\]", raw, re.DOTALL).group())[:2]
        except:
            model_items = []
        
        print(f"\n  {dataset_name}:")
        print(f"    Ground truth: {json.dumps(gt[0]['itemset']) if gt else 'N/A'}")
        print(f"    Model output: {json.dumps(model_items[0]['itemset']) if model_items else 'N/A'}")


if __name__ == "__main__":
    main()
