"""Evaluation Pipeline for Fine-tuned Itemset Extraction Model.

Compares fine-tuned Qwen2.5-3B model against ground-truth Apriori results.
Runs inference on evaluation datasets and computes comprehensive metrics.

Requirements:
  pip install torch transformers peft accelerate pandas

Usage:
  python eval_finetuned_model.py --data-dir eval_datasets --min-support 3 --max-size 3
  python eval_finetuned_model.py --data eval_datasets/eval_0001_20x30.csv --min-support 2
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# =========================
# Model Configuration
# =========================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_NAME = "OliverSlivka/qwen2.5-3b-itemset-extractor"

# Global model cache (loaded once)
_model = None
_tokenizer = None


def load_model():
    """Load fine-tuned model (cached for reuse across datasets)."""
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"📥 Loading base model: {MODEL_NAME}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device_map = {"": 0}
        dtype = torch.float16
        print(f"   Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device_map = "cpu"
        dtype = torch.float32
        print("   ⚠️ No GPU available, using CPU (will be slow)")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    print(f"🔗 Loading adapter: {ADAPTER_NAME}")
    _model = PeftModel.from_pretrained(base_model, ADAPTER_NAME)
    _model.eval()
    
    print("✅ Model loaded successfully!")
    return _model, _tokenizer


def extract_itemsets_with_model(
    transactions: List[List[str]], 
    min_support: int,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
) -> List[Dict[str, Any]]:
    """Extract frequent itemsets using fine-tuned model."""
    import torch
    
    model, tokenizer = load_model()
    
    # Build prompt
    rows_text = "\n".join(
        f"Row {i+1}: {', '.join(str(x) for x in row)}" 
        for i, row in enumerate(transactions)
    )
    
    prompt = f"""Extract all frequent itemsets from the following transactional data.
Minimum support threshold: {min_support}

Transactions:
{rows_text}

Return the result as a JSON array where each object has:
- "itemset": list of items
- "support": number of transactions containing the itemset
- "rows": list of row numbers where the itemset appears"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Parse JSON from response
    try:
        # Try direct parse first
        parsed = json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON array from text
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = []
        else:
            parsed = []
    
    # Normalize output format
    results = []
    for rec in parsed:
        if not isinstance(rec, dict):
            continue
        itemset = rec.get('itemset', [])
        if not isinstance(itemset, list) or not itemset:
            continue
        
        support = rec.get('support', rec.get('count', 0))
        rows = rec.get('rows', rec.get('evidence_rows', []))
        
        # Normalize items to lowercase strings
        itemset = [str(x).strip().lower() for x in itemset]
        
        # Normalize rows
        norm_rows = []
        for r in rows:
            if isinstance(r, int):
                norm_rows.append(f"Row {r}")
            elif isinstance(r, str) and r.lower().startswith('row '):
                try:
                    idx = int(r.split()[1])
                    norm_rows.append(f"Row {idx}")
                except:
                    pass
            else:
                try:
                    idx = int(str(r))
                    norm_rows.append(f"Row {idx}")
                except:
                    pass
        
        results.append({
            'itemset': sorted(itemset),
            'count': support if isinstance(support, int) else len(norm_rows),
            'rows': sorted(set(norm_rows)),
        })
    
    return results


# =========================
# Apriori Implementation (from pipeline.py)
# =========================

def apriori_frequent_itemsets(
    transactions: List[List[str]], 
    min_support: int = 3, 
    max_size: int = 3
) -> List[Dict[str, Any]]:
    """Generate ground-truth frequent itemsets using Apriori."""
    if not transactions:
        return []
    
    row_labels = [f"Row {i+1}" for i in range(len(transactions))]
    counts: Dict[Tuple[str,...], Dict[str, Any]] = {}
    
    # Level 1: single items
    for idx, trans in enumerate(transactions):
        seen = set()
        for item in trans:
            item = str(item).strip().lower()
            if not item or item in seen:
                continue
            seen.add(item)
            k = (item,)
            if k not in counts:
                counts[k] = {'count': 0, 'rows': []}
            counts[k]['count'] += 1
            counts[k]['rows'].append(row_labels[idx])
    
    def prune(d):
        return {k: v for k, v in d.items() if v['count'] >= min_support}
    
    freq_levels = []
    L1 = prune(counts)
    if not L1:
        return []
    freq_levels.append(L1)
    
    current = L1
    k = 2
    while k <= max_size and current:
        prev_keys = sorted(current.keys())
        candidates = set()
        for i in range(len(prev_keys)):
            for j in range(i+1, len(prev_keys)):
                a, b = prev_keys[i], prev_keys[j]
                if a[:k-2] == b[:k-2]:
                    merged = tuple(sorted(set(a) | set(b)))
                    if len(merged) == k:
                        if all(tuple(sorted(sub)) in current for sub in itertools.combinations(merged, k-1)):
                            candidates.add(merged)
        if not candidates:
            break
        
        cand_counts = {c: {'count': 0, 'rows': []} for c in candidates}
        for idx, trans in enumerate(transactions):
            tset = set(map(lambda x: str(x).strip().lower(), trans))
            for cand in candidates:
                if set(cand).issubset(tset):
                    cand_counts[cand]['count'] += 1
                    cand_counts[cand]['rows'].append(row_labels[idx])
        
        current = prune(cand_counts)
        if current:
            freq_levels.append(current)
        k += 1
    
    out = []
    total = len(transactions)
    for level in freq_levels:
        for itemset, info in level.items():
            rows = info['rows']
            count = info['count']
            out.append({
                'itemset': list(itemset),
                'count': count,
                'rows': rows,
                'support': count / total if total else 0,
                'size': len(itemset)
            })
    return out


# =========================
# CSV Loading (from pipeline.py)
# =========================

def load_transactions_csv(path: str) -> Tuple[pd.DataFrame, List[List[str]], Dict[str, Any]]:
    """Load transactions from CSV file."""
    import csv
    
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    
    # Detect format
    sample_lines = []
    with p.open('r', encoding='utf-8', errors='ignore') as fh:
        for _ in range(40):
            ln = fh.readline()
            if not ln:
                break
            sample_lines.append(ln)
    
    cleaned_lines = [ln for ln in sample_lines if ln.strip()]
    try:
        dialect = csv.Sniffer().sniff(''.join(cleaned_lines) or '', delimiters=",;\t|")
        sep_guess = dialect.delimiter
    except:
        sep_guess = ','
    
    try:
        df = pd.read_csv(path, sep=sep_guess, low_memory=False)
    except:
        df = pd.read_csv(path, sep=sep_guess, engine='python', on_bad_lines='skip')
    
    # Detect format
    lower_cols = [c.lower() for c in df.columns]
    has_tid = any(c in lower_cols for c in ['transaction_id','tid','trans_id','t_id'])
    has_item = any(c in lower_cols for c in ['item','product','sku'])
    
    if has_tid and has_item:
        fmt = 'long'
    elif df.shape[1] == 1:
        fmt = 'single-column'
    else:
        fmt = 'wide'
    
    transactions = []
    if fmt == 'long':
        cols_map = {c.lower(): c for c in df.columns}
        tid_col = next((cols_map[c] for c in ['transaction_id','tid','trans_id','t_id'] if c in cols_map), None)
        item_col = next((cols_map[c] for c in ['item','product','sku'] if c in cols_map), None)
        grouped = df.groupby(tid_col)[item_col].apply(lambda s: [str(v).strip() for v in s if str(v).strip()])
        transactions = grouped.tolist()
    elif fmt == 'wide':
        for _, row in df.iterrows():
            items = []
            for v in row.tolist():
                if pd.isna(v):
                    continue
                sval = str(v).strip()
                if not sval or sval.lower() in {'nan','none'}:
                    continue
                items.append(sval)
            if items:
                transactions.append(items)
    else:
        for raw in df.iloc[:, 0].astype(str):
            parts = re.split(r'[;,|\t]', raw)
            if len(parts) == 1 and ' ' in raw and ',' not in raw and ';' not in raw:
                parts = raw.split()
            cleaned = [p.strip() for p in parts if p.strip()]
            if cleaned:
                transactions.append(cleaned)
    
    return df, transactions, {'format': fmt, 'sep': sep_guess}


# =========================
# Comparison Metrics
# =========================

def compute_metrics(
    apriori_sets: List[Dict[str, Any]], 
    model_sets: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute comparison metrics between Apriori and model outputs."""
    
    # Canonicalize itemsets for comparison
    def canon(itemset):
        return tuple(sorted(str(x).strip().lower() for x in itemset))
    
    apriori_itemsets = {canon(s['itemset']) for s in apriori_sets}
    model_itemsets = {canon(s['itemset']) for s in model_sets}
    
    # Set-based metrics
    true_positives = apriori_itemsets & model_itemsets
    false_positives = model_itemsets - apriori_itemsets
    false_negatives = apriori_itemsets - model_itemsets
    
    precision = len(true_positives) / len(model_itemsets) if model_itemsets else 0.0
    recall = len(true_positives) / len(apriori_itemsets) if apriori_itemsets else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Jaccard similarity
    union = apriori_itemsets | model_itemsets
    jaccard = len(true_positives) / len(union) if union else 1.0
    
    # Per-size metrics
    def by_size(sets_list):
        result = {}
        for s in sets_list:
            size = len(s['itemset'])
            if size not in result:
                result[size] = set()
            result[size].add(canon(s['itemset']))
        return result
    
    apriori_by_size = by_size(apriori_sets)
    model_by_size = by_size(model_sets)
    
    size_metrics = {}
    all_sizes = set(apriori_by_size.keys()) | set(model_by_size.keys())
    for size in sorted(all_sizes):
        apr = apriori_by_size.get(size, set())
        mod = model_by_size.get(size, set())
        tp = len(apr & mod)
        fp = len(mod - apr)
        fn = len(apr - mod)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        size_metrics[f"size_{size}"] = {
            "apriori_count": len(apr),
            "model_count": len(mod),
            "precision": round(p, 4),
            "recall": round(r, 4),
        }
    
    return {
        "apriori_total": len(apriori_itemsets),
        "model_total": len(model_itemsets),
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "jaccard_similarity": round(jaccard, 4),
        "exact_match": apriori_itemsets == model_itemsets,
        "size_breakdown": size_metrics,
    }


# =========================
# Main Evaluation Pipeline
# =========================

def evaluate_single_dataset(
    data_path: str,
    min_support: int,
    max_size: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate model on a single dataset."""
    
    # Load data
    df, transactions, meta = load_transactions_csv(data_path)
    print(f"   Loaded: {len(transactions)} transactions, format={meta['format']}")
    
    # Compute hash for filenames
    sha = hashlib.sha256()
    with open(data_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    file_hash = sha.hexdigest()[:12]
    stem = Path(data_path).stem
    
    # Run Apriori (ground truth)
    apriori_start = time.perf_counter()
    apriori_sets = apriori_frequent_itemsets(transactions, min_support, max_size)
    apriori_time = time.perf_counter() - apriori_start
    print(f"   Apriori: {len(apriori_sets)} itemsets in {apriori_time*1000:.1f}ms")
    
    # Run fine-tuned model
    model_start = time.perf_counter()
    model_sets = extract_itemsets_with_model(transactions, min_support)
    model_time = time.perf_counter() - model_start
    print(f"   Model: {len(model_sets)} itemsets in {model_time*1000:.1f}ms")
    
    # Compute metrics
    metrics = compute_metrics(apriori_sets, model_sets)
    
    # Save outputs
    result = {
        "dataset": Path(data_path).name,
        "dataset_hash": file_hash,
        "num_transactions": len(transactions),
        "min_support": min_support,
        "max_size": max_size,
        "apriori_time_ms": round(apriori_time * 1000, 2),
        "model_time_ms": round(model_time * 1000, 2),
        "metrics": metrics,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    
    # Save detailed outputs
    output_file = output_dir / f"eval_{stem}_{file_hash}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            **result,
            "apriori_output": apriori_sets,
            "model_output": model_sets,
        }, f, ensure_ascii=False, indent=2)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned itemset extraction model")
    parser.add_argument('--data', help='Single dataset CSV path')
    parser.add_argument('--data-dir', help='Directory containing CSV datasets')
    parser.add_argument('--min-support', type=int, default=3, help='Minimum support threshold')
    parser.add_argument('--max-size', type=int, default=3, help='Maximum itemset size for Apriori')
    parser.add_argument('--output-dir', default='eval_results', help='Output directory for results')
    parser.add_argument('--limit', type=int, help='Limit number of datasets to evaluate')
    args = parser.parse_args()
    
    # Determine datasets to evaluate
    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            print(f"❌ Directory not found: {args.data_dir}", file=sys.stderr)
            sys.exit(1)
        data_files = sorted([
            os.path.join(args.data_dir, f) 
            for f in os.listdir(args.data_dir) 
            if f.lower().endswith('.csv')
        ])
        if args.limit:
            data_files = data_files[:args.limit]
    elif args.data:
        data_files = [args.data]
    else:
        print("❌ Please specify --data or --data-dir", file=sys.stderr)
        sys.exit(1)
    
    if not data_files:
        print("❌ No CSV files found", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FINE-TUNED MODEL EVALUATION")
    print("=" * 60)
    print(f"Datasets: {len(data_files)}")
    print(f"Min support: {args.min_support}")
    print(f"Max itemset size: {args.max_size}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Load model once
    load_model()
    
    # Run evaluation
    all_results = []
    total_start = time.perf_counter()
    
    for i, data_path in enumerate(data_files, 1):
        print(f"\n[{i}/{len(data_files)}] {Path(data_path).name}")
        try:
            result = evaluate_single_dataset(
                data_path, 
                args.min_support, 
                args.max_size,
                output_dir,
            )
            all_results.append(result)
            
            m = result['metrics']
            print(f"   → P={m['precision']:.2%} R={m['recall']:.2%} F1={m['f1_score']:.2%} | "
                  f"TP={m['true_positives']} FP={m['false_positives']} FN={m['false_negatives']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_results.append({
                "dataset": Path(data_path).name,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            })
    
    total_time = time.perf_counter() - total_start
    
    # Compute aggregate metrics
    successful = [r for r in all_results if 'metrics' in r]
    
    if successful:
        avg_precision = sum(r['metrics']['precision'] for r in successful) / len(successful)
        avg_recall = sum(r['metrics']['recall'] for r in successful) / len(successful)
        avg_f1 = sum(r['metrics']['f1_score'] for r in successful) / len(successful)
        avg_jaccard = sum(r['metrics']['jaccard_similarity'] for r in successful) / len(successful)
        exact_matches = sum(1 for r in successful if r['metrics']['exact_match'])
        
        summary = {
            "total_datasets": len(data_files),
            "successful_evaluations": len(successful),
            "failed_evaluations": len(all_results) - len(successful),
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_f1_score": round(avg_f1, 4),
            "avg_jaccard": round(avg_jaccard, 4),
            "exact_match_count": exact_matches,
            "exact_match_rate": round(exact_matches / len(successful), 4) if successful else 0,
            "total_time_seconds": round(total_time, 2),
            "timestamp": datetime.now(UTC).isoformat(),
            "config": {
                "min_support": args.min_support,
                "max_size": args.max_size,
                "model": ADAPTER_NAME,
            },
        }
        
        # Save summary
        summary_file = output_dir / "evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Save all results
        all_results_file = output_dir / "all_results.json"
        with open(all_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Datasets evaluated: {len(successful)}/{len(data_files)}")
        print(f"Average Precision: {avg_precision:.2%}")
        print(f"Average Recall: {avg_recall:.2%}")
        print(f"Average F1 Score: {avg_f1:.2%}")
        print(f"Average Jaccard: {avg_jaccard:.2%}")
        print(f"Exact Matches: {exact_matches}/{len(successful)} ({exact_matches/len(successful)*100:.1f}%)")
        print(f"Total Time: {total_time:.1f}s")
        print(f"\n📁 Results saved to: {output_dir}")
    else:
        print("\n❌ No successful evaluations")


if __name__ == "__main__":
    main()
