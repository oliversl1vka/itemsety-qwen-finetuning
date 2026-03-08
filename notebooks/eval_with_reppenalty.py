#!/usr/bin/env python3
"""
Eval with repetition_penalty fix — same as eval_raw_capture.py but adds
repetition_penalty=1.3 and no_repeat_ngram_size=3 to prevent degenerate loops.

Root cause: SFT-only model enters infinite repetition in <think> and JSON.
This fix should dramatically improve F1 by letting the model actually FINISH.

Upload to TLJH next to sft_checkpoint/ and datasets_v2/ and run:
    python eval_with_reppenalty.py

Results saved to eval_reppenalty/ with one .txt per dataset + summary.
"""

import os, json, subprocess, gc, glob, random, re, time, itertools
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ═══════════════════════════════════════════════════════════════════
ADAPTER_PATH = "./sft_checkpoint"
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 4096          # Reduced from 8192 — with rep penalty, model should finish sooner
TEMPERATURE = 0.1
REPETITION_PENALTY = 1.3       # ← THE FIX: prevents degenerate repetition loops
NO_REPEAT_NGRAM_SIZE = 3       # ← EXTRA GUARD: prevents exact 3-gram repetitions
DATASET_DIR = "./datasets_v2"
NUM_DATASETS = 30              # More datasets for better signal
MIN_SUPPORT = 3
MAX_SIZE = 3
SEED = 42
OUTPUT_DIR = "./eval_reppenalty"

# ═══════════════════════════════════════════════════════════════════
# GPU setup
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("  EVAL WITH REPETITION PENALTY FIX")
print(f"  repetition_penalty={REPETITION_PENALTY}")
print(f"  no_repeat_ngram_size={NO_REPEAT_NGRAM_SIZE}")
print(f"  max_new_tokens={MAX_NEW_TOKENS}")
print("=" * 70)

try:
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
    )
    gpus = [(int(l.split(",")[0].strip()), int(l.split(",")[1].strip())) for l in smi.strip().split("\n")]
    best = max(gpus, key=lambda g: g[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best[0])
    print(f"✅ Using GPU {best[0]} ({best[1]/1024:.1f} GB free)")
except Exception as e:
    print(f"⚠️ nvidia-smi failed: {e}")

os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
gc.collect()
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════
# Load model
# ═══════════════════════════════════════════════════════════════════
print(f"\n📥 Loading model from {ADAPTER_PATH}...")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(model)
print(f"✅ Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ═══════════════════════════════════════════════════════════════════
# System prompt (matches training_utils.py)
# ═══════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "You are a frequent itemset extractor. Given CSV transaction data and a "
    "minimum support count, identify all itemsets whose items co-occur in at "
    "least that many rows.\n\n"
    "Rules:\n"
    "1. Scan single items, pairs, and triples (up to size 3)\n"
    "2. Count = number of distinct rows containing ALL items in the itemset\n"
    "3. Only report itemsets with count >= min_support\n"
    "4. Canonicalize items: lowercase, trimmed, sorted alphabetically\n"
    '5. Row references: "Row N" format, 1-based indexing\n\n'
    "Think step by step inside <think> tags, then output ONLY a JSON array:\n"
    '[{"itemset": ["item1", "item2"], "count": N, "rows": ["Row 1", "Row 3"]}]'
)

# ═══════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape
    transactions, rows_text, all_items = [], [], set()
    for idx, row in df.iterrows():
        items = [str(v).strip() for v in row.values if pd.notna(v) and str(v).strip()]
        transactions.append(items)
        rows_text.append(f"Row {idx + 1}: {', '.join(items)}")
        for item in items:
            all_items.add(item.lower())
    return transactions, "\n".join(rows_text), n_rows, n_cols, all_items


def apriori(transactions, min_support=3, max_size=3):
    if not transactions:
        return []
    row_labels = [f"Row {i+1}" for i in range(len(transactions))]
    counts = {}
    for idx, trans in enumerate(transactions):
        seen = set()
        for item in trans:
            norm = str(item).strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            k = (norm,)
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


def run_inference(csv_text, min_support):
    """Run inference with repetition_penalty to prevent degenerate loops."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Find all frequent itemsets with minimum support count = {min_support} in this dataset:\n\n{csv_text}"},
    ]
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if isinstance(tokenized, dict):
        input_ids = tokenized["input_ids"]
    else:
        input_ids = tokenized
    attention_mask = torch.ones_like(input_ids)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    
    prompt_tokens = input_ids.shape[1]
    
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            repetition_penalty=REPETITION_PENALTY,        # ← THE FIX
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,    # ← EXTRA GUARD
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    
    new_tokens = outputs.shape[1] - prompt_tokens
    raw = tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)
    
    return {
        "raw_output": raw,
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "hit_limit": new_tokens >= MAX_NEW_TOKENS - 10,  # small buffer for off-by-one
        "time_s": round(elapsed, 1),
    }


def try_parse_json(raw):
    """Try to extract JSON from raw output — flexible parsing."""
    has_think = "<think>" in raw and "</think>" in raw
    
    # Get text after </think> if present
    json_text = raw
    if has_think:
        parts = raw.split("</think>", 1)
        json_text = parts[1].strip() if len(parts) > 1 else ""
    
    # Try direct parse of post-think text
    for text in [json_text, raw]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed, True, has_think
        except:
            pass
        # Try regex extraction — but ONLY on post-think text first
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, list):
                    return parsed, True, has_think
            except:
                pass
    
    return [], False, has_think


def canon(itemset):
    return frozenset(str(x).strip().lower() for x in itemset)


def compute_f1(apriori_sets, model_items, all_items=None):
    """F1 computation with hallucination detection."""
    apr_c = {canon(s["itemset"]) for s in apriori_sets}
    mod_c = set()
    hallucinated_items = set()
    
    for rec in model_items:
        if isinstance(rec, dict) and "itemset" in rec:
            itemset = rec["itemset"]
            if isinstance(itemset, list) and itemset:
                items_fs = frozenset(str(x).strip().lower() for x in itemset)
                mod_c.add(items_fs)
                # Check for hallucinated items (not in the CSV)
                if all_items:
                    for item in items_fs:
                        if item not in all_items:
                            hallucinated_items.add(item)
    
    tp = apr_c & mod_c
    p = len(tp) / len(mod_c) if mod_c else 0.0
    r = len(tp) / len(apr_c) if apr_c else 0.0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    
    return {
        "apriori": len(apr_c), "model": len(mod_c), "tp": len(tp),
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "hallucinated_items": len(hallucinated_items),
    }


# ═══════════════════════════════════════════════════════════════════
# Select datasets — mix of sizes for good coverage
# ═══════════════════════════════════════════════════════════════════
all_csvs = sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv")))
print(f"\n📊 Found {len(all_csvs)} CSV files")

random.seed(SEED)
selected = random.sample(all_csvs, min(NUM_DATASETS, len(all_csvs)))

# Sort by expected complexity (rows × cols from filename)
def complexity(path):
    m = re.search(r"_(\d+)x(\d+)_", os.path.basename(path))
    return int(m.group(1)) * int(m.group(2)) if m else 0
selected.sort(key=complexity)

print(f"   Selected {len(selected)} datasets (sorted by complexity)")

# ═══════════════════════════════════════════════════════════════════
# Run evaluation
# ═══════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*70}")
print(f"  Running with repetition_penalty={REPETITION_PENALTY}")
print(f"  no_repeat_ngram_size={NO_REPEAT_NGRAM_SIZE}")
print(f"  max_new_tokens={MAX_NEW_TOKENS}")
print(f"{'='*70}")

results = []
total_start = time.perf_counter()

for i, csv_path in enumerate(selected, 1):
    ds_name = os.path.basename(csv_path)
    print(f"\n[{i}/{len(selected)}] {ds_name}")
    
    try:
        # Load + Apriori
        transactions, csv_text, n_rows, n_cols, all_items = load_csv(csv_path)
        apriori_sets = apriori(transactions, MIN_SUPPORT, MAX_SIZE)
        print(f"   {n_rows}×{n_cols} | Apriori: {len(apriori_sets)} itemsets")
        
        # Inference
        inf = run_inference(csv_text, MIN_SUPPORT)
        
        # Parse
        parsed_items, parse_ok, has_think = try_parse_json(inf["raw_output"])
        metrics = compute_f1(apriori_sets, parsed_items, all_items)
        
        # Status line
        limit_warn = " ⚠️ HIT LIMIT" if inf["hit_limit"] else ""
        think_emoji = "✅" if has_think else "❌"
        json_emoji = "✅" if parse_ok else "❌"
        
        print(f"   Tokens: {inf['prompt_tokens']}→{inf['new_tokens']} | {inf['time_s']}s{limit_warn}")
        print(f"   Think={think_emoji} | JSON={json_emoji} ({len(parsed_items)} items) | F1={metrics['f1']:.0%} (P={metrics['precision']:.0%} R={metrics['recall']:.0%})")
        if metrics['hallucinated_items'] > 0:
            print(f"   ⚠️ {metrics['hallucinated_items']} hallucinated items")
        
        # Save raw output
        raw_path = os.path.join(OUTPUT_DIR, f"{ds_name.replace('.csv', '')}_raw.txt")
        with open(raw_path, "w") as f:
            f.write(f"Dataset: {ds_name}\n")
            f.write(f"Size: {n_rows}×{n_cols}\n")
            f.write(f"Apriori itemsets: {len(apriori_sets)}\n")
            f.write(f"Prompt tokens: {inf['prompt_tokens']}\n")
            f.write(f"New tokens: {inf['new_tokens']}\n")
            f.write(f"Hit limit ({MAX_NEW_TOKENS}): {inf['hit_limit']}\n")
            f.write(f"Time: {inf['time_s']}s\n")
            f.write(f"Think: {has_think}\n")
            f.write(f"JSON parsed: {parse_ok} ({len(parsed_items)} items)\n")
            f.write(f"F1: {metrics['f1']}\n")
            f.write(f"Precision: {metrics['precision']}\n")
            f.write(f"Recall: {metrics['recall']}\n")
            f.write(f"Hallucinated items: {metrics['hallucinated_items']}\n")
            f.write(f"repetition_penalty: {REPETITION_PENALTY}\n")
            f.write(f"no_repeat_ngram_size: {NO_REPEAT_NGRAM_SIZE}\n")
            f.write(f"\n{'='*70}\n")
            f.write(f"RAW MODEL OUTPUT ({len(inf['raw_output'])} chars):\n")
            f.write(f"{'='*70}\n\n")
            f.write(inf["raw_output"])
            f.write(f"\n\n{'='*70}\n")
            f.write(f"APRIORI GROUND TRUTH ({len(apriori_sets)} itemsets):\n")
            f.write(f"{'='*70}\n\n")
            f.write(json.dumps(apriori_sets, indent=2))
        
        results.append({
            "dataset": ds_name, "rows": n_rows, "cols": n_cols,
            "apriori_count": len(apriori_sets),
            "prompt_tokens": inf["prompt_tokens"],
            "new_tokens": inf["new_tokens"],
            "hit_limit": inf["hit_limit"],
            "time_s": inf["time_s"],
            "has_think": has_think,
            "parse_ok": parse_ok,
            "model_count": len(parsed_items),
            "hallucinated_items": metrics["hallucinated_items"],
            **{k: v for k, v in metrics.items() if k != "hallucinated_items"},
        })
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results.append({"dataset": ds_name, "error": str(e)})

total_time = time.perf_counter() - total_start

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
ok = [r for r in results if "f1" in r]
n = len(ok)

print(f"\n\n{'='*70}")
print(f"  SUMMARY ({n} datasets)")
print(f"  repetition_penalty={REPETITION_PENALTY}  no_repeat_ngram_size={NO_REPEAT_NGRAM_SIZE}")
print(f"{'='*70}")

if ok:
    print(f"\n  {'Dataset':<35} {'Size':>6} {'Apr':>4} {'Mod':>4} {'Tok':>5} {'Lim':>4} {'Think':>6} {'JSON':>5} {'F1':>6} {'Time':>6}")
    print(f"  {'─'*33}  {'─'*4}  {'─'*4} {'─'*4} {'─'*5} {'─'*4} {'─'*5} {'─'*5} {'─'*5} {'─'*5}")
    for r in ok:
        print(f"  {r['dataset']:<35} {r['rows']}×{r['cols']:>2} {r['apriori_count']:>4} {r['model_count']:>4} "
              f"{r['new_tokens']:>5} {'⚠️' if r['hit_limit'] else '  ':>4} "
              f"{'✅' if r['has_think'] else '❌':>5} {'✅' if r['parse_ok'] else '❌':>5} "
              f"{r['f1']:>5.0%} {r['time_s']:>5.0f}s")
    
    hit_limit = sum(1 for r in ok if r["hit_limit"])
    avg_f1 = sum(r["f1"] for r in ok) / n
    parse_rate = sum(1 for r in ok if r["parse_ok"]) / n
    think_rate = sum(1 for r in ok if r["has_think"]) / n
    avg_precision = sum(r["precision"] for r in ok) / n
    avg_recall = sum(r["recall"] for r in ok) / n
    total_hallucinated = sum(r.get("hallucinated_items", 0) for r in ok)
    
    # F1 only for datasets that parsed OK
    parsed_ok = [r for r in ok if r["parse_ok"]]
    avg_f1_parsed = sum(r["f1"] for r in parsed_ok) / len(parsed_ok) if parsed_ok else 0
    
    print(f"\n  AGGREGATES (all {n} datasets):")
    print(f"    Avg F1:            {avg_f1:.1%}")
    print(f"    Avg Precision:     {avg_precision:.1%}")
    print(f"    Avg Recall:        {avg_recall:.1%}")
    print(f"    Parse rate:        {parse_rate:.0%} ({sum(1 for r in ok if r['parse_ok'])}/{n})")
    print(f"    Think rate:        {think_rate:.0%}")
    print(f"    Hit token limit:   {hit_limit}/{n}")
    print(f"    Hallucinated items: {total_hallucinated} total")
    
    if parsed_ok:
        print(f"\n  AGGREGATES (parsed-only, {len(parsed_ok)} datasets):")
        print(f"    Avg F1:            {avg_f1_parsed:.1%}")
        avg_p_parsed = sum(r["precision"] for r in parsed_ok) / len(parsed_ok)
        avg_r_parsed = sum(r["recall"] for r in parsed_ok) / len(parsed_ok)
        print(f"    Avg Precision:     {avg_p_parsed:.1%}")
        print(f"    Avg Recall:        {avg_r_parsed:.1%}")
    
    print(f"\n    Avg tokens:        {sum(r['new_tokens'] for r in ok)/n:.0f}")
    print(f"    Avg time:          {sum(r['time_s'] for r in ok)/n:.0f}s")
    print(f"    Total time:        {total_time:.0f}s ({total_time/60:.1f}min)")
    
    # Breakdown: did hitting the limit cause parse failures?
    limit_parse_fail = sum(1 for r in ok if r["hit_limit"] and not r["parse_ok"])
    nolimit_parse_fail = sum(1 for r in ok if not r["hit_limit"] and not r["parse_ok"])
    limit_parse_ok = sum(1 for r in ok if r["hit_limit"] and r["parse_ok"])
    nolimit_parse_ok = sum(1 for r in ok if not r["hit_limit"] and r["parse_ok"])
    
    print(f"\n  TOKEN LIMIT vs JSON PARSE:")
    print(f"    Under limit + JSON OK:     {nolimit_parse_ok}")
    print(f"    Under limit + JSON FAIL:   {nolimit_parse_fail}")
    print(f"    HIT limit + JSON OK:       {limit_parse_ok}")
    print(f"    HIT limit + JSON FAIL:     {limit_parse_fail}")
    
    # F1 by complexity bucket
    small = [r for r in ok if r["rows"] * r["cols"] < 60]
    medium = [r for r in ok if 60 <= r["rows"] * r["cols"] < 120]
    large = [r for r in ok if r["rows"] * r["cols"] >= 120]
    
    print(f"\n  F1 BY DATASET COMPLEXITY:")
    for label, bucket in [("Small (<60 cells)", small), ("Medium (60-120)", medium), ("Large (120+)", large)]:
        if bucket:
            bf1 = sum(r["f1"] for r in bucket) / len(bucket)
            bparse = sum(1 for r in bucket if r["parse_ok"]) / len(bucket)
            blimit = sum(1 for r in bucket if r["hit_limit"]) / len(bucket)
            print(f"    {label:.<25} F1={bf1:.0%}  Parse={bparse:.0%}  Limit={blimit:.0%}  n={len(bucket)}")
    
    # vs previous run comparison
    print(f"\n  COMPARISON vs PREVIOUS RUN (no repetition penalty):")
    print(f"    Previous: 2/15 completed, Avg F1=7.5%, Parse=13%")
    print(f"    Current:  {sum(1 for r in ok if not r['hit_limit'])}/{n} completed, Avg F1={avg_f1:.1%}, Parse={parse_rate:.0%}")

# Save summary JSON
summary_path = os.path.join(OUTPUT_DIR, "summary.json")
with open(summary_path, "w") as f:
    json.dump({
        "config": {
            "repetition_penalty": REPETITION_PENALTY,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "num_datasets": NUM_DATASETS,
            "adapter_path": ADAPTER_PATH,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "aggregates": {
            "total": n,
            "avg_f1": round(avg_f1, 4) if ok else 0,
            "avg_precision": round(avg_precision, 4) if ok else 0,
            "avg_recall": round(avg_recall, 4) if ok else 0,
            "parse_rate": round(parse_rate, 4) if ok else 0,
            "think_rate": round(think_rate, 4) if ok else 0,
            "hit_limit_count": hit_limit if ok else 0,
            "total_hallucinated_items": total_hallucinated if ok else 0,
        },
        "results": results,
    }, f, indent=2)

print(f"\n💾 Results saved to {OUTPUT_DIR}/")
print(f"   {len(ok)} .txt files with full model output + ground truth")
print(f"   summary.json with config + aggregates + per-dataset metrics")
print(f"\n{'='*70}")
print(f"  DONE — compare F1 with previous run to see if repetition fix helped")
print(f"{'='*70}")
