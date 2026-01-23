# ==========================================
# EVALUÁCIA FINE-TUNED ITEMSET EXTRACTION MODELU
# ==========================================
# Skopíruj každú sekciu do samostatnej bunky v Google Colab
# PRED SPUSTENÍM: Runtime → Change runtime type → T4 GPU

# ==========================================
# CELL 1: Inštalácia
# ==========================================
# !pip install -q torch transformers accelerate peft pandas bitsandbytes

# ==========================================
# CELL 2: GPU Check + Imports
# ==========================================
import torch
import gc
import json
import re
import csv
import itertools
import time
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

print(f"🔥 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   Memory: {mem:.1f} GB")
else:
    raise RuntimeError("❌ GPU nie je zapnuté! Runtime → Change runtime type → T4 GPU")

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ==========================================
# CELL 3: Upload CSV súborov
# ==========================================
from google.colab import files

os.makedirs("eval_datasets", exist_ok=True)
print("📁 Nahraj eval CSV súbory:")
uploaded = files.upload()

for filename in uploaded.keys():
    if filename.endswith('.csv'):
        os.rename(filename, f"eval_datasets/{filename}")
        print(f"   ✓ {filename}")

print(f"\n✅ Nahratých {len(os.listdir('eval_datasets'))} datasetov")

# ==========================================
# CELL 4: Načítanie modelu s 4-bit quantizáciou
# ==========================================
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_NAME = "OliverSlivka/qwen2.5-3b-itemset-extractor"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"📥 Loading model with 4-bit quantization...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print(f"🔗 Loading adapter: {ADAPTER_NAME}")
model = PeftModel.from_pretrained(base_model, ADAPTER_NAME)
model.eval()

print(f"✅ Model loaded! GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ==========================================
# CELL 5: Konfigurácia + SYSTEM PROMPT
# ==========================================
MIN_SUPPORT = 3
MAX_SIZE = 3
MAX_TRANSACTIONS = 15  # Smaller chunks for 3B model
MAX_ITEMS_PER_ROW = 8  # Limit items per row
MAX_NEW_TOKENS = 2048  # Increased for full JSON output
LIMIT_DATASETS = 5     # Test 5 datasets

# System prompt - skrátená verzia z tréningových dát
SYSTEM_PROMPT = """You are a frequent itemset extractor. Your output must be a single JSON array of objects.

Each object must have:
- "itemset": array of lowercase strings (sorted alphabetically)
- "count": integer (number of rows containing this itemset)
- "rows": array of row references like ["Row 1", "Row 2", ...]

Rules:
- Only include itemsets with count >= minimum support
- Output ONLY the JSON array, no other text
- If no itemsets qualify, output []

Example output:
[{"itemset": ["bread", "milk"], "count": 3, "rows": ["Row 1", "Row 2", "Row 5"]}]"""

print(f"⚙️ Config: min_support={MIN_SUPPORT}, max_trans={MAX_TRANSACTIONS}, max_items={MAX_ITEMS_PER_ROW}")
print(f"   Testing {LIMIT_DATASETS} datasets with system prompt")

# ==========================================
# CELL 6: Pomocné funkcie
# ==========================================
def load_transactions_csv(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter
    except:
        sep = ','
    
    df = pd.read_csv(path, sep=sep, low_memory=False)
    transactions = []
    for _, row in df.iterrows():
        items = [str(v).strip() for v in row.tolist() 
                 if pd.notna(v) and str(v).strip() not in {'nan', 'none', ''}]
        if items:
            transactions.append(items)
    return transactions


def extract_itemsets_chunked(transactions, min_support):
    limited = [row[:MAX_ITEMS_PER_ROW] for row in transactions]
    all_results = {}
    
    for start in range(0, len(limited), MAX_TRANSACTIONS):
        end = min(start + MAX_TRANSACTIONS, len(limited))
        chunk = limited[start:end]
        
        # Formát riadkov
        rows_text = "\n".join(
            f"Row {start+i+1}: {', '.join(row)}" for i, row in enumerate(chunk)
        )
        
        # User prompt - jednoduchý a jasný
        user_prompt = f"""Transactions:
{rows_text}

Find all frequent itemsets with minimum support count = {min_support}."""

        # S SYSTEM PROMPTOM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.1, 
                do_sample=True, 
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Zabraň opakovaniu
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # DEBUG: Print first chunk's response
        if start == 0:
            print(f"\n   [DEBUG] Raw response (first 800 chars):\n   {response[:800]}\n")
        
        # VYLEPŠENÝ JSON PARSING
        parsed = []
        try:
            # Skús priamo
            parsed = json.loads(response.strip())
        except json.JSONDecodeError:
            # Nájdi JSON array - od prvého [ po posledný ]
            first_bracket = response.find('[')
            last_bracket = response.rfind(']')
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                json_str = response[first_bracket:last_bracket+1]
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    # Skús opraviť orezaný JSON
                    try:
                        # Pridaj chýbajúce zátvorky
                        fixed = json_str.rstrip(',') + ']'
                        parsed = json.loads(fixed)
                    except:
                        print(f"   [WARN] Could not parse JSON from response")
            else:
                print(f"   [WARN] No JSON array found in response")
        
        for rec in parsed:
            if not isinstance(rec, dict) or 'itemset' not in rec:
                continue
            itemset = rec.get('itemset', [])
            if not itemset:
                continue
            canon = tuple(sorted(str(x).strip().lower() for x in itemset))
            # Model returns "rows" with integers, e.g. [1, 2, 3]
            rows = rec.get('rows', rec.get('evidence_rows', []))
            norm_rows = set()
            for r in rows:
                try:
                    if isinstance(r, int):
                        norm_rows.add(f"Row {r}")
                    elif isinstance(r, str) and r.startswith("Row "):
                        norm_rows.add(r)
                    else:
                        norm_rows.add(f"Row {int(str(r).split()[-1])}")
                except:
                    pass
            if canon not in all_results:
                all_results[canon] = set()
            all_results[canon].update(norm_rows)
        
        del inputs, outputs
        clear_gpu_memory()
    
    return [{'itemset': list(k), 'count': len(v), 'rows': sorted(v)} 
            for k, v in all_results.items() if len(v) >= min_support]


def apriori_itemsets(transactions, min_support, max_size):
    if not transactions:
        return []
    
    limited = [row[:MAX_ITEMS_PER_ROW] for row in transactions]
    row_labels = [f"Row {i+1}" for i in range(len(limited))]
    counts = {}
    
    for idx, trans in enumerate(limited):
        seen = set()
        for item in trans:
            item = str(item).strip().lower()
            if item and item not in seen:
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
        for idx, trans in enumerate(limited):
            tset = set(str(x).strip().lower() for x in trans)
            for cand in candidates:
                if set(cand).issubset(tset):
                    cand_counts[cand]['count'] += 1
                    cand_counts[cand]['rows'].append(row_labels[idx])
        
        current = prune(cand_counts)
        if current:
            freq_levels.append(current)
        k += 1
    
    return [{'itemset': list(k), 'count': v['count'], 'rows': v['rows']} 
            for level in freq_levels for k, v in level.items()]


def compute_metrics(apriori_sets, model_sets):
    def canon(s):
        return tuple(sorted(str(x).strip().lower() for x in s['itemset']))
    
    apr = {canon(s) for s in apriori_sets}
    mod = {canon(s) for s in model_sets}
    
    tp = apr & mod
    fp = mod - apr
    fn = apr - mod
    
    precision = len(tp) / len(mod) if mod else 0
    recall = len(tp) / len(apr) if apr else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    jaccard = len(tp) / len(apr | mod) if (apr | mod) else 1
    
    return {
        "apriori": len(apr), "model": len(mod),
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "jaccard": round(jaccard, 4),
        "exact": apr == mod
    }

print("✅ Functions defined")

# ==========================================
# CELL 7: Spustenie evaluácie
# ==========================================
data_files = sorted([f"eval_datasets/{f}" for f in os.listdir("eval_datasets") if f.endswith('.csv')])
if LIMIT_DATASETS:
    data_files = data_files[:LIMIT_DATASETS]

print(f"📊 Evaluating {len(data_files)} datasets...")
print("=" * 60)

all_results = []
total_start = time.time()

for i, path in enumerate(data_files, 1):
    name = Path(path).name
    print(f"\n[{i}/{len(data_files)}] {name}")
    
    try:
        trans = load_transactions_csv(path)
        print(f"   Loaded: {len(trans)} transactions")
        
        apr_start = time.time()
        apr_sets = apriori_itemsets(trans, MIN_SUPPORT, MAX_SIZE)
        apr_time = time.time() - apr_start
        print(f"   Apriori: {len(apr_sets)} itemsets ({apr_time*1000:.0f}ms)")
        
        clear_gpu_memory()
        
        mod_start = time.time()
        mod_sets = extract_itemsets_chunked(trans, MIN_SUPPORT)
        mod_time = time.time() - mod_start
        print(f"   Model: {len(mod_sets)} itemsets ({mod_time*1000:.0f}ms)")
        
        m = compute_metrics(apr_sets, mod_sets)
        status = "✅" if m['exact'] else "⚠️"
        print(f"   {status} P={m['precision']:.1%} R={m['recall']:.1%} F1={m['f1']:.1%} | TP={m['tp']} FP={m['fp']} FN={m['fn']}")
        
        all_results.append({"dataset": name, "trans": len(trans), **m})
    except Exception as e:
        print(f"   ❌ Error: {e}")
        all_results.append({"dataset": name, "error": str(e)})
    
    clear_gpu_memory()

total_time = time.time() - total_start
print(f"\n{'='*60}\n⏱️ Total: {total_time:.1f}s")

# ==========================================
# CELL 8: Súhrn výsledkov
# ==========================================
successful = [r for r in all_results if 'precision' in r]

if successful:
    avg_p = sum(r['precision'] for r in successful) / len(successful)
    avg_r = sum(r['recall'] for r in successful) / len(successful)
    avg_f1 = sum(r['f1'] for r in successful) / len(successful)
    exact = sum(1 for r in successful if r['exact'])
    
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Datasets: {len(successful)}/{len(data_files)}")
    print(f"🎯 Avg Precision: {avg_p:.2%}")
    print(f"🎯 Avg Recall:    {avg_r:.2%}")
    print(f"🎯 Avg F1:        {avg_f1:.2%}")
    print(f"✅ Exact Matches: {exact}/{len(successful)} ({exact/len(successful)*100:.1f}%)")
    print("=" * 60)
    
    # Display table
    df = pd.DataFrame(successful)
    display(df)

# ==========================================
# CELL 9: Uloženie a stiahnutie
# ==========================================
if successful:
    with open('eval_results.json', 'w') as f:
        json.dump({"results": all_results, "summary": {
            "avg_precision": avg_p, "avg_recall": avg_r, "avg_f1": avg_f1,
            "exact_match_rate": exact/len(successful)
        }}, f, indent=2)
    files.download('eval_results.json')
    print("✅ Downloaded eval_results.json")
