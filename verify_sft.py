#!/usr/bin/env python3
"""Minimal SFT verification: load model, run on 10 datasets, print avg F1."""
import os, sys, json, time, random

# ── Fix TLJH GPU visibility: force single GPU before torch import ─────────────
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL = "OliverSlivka/qwen2.5-7b-itemset-extractor"
DATA_DIR = os.path.expanduser("~/datasets_v2")
COUNT = 10
MIN_SUPPORT = 3
MAX_NEW_TOKENS = 512
SEED = 42

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

print(f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "GPU: NONE")
print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map={"": 0}, token=HF_TOKEN)
model.eval()
print(f"Loaded. Vocab: {len(tok)}")

# ── Apriori ───────────────────────────────────────────────────────────────────
def apriori(transactions, min_support=3, max_size=3):
    import itertools
    if not transactions: return []
    rl = [f"Row {i+1}" for i in range(len(transactions))]
    counts = {}
    for idx, trans in enumerate(transactions):
        seen = set()
        for item in trans:
            n = item.strip().lower()
            if not n or n in seen: continue
            seen.add(n)
            counts.setdefault((n,), {"count": 0, "rows": []})
            counts[(n,)]["count"] += 1
            counts[(n,)]["rows"].append(rl[idx])

    def prune(d): return {k: v for k, v in d.items() if v["count"] >= min_support}
    freq = [prune(counts)]
    if not freq[0]: return []

    k = 2
    while k <= max_size and freq[-1]:
        prev = sorted(freq[-1].keys())
        cands = set()
        for i in range(len(prev)):
            for j in range(i+1, len(prev)):
                a, b = prev[i], prev[j]
                if a[:k-2] == b[:k-2]:
                    m = tuple(sorted(set(a) | set(b)))
                    if len(m) == k and all(tuple(sorted(sub)) in freq[-1] for sub in itertools.combinations(m, k-1)):
                        cands.add(m)
        if not cands: break
        cc = {c: {"count": 0, "rows": []} for c in cands}
        for idx, trans in enumerate(transactions):
            ts = {x.strip().lower() for x in trans}
            for c in cands:
                if set(c).issubset(ts):
                    cc[c]["count"] += 1
                    cc[c]["rows"].append(rl[idx])
        freq.append(prune(cc))
        k += 1

    out = []
    for lvl in freq:
        for iset, info in lvl.items():
            out.append({"itemset": list(iset), "count": info["count"], "rows": info["rows"]})
    return out

# ── Datasets ──────────────────────────────────────────────────────────────────
import pandas as pd
csvs = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
random.seed(SEED)
random.shuffle(csvs)
csvs = csvs[:COUNT]

print(f"\nEvaluating {len(csvs)} datasets...\n")

f1s, times, parses, hals = [], [], [], []

for i, fn in enumerate(csvs, 1):
    path = os.path.join(DATA_DIR, fn)
    df = pd.read_csv(path)
    n_rows, n_cols = df.shape
    txns = []
    lines = []
    for idx, row in df.iterrows():
        items = [str(v).strip() for v in row.values if pd.notna(v) and str(v).strip()]
        txns.append(items)
        lines.append(f"Row {idx+1}: {', '.join(items)}")
    csv_text = "\n".join(lines)

    all_items = {item.strip().lower() for t in txns for item in t}
    apr = apriori(txns, MIN_SUPPORT)

    # Inference
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Find all frequent itemsets with minimum support count = {MIN_SUPPORT} in this dataset:\n\n{csv_text}"},
    ]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = getattr(enc, "input_ids", enc).to(model.device)
    am = getattr(enc, "attention_mask", None)
    if am is not None: am = am.to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=am,
                             max_new_tokens=MAX_NEW_TOKENS, temperature=0.3,
                             do_sample=True, top_k=50, top_p=0.90,
                             pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    elapsed = time.perf_counter() - t0
    raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    # Parse JSON
    model_sets = []
    parse_ok = False
    json_text = raw.split("</think>", 1)[-1].strip() if "</think>" in raw else raw
    for src in [json_text, raw]:
        try:
            import re
            parsed = json.loads(src)
            if isinstance(parsed, list):
                for rec in parsed:
                    if isinstance(rec, dict):
                        iset = rec.get("itemset", [])
                        if isinstance(iset, list) and iset:
                            model_sets.append({
                                "itemset": sorted(str(x).strip().lower() for x in iset),
                                "count": rec.get("count", 0),
                            })
                parse_ok = True
                break
        except (json.JSONDecodeError, ValueError):
            m = re.search(r"\[.*\]", src, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                    if isinstance(parsed, list):
                        for rec in parsed:
                            if isinstance(rec, dict):
                                iset = rec.get("itemset", [])
                                if isinstance(iset, list) and iset:
                                    model_sets.append({
                                        "itemset": sorted(str(x).strip().lower() for x in iset),
                                        "count": rec.get("count", 0),
                                    })
                        parse_ok = True
                        break
                except (json.JSONDecodeError, ValueError):
                    pass

    # Metrics
    apr_c = {frozenset(s["itemset"]) for s in apr}
    mod_c = {frozenset(s["itemset"]) for s in model_sets}
    tp = len(apr_c & mod_c)
    p = tp / len(mod_c) if mod_c else 0.0
    r = tp / len(apr_c) if apr_c else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    hal = sum(1 for s in model_sets if any(x not in all_items for x in s["itemset"]))
    hal_rate = hal / len(model_sets) if model_sets else 0.0

    f1s.append(f1)
    times.append(elapsed)
    parses.append(parse_ok)
    hals.append(hal_rate)

    print(f"[{i:2d}/{len(csvs)}] {fn:<45} {n_rows}x{n_cols} | A:{len(apr):>4} M:{len(mod_c):>4} | F1={f1:.1%} {'✅' if parse_ok else '❌parse'} | {elapsed:.0f}s")

print(f"\n{'='*60}")
print(f"AVG F1:        {sum(f1s)/len(f1s):.2%}")
print(f"PARSE RATE:    {sum(parses)/len(parses):.1%}")
print(f"HALLUCINATION: {sum(hals)/len(hals):.2%}")
print(f"AVG TIME:      {sum(times)/len(times):.0f}s")
print(f"TOTAL TIME:    {sum(times):.0f}s")
print(f"\nExpected: F1 ~ 0.13 (Iteration 4 SFT)")
print(f"Observed: F1 ~ {sum(f1s)/len(f1s):.2f}")
print(f"{'✅ CORRECT SFT MODEL' if 0.05 <= sum(f1s)/len(f1s) <= 0.25 else '⚠️ CHECK'}")
