#!/usr/bin/env python3
"""
Quick eval runner for the HF SFT model.
Runs inference on N evaluation datasets and prints F1 / parse / hallucination stats.

Usage (TLJH server):
    # Optional: set HF token if model is gated
    export HF_TOKEN="hf_..."
    python eval_verify_sft.py --count 20 --exclude-training

Flags:
    --count N          datasets to evaluate (default 20)
    --exclude-training skip the 272 datasets used in v3 SFT training
    --two-phase        use two-phase generation (think @ 0.3, JSON @ 0.05)
    --output-dir DIR   where to save results (default: eval_verify_results)
"""
from __future__ import annotations

import argparse, itertools, json, os, random, re, sys, time
from pathlib import Path

# ── Project root detection ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

# ── System prompt (must match training EXACTLY) ───────────────────────────────
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


# ── CSV loading ───────────────────────────────────────────────────────────────
def load_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    transactions = []
    lines = []
    for idx, row in df.iterrows():
        items = [str(v).strip() for v in row.values if pd.notna(v) and str(v).strip()]
        transactions.append(items)
        lines.append(f"Row {idx+1}: {', '.join(items)}")
    return transactions, "\n".join(lines), len(df), len(df.columns)


# ── Apriori ground truth ──────────────────────────────────────────────────────
def apriori(transactions, min_support=3, max_size=3):
    if not transactions:
        return []
    row_labels = [f"Row {i+1}" for i in range(len(transactions))]
    counts = {}

    # Level 1
    for idx, trans in enumerate(transactions):
        seen = set()
        for item in trans:
            item_norm = item.strip().lower()
            if not item_norm or item_norm in seen:
                continue
            seen.add(item_norm)
            k = (item_norm,)
            counts.setdefault(k, {"count": 0, "rows": []})
            counts[k]["count"] += 1
            counts[k]["rows"].append(row_labels[idx])

    def prune(d):
        return {k: v for k, v in d.items() if v["count"] >= min_support}

    freq = [prune(counts)]
    if not freq[0]:
        return []

    k = 2
    while k <= max_size and freq[-1]:
        prev_keys = sorted(freq[-1].keys())
        candidates = set()
        for i in range(len(prev_keys)):
            for j in range(i+1, len(prev_keys)):
                a, b = prev_keys[i], prev_keys[j]
                if a[:k-2] == b[:k-2]:
                    merged = tuple(sorted(set(a) | set(b)))
                    if len(merged) == k:
                        if all(tuple(sorted(sub)) in freq[-1]
                               for sub in itertools.combinations(merged, k-1)):
                            candidates.add(merged)
        if not candidates:
            break
        cand_counts = {c: {"count": 0, "rows": []} for c in candidates}
        for idx, trans in enumerate(transactions):
            tset = {x.strip().lower() for x in trans}
            for c in candidates:
                if set(c).issubset(tset):
                    cand_counts[c]["count"] += 1
                    cand_counts[c]["rows"].append(row_labels[idx])
        freq.append(prune(cand_counts))
        k += 1

    out = []
    for level in freq:
        for itemset, info in level.items():
            out.append({"itemset": list(itemset), "count": info["count"], "rows": info["rows"]})
    return out


# ── Model loading ─────────────────────────────────────────────────────────────
_MODEL = None
_TOKENIZER = None

def load_model(model_path: str, hf_token: str = ""):
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"📥 Loading model: {model_path}")
    _TOKENIZER = AutoTokenizer.from_pretrained(model_path, token=hf_token or None)

    if torch.cuda.is_available():
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token or None,
        )
        print(f"✅ Loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            token=hf_token or None,
        )
        print("⚠️  Loaded on CPU (slow)")

    _MODEL.eval()
    return _MODEL, _TOKENIZER


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(csv_text: str, min_support: int, max_new_tokens: int = 2048,
                  temperature: float = 0.3, two_phase: bool = False):
    import torch
    model, tokenizer = _MODEL, _TOKENIZER

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Find all frequent itemsets with minimum support count = {min_support} "
            f"in this dataset:\n\n{csv_text}"
        )},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)

    if two_phase:
        return _generate_two_phase(model, tokenizer, inputs, temperature, max_new_tokens)
    else:
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.90,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def _generate_two_phase(model, tokenizer, inputs, temperature, max_new_tokens):
    import torch
    device = inputs.device

    # Phase 1: think
    with torch.no_grad():
        think_out = model.generate(
            input_ids=inputs,
            max_new_tokens=min(max_new_tokens, 2000),
            temperature=temperature,
            top_k=50,
            top_p=0.90,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    think_text = tokenizer.decode(think_out[0][inputs.shape[1]:], skip_special_tokens=True)

    if "</think>" not in think_text:
        think_text += "\n</think>\n"

    # Phase 2: JSON
    full_text = tokenizer.decode(think_out[0], skip_special_tokens=False)
    if not full_text.rstrip().endswith("</think>"):
        full_text = full_text.rstrip() + "\n</think>\n"
    json_prompt = full_text + "["
    json_ids = tokenizer(json_prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        json_out = model.generate(
            input_ids=json_ids,
            max_new_tokens=min(max_new_tokens, 1500),
            temperature=0.05,
            top_k=20,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    json_text = tokenizer.decode(json_out[0][json_ids.shape[1]:], skip_special_tokens=True)

    return think_text + "[" + json_text


# ── JSON extraction ───────────────────────────────────────────────────────────
def extract_json(raw: str):
    """Extract and parse JSON array from model output. Returns (items, parse_ok)."""
    has_think = "</think>" in raw
    json_text = raw
    if has_think:
        parts = raw.split("</think>", 1)
        json_text = parts[1].strip() if len(parts) > 1 else ""

    parsed = []
    for text_source in [json_text, raw]:
        try:
            parsed = json.loads(text_source)
            if isinstance(parsed, list):
                return parsed, True
        except json.JSONDecodeError:
            pass
        # Try regex extraction
        m = re.search(r"\[.*\]", text_source, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, list):
                    return parsed, True
            except json.JSONDecodeError:
                pass
    return [], False


def normalize_itemsets(raw_items):
    """Normalize model output to canonical form."""
    items = []
    for rec in raw_items:
        if not isinstance(rec, dict):
            continue
        itemset = rec.get("itemset", [])
        if not isinstance(itemset, list) or not itemset:
            continue
        count = rec.get("count", 0)
        rows = rec.get("rows", [])
        norm_set = sorted(str(x).strip().lower() for x in itemset)
        norm_rows = []
        for r in rows:
            if isinstance(r, int):
                norm_rows.append(f"Row {r}")
            elif isinstance(r, str) and re.match(r"Row \d+", r):
                norm_rows.append(r)
            else:
                try:
                    norm_rows.append(f"Row {int(r)}")
                except (ValueError, TypeError):
                    pass
        items.append({
            "itemset": norm_set,
            "count": count if isinstance(count, int) else 0,
            "rows": sorted(set(norm_rows)),
        })
    return items


# ── Training dataset exclusion ────────────────────────────────────────────────
def get_training_datasets(root: Path) -> set:
    """Returns set of filenames used in v3 SFT training."""
    files = set()
    for json_path in ["data/sft_cot_v3.json", "data/dpo_real_v2.json"]:
        p = root / json_path
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
            for entry in data:
                ds_id = entry.get("dataset_id", "")
                filename = ds_id.split(":")[0] if ":" in ds_id else ds_id
                if filename:
                    files.add(filename)
        except Exception:
            pass
    return files


# ── Metrics ───────────────────────────────────────────────────────────────────
def canon(itemset):
    return frozenset(str(x).strip().lower() for x in itemset)


def compute_metrics(apriori_sets, model_sets, all_csv_items):
    apr_c = {canon(s["itemset"]) for s in apriori_sets}
    mod_c = {canon(s["itemset"]) for s in model_sets}

    tp = apr_c & mod_c
    fp = mod_c - apr_c
    fn = apr_c - mod_c

    precision = len(tp) / len(mod_c) if mod_c else 0.0
    recall = len(tp) / len(apr_c) if apr_c else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Count accuracy
    apr_map = {canon(s["itemset"]): s["count"] for s in apriori_sets}
    count_ok = count_total = 0
    for s in model_sets:
        c = canon(s["itemset"])
        if c in apr_map:
            count_total += 1
            if abs(s["count"] - apr_map[c]) <= 1:
                count_ok += 1
    count_acc = count_ok / count_total if count_total else 0.0

    # Hallucination
    hal = sum(1 for s in model_sets
              if any(item not in all_csv_items for item in canon(s["itemset"])))
    hal_rate = hal / len(model_sets) if model_sets else 0.0

    return {
        "apriori_count": len(apr_c),
        "model_count": len(mod_c),
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": f1 == 1.0,
        "count_accuracy": round(count_acc, 4),
        "hallucination_rate": round(hal_rate, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Verify SFT model F1 score")
    parser.add_argument("--model", default="OliverSlivka/qwen2.5-7b-itemset-extractor")
    parser.add_argument("--data-dir", default="data/datasets_v2")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--two-phase", action="store_true")
    parser.add_argument("--exclude-training", action="store_true")
    parser.add_argument("--output-dir", default="eval_verify_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # HF token from env
    hf_token = os.environ.get("HF_TOKEN", "")

    # Locate datasets
    data_dir = SCRIPT_DIR / args.data_dir
    if not data_dir.exists():
        print(f"❌ data dir not found: {data_dir}")
        sys.exit(1)

    all_csvs = sorted(data_dir.glob("*.csv"))
    print(f"📁 Found {len(all_csvs)} CSV datasets")

    # Exclude training
    if args.exclude_training:
        training = get_training_datasets(SCRIPT_DIR)
        before = len(all_csvs)
        all_csvs = [f for f in all_csvs if f.name not in training]
        print(f"🔒 Excluded {before - len(all_csvs)} training datasets ({len(all_csvs)} remaining)")

    # Sample
    random.seed(args.seed)
    if len(all_csvs) > args.count:
        selected = random.sample(all_csvs, args.count)
    else:
        selected = all_csvs
    print(f"🎯 Evaluating {len(selected)} datasets\n")

    # Load model once
    load_model(args.model, hf_token)

    # Output dir
    out_dir = SCRIPT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run
    results = []
    total_start = time.perf_counter()

    for i, csv_path in enumerate(selected, 1):
        dname = csv_path.name
        print(f"[{i}/{len(selected)}] {dname}")

        try:
            transactions, csv_text, n_rows, n_cols = load_csv(str(csv_path))
            print(f"   {n_rows}×{n_cols}, ", end="", flush=True)

            all_items = set()
            for t in transactions:
                for item in t:
                    all_items.add(item.strip().lower())

            # Apriori
            t0 = time.perf_counter()
            apr_sets = apriori(transactions, args.min_support, args.max_size)
            print(f"Apriori: {len(apr_sets)} itemsets, ", end="", flush=True)

            # Model
            t1 = time.perf_counter()
            raw = run_inference(csv_text, args.min_support, args.max_new_tokens,
                                args.temperature, args.two_phase)
            model_time = time.perf_counter() - t1

            parsed, parse_ok = extract_json(raw)
            model_sets = normalize_itemsets(parsed)
            print(f"Model: {len(model_sets)} itemsets ({model_time:.1f}s), ", end="")

            m = compute_metrics(apr_sets, model_sets, all_items)
            print(f"F1={m['f1']:.1%} {'✅' if parse_ok else '❌parse'}")

            results.append({
                "dataset": dname,
                "n_rows": n_rows, "n_cols": n_cols,
                "apriori_count": len(apr_sets),
                "model_count": len(model_sets),
                "model_time_s": round(model_time, 2),
                "parse_ok": parse_ok,
                "metrics": m,
            })

            # Save raw output
            raw_out = out_dir / f"raw_{dname.replace('.csv','')}.txt"
            raw_out.write_text(raw, encoding="utf-8")

        except Exception as e:
            print(f"❌ {e}")
            results.append({"dataset": dname, "error": str(e)})

    total_time = time.perf_counter() - total_start

    # ── Aggregate ──────────────────────────────────────────────────────────
    ok = [r for r in results if "metrics" in r]
    if not ok:
        print("\n❌ No successful evaluations")
        sys.exit(1)

    n = len(ok)
    summary = {
        "model": args.model,
        "datasets_evaluated": n,
        "datasets_attempted": len(selected),
        "avg_precision": sum(r["metrics"]["precision"] for r in ok) / n,
        "avg_recall": sum(r["metrics"]["recall"] for r in ok) / n,
        "avg_f1": sum(r["metrics"]["f1"] for r in ok) / n,
        "exact_matches": sum(1 for r in ok if r["metrics"]["exact_match"]),
        "exact_match_rate": sum(1 for r in ok if r["metrics"]["exact_match"]) / n,
        "parse_rate": sum(1 for r in ok if r["parse_ok"]) / n,
        "avg_hallucination": sum(r["metrics"]["hallucination_rate"] for r in ok) / n,
        "avg_count_accuracy": sum(r["metrics"]["count_accuracy"] for r in ok) / n,
        "avg_time_s": sum(r["model_time_s"] for r in ok) / n,
        "total_time_s": round(total_time, 2),
    }
    for k in summary:
        if isinstance(summary[k], float):
            summary[k] = round(summary[k], 4)

    # ── Print ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Datasets:      {summary['datasets_evaluated']}")
    print(f"  Avg F1:        {summary['avg_f1']:.2%}")
    print(f"  Avg Precision: {summary['avg_precision']:.2%}")
    print(f"  Avg Recall:    {summary['avg_recall']:.2%}")
    print(f"  Exact Match:   {summary['exact_matches']}/{n} ({summary['exact_match_rate']:.1%})")
    print(f"  Parse Rate:    {summary['parse_rate']:.1%}")
    print(f"  Hallucination: {summary['avg_hallucination']:.2%}")
    print(f"  Count Acc:     {summary['avg_count_accuracy']:.2%}")
    print(f"  Avg Time:      {summary['avg_time_s']:.1f}s/dataset")
    print(f"  Total Time:    {summary['total_time_s']:.0f}s")

    # ── Per-dataset table ──────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print(f"{'Dataset':<40} {'Size':>7} {'Apriori':>7} {'Model':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'P%':>5} {'R%':>5} {'F1%':>5}")
    print("─" * 80)
    for r in ok:
        m = r["metrics"]
        print(f"{r['dataset']:<40} {f'{r[\"n_rows\"]}×{r[\"n_cols\"]}':>7} "
              f"{r['apriori_count']:>7} {r['model_count']:>7} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
              f"{m['precision']*100:>5.0f} {m['recall']*100:>5.0f} {m['f1']*100:>5.0f}")

    # ── Compare to known result ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON TO EXPECTED RESULT")
    print("=" * 60)
    print(f"  Expected:   avg F1 ≈ 0.13  (Iteration 4 SFT, primary_v3)")
    print(f"  Observed:   avg F1 ≈ {summary['avg_f1']:.2f}")
    if 0.10 <= summary["avg_f1"] <= 0.16:
        print("  ✅ F1 in expected range — this IS the correct SFT model")
    elif summary["avg_f1"] > 0.16:
        print(f"  ⚠️  F1 ({summary['avg_f1']:.2f}) is HIGHER than expected (0.13)")
        print("      This may be the DPO model or evaluation changed")
    else:
        print(f"  ⚠️  F1 ({summary['avg_f1']:.2f}) is LOWER than expected (0.13)")
        print("      Verify settings (temperature, prompt, etc.)")
    print("=" * 60)

    # ── Save ────────────────────────────────────────────────────────────────
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "per_dataset.json").write_text(
        json.dumps(ok, indent=2, ensure_ascii=False))
    print(f"\n📁 Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
