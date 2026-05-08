#!/usr/bin/env python3
"""Evaluation script for fine-tuned Qwen2.5-7B itemset extraction model.

v3 (2026-03-09) — Updated with LLM Council inference fixes:
  - Temperature: 0.3 (was 0.1) — escape attractor loops
  - StoppingCriteria at </think> — prevent runaway generation
  - Dynamic token budget — dataset-specific max_new_tokens
  - Two-phase generation option — <think> at temp=0.3, JSON at temp=0.05
  - RepetitionDetector — catch looping patterns

Loads the model from HuggingFace Hub (adapter or merged), runs inference on
evaluation datasets using the EXACT same prompt format as training,
and computes 7 metrics vs Apriori ground truth.

Metrics computed:
  1. Precision   — of predicted itemsets, how many are correct
  2. Recall      — of correct itemsets, how many were found
  3. F1 Score    — harmonic mean of P and R
  4. Exact Match — % of datasets with perfect F1=1.0
  5. JSON Parse  — % of outputs that are valid JSON
  6. Hallucination Rate — % of predicted itemsets with items NOT in CSV
  7. Inference Time — seconds per dataset

Usage (on school TLJH server with GPU):
  python eval_finetuned_model.py \\
      --model OliverSlivka/qwen2.5-7b-itemset-extractor \\
      --data-dir data/datasets_v2 \\
      --count 20 --exclude-training

Usage (two-phase generation for best results):
  python eval_finetuned_model.py \\
      --model OliverSlivka/qwen2.5-7b-itemset-extractor \\
      --data-dir data/datasets_v2 --count 20 --two-phase

Usage (CPU fallback — slow):
  python eval_finetuned_model.py \\
      --model OliverSlivka/qwen2.5-7b-itemset-extractor \\
      --data-dir data/datasets_v2 --count 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.utils.diversity_metrics import compute_diversity_report

# ═════════════════════════════════════════════════════════════════════════════
# System prompt — MUST match training_utils.py EXACTLY
# ═════════════════════════════════════════════════════════════════════════════
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


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════
_model = None
_tokenizer = None


def load_model(model_path: str) -> tuple:
    """Load the merged 4-bit model. Tries Unsloth first, falls back to transformers."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    # Suppress TF/JAX (TLJH has Keras 3 which crashes transformers)
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_JAX", "0")

    import torch

    # ── SKIP Unsloth for merged models (merged HF models are not 4-bit LoRA) ──
    # ── Load via plain transformers ────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"📥 Loading model via transformers: {model_path}")
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"✅ Model loaded (transformers, 4-bit on GPU: {torch.cuda.get_device_name(0)})")
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        print("⚠️  Model loaded on CPU (will be slow)")

    _model.eval()
    return _model, _tokenizer


# ═════════════════════════════════════════════════════════════════════════════
# CSV loading (matches pipeline.py / training_utils.py format)
# ═════════════════════════════════════════════════════════════════════════════
def load_csv_as_rows(csv_path: str) -> tuple[list[list[str]], str, int, int]:
    """Load CSV and return (transactions, formatted_text, n_rows, n_cols).

    The formatted text matches exactly what training_utils.load_csv_as_prompt produces:
        Row 1: item1, item2, item3
        Row 2: ...
    """
    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape

    transactions = []
    rows_text_parts = []
    for idx, row in df.iterrows():
        items = [str(v).strip() for v in row.values if pd.notna(v) and str(v).strip()]
        transactions.append(items)
        rows_text_parts.append(f"Row {idx + 1}: {', '.join(items)}")

    return transactions, "\n".join(rows_text_parts), n_rows, n_cols


# ═════════════════════════════════════════════════════════════════════════════
# Apriori ground truth (deterministic, from pipeline.py)
# ═════════════════════════════════════════════════════════════════════════════
def apriori_frequent_itemsets(
    transactions: list[list[str]],
    min_support: int = 3,
    max_size: int = 3,
) -> list[dict]:
    """Apriori algorithm — returns ground truth itemsets."""
    if not transactions:
        return []

    row_labels = [f"Row {i + 1}" for i in range(len(transactions))]
    counts: dict[tuple, dict] = {}

    # Level 1: single items
    for idx, trans in enumerate(transactions):
        seen: set[str] = set()
        for item in trans:
            item_norm = str(item).strip().lower()
            if not item_norm or item_norm in seen:
                continue
            seen.add(item_norm)
            k = (item_norm,)
            if k not in counts:
                counts[k] = {"count": 0, "rows": []}
            counts[k]["count"] += 1
            counts[k]["rows"].append(row_labels[idx])

    def prune(d: dict) -> dict:
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
                if a[: k - 2] == b[: k - 2]:
                    merged = tuple(sorted(set(a) | set(b)))
                    if len(merged) == k:
                        if all(
                            tuple(sorted(sub)) in current
                            for sub in itertools.combinations(merged, k - 1)
                        ):
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
            out.append({
                "itemset": list(itemset),
                "count": info["count"],
                "rows": info["rows"],
            })
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Model inference — v3 with council-recommended fixes
# ═════════════════════════════════════════════════════════════════════════════

# Import inference utilities for StoppingCriteria, two-phase gen, JSON repair
# Support both `python src/evaluation/eval_finetuned_model.py` and `python -m` invocations
try:
    from src.evaluation.inference_utils import (
        ThinkStoppingCriteria,
        RepetitionDetector,
        calculate_dynamic_budget,
        count_unique_values,
        generate_two_phase,
        generate_with_guards,
        extract_and_repair_json,
        V3_INFERENCE_CONFIG,
    )
except ImportError:
    from inference_utils import (
        ThinkStoppingCriteria,
        RepetitionDetector,
        calculate_dynamic_budget,
        count_unique_values,
        generate_two_phase,
        generate_with_guards,
        extract_and_repair_json,
        V3_INFERENCE_CONFIG,
    )

# Try importing StoppingCriteriaList from transformers
try:
    from transformers import StoppingCriteriaList
except ImportError:
    try:
        from src.evaluation.inference_utils import StoppingCriteriaList
    except ImportError:
        from inference_utils import StoppingCriteriaList


def run_inference(
    csv_text: str,
    min_support: int,
    max_new_tokens: int = 2048,
    temperature: float = 0.3,
    two_phase: bool = False,
    n_rows: int = 0,
    n_cols: int = 0,
) -> dict:
    """Run model inference with v3 council fixes.

    Changes from v2:
      - temperature: 0.1 → 0.3 (escape attractor loops)
      - StoppingCriteria at </think> (prevent runaway generation)
      - RepetitionDetector (catch repeated line patterns)
      - Dynamic token budget from dataset dimensions
      - Optional two-phase generation (think at 0.3, JSON at 0.05)
    """
    import torch

    model, tokenizer = _model, _tokenizer
    assert model is not None, "Call load_model() first"

    # Dynamic token budget if dimensions provided
    if n_rows > 0 and n_cols > 0:
        n_unique = count_unique_values(csv_text)
        dynamic_budget = calculate_dynamic_budget(n_rows, n_cols, n_unique)
        max_new_tokens = min(max_new_tokens, dynamic_budget)

    # Build messages — exact same format as training
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Find all frequent itemsets with minimum support count = {min_support} "
                f"in this dataset:\n\n{csv_text}"
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    if two_phase:
        # Two-phase: <think> at temp=0.3, JSON at temp=0.05
        raw = generate_two_phase(
            model, tokenizer, inputs,
            think_temperature=temperature,
            json_temperature=0.05,
            think_max_tokens=min(max_new_tokens, 2000),
            json_max_tokens=min(max_new_tokens, 1500),
            top_k=50,
            top_p=0.90,
        )
    else:
        # Single-phase with StoppingCriteria guards
        raw = generate_with_guards(
            model, tokenizer, inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.90,
        )

    # ── Parse output using shared JSON repair ────────────────────────────
    has_think = "<think>" in raw and "</think>" in raw
    think_content = ""

    if has_think:
        parts = raw.split("</think>", 1)
        think_content = parts[0].split("<think>", 1)[-1].strip()

    parsed, parse_ok, _ = extract_and_repair_json(raw)

    # Normalize items
    items = []
    for rec in parsed:
        if not isinstance(rec, dict):
            continue
        itemset = rec.get("itemset", [])
        if not isinstance(itemset, list) or not itemset:
            continue
        count = rec.get("count", 0)
        rows = rec.get("rows", [])

        # Normalize
        norm_itemset = sorted(str(x).strip().lower() for x in itemset)
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
            "itemset": norm_itemset,
            "count": count if isinstance(count, int) else 0,
            "rows": sorted(set(norm_rows)),
        })

    return {
        "raw_output": raw,
        "parsed_items": items,
        "parse_ok": parse_ok,
        "has_think": has_think,
        "think_content": think_content,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Metrics computation
# ═════════════════════════════════════════════════════════════════════════════
def canon(itemset: list) -> frozenset:
    """Canonicalize an itemset to a frozenset of lowercase stripped strings."""
    return frozenset(str(x).strip().lower() for x in itemset)


def compute_metrics(
    apriori_sets: list[dict],
    model_sets: list[dict],
    all_csv_items: set[str],
    count_tolerance: int = 1,
) -> dict:
    """Compute all evaluation metrics for one dataset."""
    apr_canonical = {canon(s["itemset"]) for s in apriori_sets}
    mod_canonical = {canon(s["itemset"]) for s in model_sets}

    tp = apr_canonical & mod_canonical
    fp = mod_canonical - apr_canonical
    fn = apr_canonical - mod_canonical

    precision = len(tp) / len(mod_canonical) if mod_canonical else 0.0
    recall = len(tp) / len(apr_canonical) if apr_canonical else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Count accuracy — for matched itemsets, how many have correct count?
    apr_count_map = {}
    for s in apriori_sets:
        apr_count_map[canon(s["itemset"])] = s["count"]

    count_correct = count_total = 0
    for s in model_sets:
        c = canon(s["itemset"])
        if c in apr_count_map:
            count_total += 1
            if abs(s["count"] - apr_count_map[c]) <= count_tolerance:
                count_correct += 1

    count_accuracy = count_correct / count_total if count_total > 0 else 0.0

    # Hallucination — items not in the CSV
    hallucinated = 0
    for s in model_sets:
        for item in s["itemset"]:
            if item not in all_csv_items:
                hallucinated += 1
                break  # count per itemset, not per item

    hallucination_rate = hallucinated / len(model_sets) if model_sets else 0.0

    # Per-size breakdown
    def by_size(sets_list):
        result: dict[int, set] = {}
        for s in sets_list:
            size = len(s["itemset"])
            result.setdefault(size, set()).add(canon(s["itemset"]))
        return result

    apr_by_size = by_size(apriori_sets)
    mod_by_size = by_size(model_sets)
    size_breakdown = {}
    for size in sorted(set(apr_by_size) | set(mod_by_size)):
        a = apr_by_size.get(size, set())
        m = mod_by_size.get(size, set())
        s_tp = len(a & m)
        s_p = s_tp / len(m) if m else 0.0
        s_r = s_tp / len(a) if a else 0.0
        s_f1 = 2 * s_p * s_r / (s_p + s_r) if (s_p + s_r) > 0 else 0.0
        size_breakdown[f"size_{size}"] = {
            "apriori": len(a),
            "model": len(m),
            "tp": s_tp,
            "precision": round(s_p, 4),
            "recall": round(s_r, 4),
            "f1": round(s_f1, 4),
        }

    return {
        "apriori_count": len(apr_canonical),
        "model_count": len(mod_canonical),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": f1 == 1.0,
        "count_accuracy": round(count_accuracy, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "size_breakdown": size_breakdown,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Training dataset exclusion
# ═════════════════════════════════════════════════════════════════════════════
def get_training_dataset_filenames(project_root: Path) -> set[str]:
    """Extract dataset filenames used in SFT/DPO training data."""
    filenames: set[str] = set()

    for json_file in ["data/sft_cot_v2.json", "data/dpo_real_v2.json"]:
        path = project_root / json_file
        if not path.exists():
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for entry in data:
                ds_id = entry.get("dataset_id", "")
                # Format: "ds_0001_7x12_85aed5f8.csv:85aed5f80c90"
                filename = ds_id.split(":")[0] if ":" in ds_id else ds_id
                if filename:
                    filenames.add(filename)
        except Exception:
            pass

    return filenames


# ═════════════════════════════════════════════════════════════════════════════
# Single dataset evaluation
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_dataset(
    csv_path: str,
    min_support: int,
    max_size: int,
    max_new_tokens: int,
    two_phase: bool = False,
    temperature: float = 0.3,
) -> dict:
    """Evaluate model on one dataset. Returns full result dict."""
    # Load CSV
    transactions, csv_text, n_rows, n_cols = load_csv_as_rows(csv_path)
    print(f"   Loaded: {n_rows} rows × {n_cols} cols")

    # All items in the CSV (for hallucination check)
    all_items: set[str] = set()
    for trans in transactions:
        for item in trans:
            all_items.add(str(item).strip().lower())

    # Ground truth
    t0 = time.perf_counter()
    apriori_sets = apriori_frequent_itemsets(transactions, min_support, max_size)
    apriori_time = time.perf_counter() - t0
    print(f"   Apriori: {len(apriori_sets)} itemsets ({apriori_time * 1000:.0f}ms)")

    # Model inference (v3: temp=0.3, StoppingCriteria, dynamic budget)
    t0 = time.perf_counter()
    inference = run_inference(
        csv_text, min_support,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        two_phase=two_phase,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    model_time = time.perf_counter() - t0
    model_sets = inference["parsed_items"]
    print(
        f"   Model:   {len(model_sets)} itemsets ({model_time:.1f}s) "
        f"| JSON={'✅' if inference['parse_ok'] else '❌'} "
        f"| <think>={'✅' if inference['has_think'] else '❌'}"
    )

    # Metrics
    metrics = compute_metrics(apriori_sets, model_sets, all_items)

    return {
        "dataset": Path(csv_path).name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "min_support": min_support,
        "max_size": max_size,
        "apriori_count": len(apriori_sets),
        "model_count": len(model_sets),
        "apriori_time_s": round(apriori_time, 3),
        "model_time_s": round(model_time, 2),
        "parse_ok": inference["parse_ok"],
        "has_think": inference["has_think"],
        "metrics": metrics,
        "raw_output": inference["raw_output"],
        "think_content": inference["think_content"],
        "apriori_output": apriori_sets,
        "model_output": model_sets,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Report generation
# ═════════════════════════════════════════════════════════════════════════════
def generate_report(results: list[dict], summary: dict, output_dir: Path) -> None:
    """Generate markdown evaluation report."""
    lines = [
        "# Model Evaluation Report",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Model:** {summary['model']}",
        f"**Datasets evaluated:** {summary['successful']}/{summary['total']}",
        f"**Total time:** {summary['total_time_s']:.0f}s",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value | Target |",
        "|--------|-------|--------|",
        f"| **F1 Score** | **{summary['avg_f1']:.1%}** | ≥ 80% |",
        f"| Precision | {summary['avg_precision']:.1%} | — |",
        f"| Recall | {summary['avg_recall']:.1%} | — |",
        f"| Exact Match | {summary['exact_match_rate']:.1%} | ≥ 50% |",
        f"| JSON Parse Rate | {summary['parse_rate']:.1%} | ≥ 90% |",
        f"| Hallucination Rate | {summary['avg_hallucination']:.1%} | ≤ 5% |",
        f"| Think Rate | {summary['think_rate']:.1%} | — |",
        f"| Count Accuracy | {summary['avg_count_accuracy']:.1%} | — |",
        f"| Avg Inference Time | {summary['avg_time_s']:.1f}s | ≤ 60s |",
        "",
        "## Per-Dataset Results",
        "",
        "| Dataset | Rows×Cols | Apriori | Model | TP | FP | FN | P | R | F1 | JSON | Think | Time |",
        "|---------|-----------|---------|-------|----|----|----|---|---|----|----- |-------|------|",
    ]

    successful = [r for r in results if "metrics" in r]
    for r in successful:
        m = r["metrics"]
        lines.append(
            f"| {r['dataset']} | {r['n_rows']}×{r['n_cols']} | "
            f"{r['apriori_count']} | {r['model_count']} | "
            f"{m['tp']} | {m['fp']} | {m['fn']} | "
            f"{m['precision']:.0%} | {m['recall']:.0%} | {m['f1']:.0%} | "
            f"{'✅' if r['parse_ok'] else '❌'} | "
            f"{'✅' if r['has_think'] else '❌'} | "
            f"{r['model_time_s']:.0f}s |"
        )

    # Failure analysis
    failures = [r for r in successful if r["metrics"]["f1"] < 0.5]
    parse_failures = [r for r in results if not r.get("parse_ok", True)]
    hallucinating = [r for r in successful if r["metrics"]["hallucination_rate"] > 0.1]

    lines.extend(["", "## Failure Analysis", ""])

    if not failures and not parse_failures:
        lines.append("No significant failures detected. 🎉")
    else:
        if parse_failures:
            lines.append(f"### JSON Parse Failures ({len(parse_failures)})")
            for r in parse_failures:
                raw = r.get("raw_output", "")[:200]
                lines.append(f"- **{r['dataset']}**: `{raw}...`")
            lines.append("")

        if failures:
            lines.append(f"### Low F1 Datasets ({len(failures)})")
            for r in failures:
                m = r["metrics"]
                lines.append(
                    f"- **{r['dataset']}**: F1={m['f1']:.0%} "
                    f"(P={m['precision']:.0%} R={m['recall']:.0%}, "
                    f"TP={m['tp']} FP={m['fp']} FN={m['fn']})"
                )
            lines.append("")

        if hallucinating:
            lines.append(f"### Hallucinating Datasets ({len(hallucinating)})")
            for r in hallucinating:
                lines.append(
                    f"- **{r['dataset']}**: "
                    f"{r['metrics']['hallucination_rate']:.0%} hallucination rate"
                )
            lines.append("")

    # Size breakdown (aggregate across all datasets)
    all_size_metrics: dict[str, list] = {}
    for r in successful:
        for size_key, size_data in r["metrics"]["size_breakdown"].items():
            all_size_metrics.setdefault(size_key, []).append(size_data)

    if all_size_metrics:
        lines.extend(["## Performance by Itemset Size", ""])
        lines.append("| Size | Avg Apriori | Avg Model | Avg P | Avg R | Avg F1 |")
        lines.append("|------|------------|-----------|-------|-------|--------|")
        for size_key in sorted(all_size_metrics):
            entries = all_size_metrics[size_key]
            n = len(entries)
            avg_apr = sum(e["apriori"] for e in entries) / n
            avg_mod = sum(e["model"] for e in entries) / n
            avg_p = sum(e["precision"] for e in entries) / n
            avg_r = sum(e["recall"] for e in entries) / n
            avg_f1 = sum(e["f1"] for e in entries) / n
            lines.append(
                f"| {size_key.replace('size_', '')} | {avg_apr:.1f} | {avg_mod:.1f} | "
                f"{avg_p:.0%} | {avg_r:.0%} | {avg_f1:.0%} |"
            )
        lines.append("")

    report_path = output_dir / "evaluation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 Report saved → {report_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned itemset extraction model vs Apriori ground truth"
    )
    parser.add_argument(
        "--model",
        default="OliverSlivka/qwen2.5-7b-itemset-extractor",
        help="HuggingFace model repo or local path (default: OliverSlivka/qwen2.5-7b-itemset-extractor)",
    )
    parser.add_argument("--data", help="Single CSV dataset path")
    parser.add_argument("--data-dir", help="Directory of CSV datasets")
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of random datasets to evaluate from --data-dir (default: 10)",
    )
    parser.add_argument(
        "--min-support", type=int, default=3, help="Min support threshold (default: 3)"
    )
    parser.add_argument(
        "--max-size", type=int, default=3, help="Max itemset size for Apriori (default: 3)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max generation tokens (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3, council recommendation)",
    )
    parser.add_argument(
        "--two-phase",
        action="store_true",
        help="Use two-phase generation: <think> at temp=0.3, JSON at temp=0.05",
    )
    parser.add_argument(
        "--output-dir", default="eval_results", help="Output directory (default: eval_results)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for dataset selection"
    )
    parser.add_argument(
        "--exclude-training",
        action="store_true",
        help="Exclude datasets used in SFT/DPO training (fair evaluation)",
    )
    args = parser.parse_args()

    # ── Collect datasets ─────────────────────────────────────────────────
    if args.data:
        data_files = [args.data]
    elif args.data_dir:
        if not os.path.isdir(args.data_dir):
            print(f"❌ Directory not found: {args.data_dir}", file=sys.stderr)
            sys.exit(1)
        all_csvs = sorted([
            os.path.join(args.data_dir, f)
            for f in os.listdir(args.data_dir)
            if f.lower().endswith(".csv")
        ])

        # Exclude training datasets
        if args.exclude_training:
            project_root = Path(__file__).resolve().parents[2]
            training_names = get_training_dataset_filenames(project_root)
            before = len(all_csvs)
            all_csvs = [f for f in all_csvs if Path(f).name not in training_names]
            excluded = before - len(all_csvs)
            print(f"🔒 Excluded {excluded} training datasets ({len(all_csvs)} remaining)")

        # Random sample
        random.seed(args.seed)
        if len(all_csvs) > args.count:
            data_files = random.sample(all_csvs, args.count)
        else:
            data_files = all_csvs
    else:
        print("❌ Specify --data or --data-dir", file=sys.stderr)
        sys.exit(1)

    if not data_files:
        print("❌ No CSV files found", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Banner ───────────────────────────────────────────────────────────
    print("═" * 60)
    print("  ITEMSET EXTRACTION MODEL EVALUATION (v3)")
    print("═" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Datasets:    {len(data_files)}")
    print(f"  Min support: {args.min_support}")
    print(f"  Max size:    {args.max_size}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Two-phase:   {'Yes' if args.two_phase else 'No'}")
    print(f"  Output:      {output_dir}")
    print("═" * 60)

    # ── Load model once ──────────────────────────────────────────────────
    load_model(args.model)

    # ── Run evaluation ───────────────────────────────────────────────────
    all_results = []
    total_start = time.perf_counter()

    for i, csv_path in enumerate(data_files, 1):
        print(f"\n[{i}/{len(data_files)}] {Path(csv_path).name}")
        try:
            result = evaluate_dataset(
                csv_path, args.min_support, args.max_size, args.max_new_tokens,
                two_phase=args.two_phase, temperature=args.temperature,
            )
            all_results.append(result)

            m = result["metrics"]
            print(
                f"   → P={m['precision']:.0%} R={m['recall']:.0%} F1={m['f1']:.0%} "
                f"| halluc={m['hallucination_rate']:.0%}"
            )
        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_results.append({"dataset": Path(csv_path).name, "error": str(e)})

    total_time = time.perf_counter() - total_start

    # ── Aggregate ────────────────────────────────────────────────────────
    successful = [r for r in all_results if "metrics" in r]
    if not successful:
        print("\n❌ No successful evaluations")
        sys.exit(1)

    n = len(successful)
    summary = {
        "model": args.model,
        "total": len(data_files),
        "successful": n,
        "failed": len(all_results) - n,
        "avg_precision": sum(r["metrics"]["precision"] for r in successful) / n,
        "avg_recall": sum(r["metrics"]["recall"] for r in successful) / n,
        "avg_f1": sum(r["metrics"]["f1"] for r in successful) / n,
        "exact_match_count": sum(1 for r in successful if r["metrics"]["exact_match"]),
        "exact_match_rate": sum(1 for r in successful if r["metrics"]["exact_match"]) / n,
        "parse_rate": sum(1 for r in successful if r["parse_ok"]) / n,
        "think_rate": sum(1 for r in successful if r["has_think"]) / n,
        "avg_hallucination": sum(r["metrics"]["hallucination_rate"] for r in successful) / n,
        "avg_count_accuracy": sum(r["metrics"]["count_accuracy"] for r in successful) / n,
        "avg_time_s": sum(r["model_time_s"] for r in successful) / n,
        "total_time_s": round(total_time, 1),
        "min_support": args.min_support,
        "max_size": args.max_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_diversity": compute_diversity_report([r["raw_output"] for r in successful]),
    }

    # Round summary floats
    for k in summary:
        if isinstance(summary[k], float):
            summary[k] = round(summary[k], 4)

    # ── Save outputs ─────────────────────────────────────────────────────
    # Summary (lightweight — no raw outputs)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Full results (includes raw outputs for debugging)
    detail_results = []
    for r in all_results:
        detail = {k: v for k, v in r.items() if k != "raw_output"}
        # Keep raw output in a separate per-dataset file for debugging
        if "raw_output" in r:
            raw_file = output_dir / f"raw_{r['dataset'].replace('.csv', '')}.txt"
            raw_file.write_text(r["raw_output"], encoding="utf-8")
        detail_results.append(detail)

    (output_dir / "all_results.json").write_text(
        json.dumps(detail_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown report
    generate_report(all_results, summary, output_dir)

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  EVALUATION SUMMARY")
    print("═" * 60)
    print(f"  Datasets:          {n}/{len(data_files)} successful")
    print(f"  Avg Precision:     {summary['avg_precision']:.1%}")
    print(f"  Avg Recall:        {summary['avg_recall']:.1%}")
    print(f"  Avg F1 Score:      {summary['avg_f1']:.1%}")
    print(
        f"  Exact Match:       "
        f"{summary['exact_match_count']}/{n} ({summary['exact_match_rate']:.1%})"
    )
    print(f"  JSON Parse Rate:   {summary['parse_rate']:.1%}")
    print(f"  Think Rate:        {summary['think_rate']:.1%}")
    print(f"  Hallucination:     {summary['avg_hallucination']:.1%}")
    print(f"  Count Accuracy:    {summary['avg_count_accuracy']:.1%}")
    print(f"  Avg Inference:     {summary['avg_time_s']:.1f}s per dataset")
    print(f"  Total Time:        {total_time:.0f}s")
    print(f"\n  📁 Results → {output_dir}/")

    # ── Pass/fail verdict ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    passed = (
        summary["avg_f1"] >= 0.80
        and summary["parse_rate"] >= 0.90
        and summary["avg_hallucination"] <= 0.05
    )
    if passed:
        print("  🎉 PASSED — Model meets production targets!")
    else:
        reasons = []
        if summary["avg_f1"] < 0.80:
            reasons.append(f"F1 {summary['avg_f1']:.1%} < 80%")
        if summary["parse_rate"] < 0.90:
            reasons.append(f"Parse rate {summary['parse_rate']:.1%} < 90%")
        if summary["avg_hallucination"] > 0.05:
            reasons.append(f"Hallucination {summary['avg_hallucination']:.1%} > 5%")
        print(f"  ⚠️  NOT YET — {', '.join(reasons)}")
    print("─" * 60)


if __name__ == "__main__":
    main()
