#!/usr/bin/env python3
"""Run LLM Council analysis on eval results with comprehensive failure data.

This script prepares a detailed council query including:
- Raw model outputs (all 20 datasets)
- Pattern analysis summary
- Training data format comparison
- Training configuration details
- GRPO reward function analysis

Then runs the 3-stage council for actionable recommendations.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.council_advisor import (
    _run_council,
    _stage1_collect_responses,
    _stage2_collect_rankings,
    _stage3_synthesize_final,
    DEFAULT_COUNCIL_MODELS,
    DEFAULT_CHAIRMAN_MODEL,
)

RAW_DIR = PROJECT_ROOT / "src/evaluation/eval_results/raw_20260301_2212"
SUMMARY_JSON = PROJECT_ROOT / "src/evaluation/eval_results/summary_20260301_2212.json"
RESULTS_CSV = PROJECT_ROOT / "src/evaluation/eval_results/results_20260301_2212.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs/reports"


def build_council_query() -> str:
    """Build the comprehensive council query."""
    
    # Load summary
    with open(SUMMARY_JSON) as f:
        summary = json.load(f)
    
    # Load all raw outputs
    raw_outputs = {}
    for fname in sorted(os.listdir(RAW_DIR)):
        with open(RAW_DIR / fname) as f:
            raw_outputs[fname.replace(".csv.txt", "")] = f.read()
    
    # Build raw output samples (all 20)
    raw_section = []
    for name, raw in raw_outputs.items():
        # Truncate very long outputs (repetition loops)
        display = raw[:500] + ("..." if len(raw) > 500 else "")
        raw_section.append(f"--- {name} ---\n{display}")
    
    query = f"""You are an expert in LLM fine-tuning, frequent itemset mining, NLP evaluation, and LoRA/QLoRA training.

A Qwen2.5-7B model was fine-tuned using 3-phase training (SFT-CoT → DPO-Real → GRPO) to extract frequent itemsets from CSV transactional data. The evaluation results are CATASTROPHIC — 0% F1 across ALL 20 test datasets. Please analyze deeply and provide actionable recommendations.

═══════════════════════════════════════════════════════════
  EVALUATION SUMMARY
═══════════════════════════════════════════════════════════
  Model: {summary['model']}
  Datasets evaluated: {summary['n_datasets']}
  Avg Precision: {summary['avg_precision']:.1%}
  Avg Recall: {summary['avg_recall']:.1%}
  Avg F1: {summary['avg_f1']:.1%}
  Exact Match Rate: {summary['exact_match_rate']:.1%}
  JSON Parse Rate: {summary['parse_rate']:.1%}
  Think Rate (has <think> tags): {summary['think_rate']:.1%}
  Hallucination Rate: {summary['hallucination_rate']:.1%}
  Avg Inference Time: {summary['avg_time_s']:.1f}s

═══════════════════════════════════════════════════════════
  FAILURE PATTERN ANALYSIS (across 20 datasets)
═══════════════════════════════════════════════════════════
  no_think (no <think> reasoning):    20/20 (100%)
  column_only (bare col names):       16/20 (80%)
  correct_col_value format:            3/20 (15%)
  item_split ("age","16" separate):    2/20 (10%)
  below_min_support:                   3/20 (15%)
  duplicate_itemsets:                   2/20 (10%)
  json_parse_fail:                     1/20 (5%)
  hallucinated_rows:                   1/20 (5%)
  post_json_text:                      1/20 (5%)
  repetition_loop:                     1/20 (5%) — same itemset repeated 50+ times

═══════════════════════════════════════════════════════════
  FORMAT MISMATCH — ROOT CAUSE
═══════════════════════════════════════════════════════════

TRAINING DATA uses col:value format:
  Ground truth: [{{"itemset": ["age:15"], "count": 3, "rows": ["Row 1", "Row 2", "Row 7"]}}]
  SFT chosen:   [{{"itemset": ["failures:0"], "count": 6, "rows": ["Row 1", ...]}}]
  DPO rejected:  [{{"itemset": ["age:15"], "count": 3, "rows": ["Row 1", "Row 2", "Row 7"]}}]

MODEL OUTPUT typically uses BARE COLUMN NAMES (no values):
  [{{"itemset": ["famrel", "health", "medu"], "count": 3, "rows": ["Row 1", "Row 3", "Row 7"]}}]
  OR splits col:value into separate items:
  [{{"itemset": ["age", "16"], "count": 3, "rows": ["Row 1", "Row 3", "Row 5"]}}]
  
Only 3/20 datasets got the col:value format right, but even those had wrong values:
  [{{"itemset": ["fedu:22", "health:5", "studytime:22"], "count": 5}}]  ← "fedu:22" is wrong (concatenation)
  [{{"itemset": ["battery_life_rating:21", "design_rating:1"], "count": 3}}]  ← "21" is wrong

═══════════════════════════════════════════════════════════
  TRAINING CONFIGURATION
═══════════════════════════════════════════════════════════
  Base model: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
  LoRA: r=16, alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
  Max seq length: 4096
  
  Phase 1 — SFT-CoT:
    Data: 348 examples (ChatML with <think> reasoning)
    Epochs: 3, LR: 2e-4, batch: 1, grad_accum: 4
    Duration: 4m10s
    Final loss: 0.0678, val_loss: 0.038
    
  Phase 2 — DPO-Real:
    Data: 606 preference pairs (chosen=CoT+correct, rejected=real GPT-4.1-nano failures)
    Epochs: 1, LR: 5e-5, beta: 0.1, batch: 1, grad_accum: 4
    Duration: 14m1s
    Final loss: 0.0255, reward_accuracy: 100%
    
  Phase 3 — GRPO:
    Data: 314 prompts (generates completions, scores with reward functions)
    Steps: 200, LR: 5e-6, batch: 1, grad_accum: 4, num_generations: 4
    Duration: 45m7s
    4 reward functions: json_format, itemset_f1, count_accuracy, thinking
    ⚠️ F1 reward was ~0.0 throughout training (red flag!)
    ⚠️ json_format reward was 0.95-1.0 (model learned JSON but not content)

═══════════════════════════════════════════════════════════
  GRPO REWARD FUNCTIONS (potential issues)
═══════════════════════════════════════════════════════════
  
  1. json_format_reward: Checks valid JSON + schema keys.
     ISSUE: Rewards ANY valid JSON with "itemset" key — doesn't check format of items.
     
  2. itemset_f1_reward: Compares pred_sets vs true_sets using frozenset(itemset).
     ISSUE: Does NOT normalize items (no lowercase, no strip). Uses frozenset comparison
     so ["age", "16"] ≠ ["age:16"], which means F1 was always ~0.0 during GRPO
     and the model got NO learning signal for correctness!
     
  3. count_accuracy_reward: Checks count values for matched itemsets.
     ISSUE: Since F1 matching failed (format mismatch), this also returned 0.0 always.
     
  4. thinking_reward: Checks for <think> tags + quality.
     This should have worked but the model lost <think> behavior entirely.

═══════════════════════════════════════════════════════════
  RAW MODEL OUTPUTS (all 20 datasets)
═══════════════════════════════════════════════════════════

{chr(10).join(raw_section)}

═══════════════════════════════════════════════════════════
  QUESTIONS FOR THE COUNCIL
═══════════════════════════════════════════════════════════

1. ROOT CAUSE: Why did the 3-phase training produce a model that:
   a) Lost the col:value format it was trained on (outputs bare column names)?
   b) Lost <think> reasoning completely (0% think rate)?
   c) Only finds 1-14 itemsets per dataset vs 3-387 ground truth?
   
2. PHASE ANALYSIS: 
   a) Did GRPO destroy what SFT+DPO learned? (catastrophic forgetting)
   b) Was 200 GRPO steps with ~0 reward signal harmful? 
   c) Should we skip GRPO entirely for the next iteration?
   
3. TRAINING DATA:
   a) Are 348 SFT + 606 DPO examples enough for 7B model?
   b) Should we increase training data quantity or quality?
   c) Should we use a different data format (e.g., raw CSV columns instead of col:value)?
   
4. MODEL SIZE:
   a) The user has access to models from 0.5B to 72B. What's the minimum model size 
      that can reliably learn this task?
   b) Should we try Qwen2.5-3B (smaller but faster to iterate) or Qwen2.5-14B (larger)?
   c) How does model size interact with the col:value format learning difficulty?
   
5. CONCRETE NEXT STEPS:
   a) What specific changes to training configuration would you recommend?
   b) Should we try SFT-only first (skip DPO+GRPO) to establish a baseline?
   c) What hyperparameter changes (epochs, LR, LoRA rank) would help?
   d) How should we fix the GRPO reward functions if we use GRPO again?
   
6. ALTERNATIVE APPROACHES:
   a) Should we consider full fine-tuning instead of LoRA?
   b) Should we try a different base model family (Llama, Mistral, Phi)?
   c) Should we simplify the task (e.g., only single items first, then pairs)?

Please provide thorough, prioritized, actionable recommendations.
The user wants the SMALLEST model that reliably achieves ≥80% F1 on this task."""
    
    return query


def build_training_advice_query() -> str:
    """Build the training script improvement query."""
    
    # Read the training notebook for context
    notebook_path = PROJECT_ROOT / "notebooks/training_3phase_7b.ipynb"
    with open(notebook_path) as f:
        notebook_content = f.read()
    
    # Extract just the code cells (skip markdown)
    import json as json_mod
    nb = json_mod.loads(notebook_content)
    code_cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if len(source.strip()) > 0:
                code_cells.append(source)
    
    training_code = "\n\n# --- NEXT CELL ---\n\n".join(code_cells)
    # Truncate if too long
    if len(training_code) > 12000:
        training_code = training_code[:12000] + "\n\n... (truncated)"
    
    query = f"""You are an expert in LLM fine-tuning with LoRA/QLoRA, HuggingFace TRL, and Unsloth optimization.

Below is the complete training notebook code for a Qwen2.5-7B model fine-tuned to extract frequent itemsets from CSV data. The model's evaluation showed CATASTROPHIC failure: 0% F1, 0% think rate, 88.8% hallucination rate.

KEY PROBLEMS IDENTIFIED:
1. Model outputs bare column names ("age", "health") instead of col:value format ("age:16", "health:5")
2. Model completely lost <think> reasoning (trained in SFT phase, lost by DPO/GRPO)
3. GRPO F1 reward was ~0.0 throughout training (no learning signal)
4. Model only finds 1-14 itemsets vs 3-387 ground truth (extreme low recall)

TRAINING CODE:

{training_code}

QUESTIONS:
1. What are the specific bugs or configuration issues in this training code?
2. How should the GRPO reward functions be fixed (normalization, format checking)?
3. What hyperparameter changes would you recommend (epochs, LR, LoRA config)?
4. Should we skip GRPO and use only SFT+DPO? Or SFT-only first?
5. Is the 3-phase approach fundamentally flawed for this task, or just misconfigured?
6. The user wants the smallest model possible (3B, 7B, 14B). What do you recommend?
7. Provide a concrete, corrected training configuration that would likely work.

Please be very specific and provide corrected code snippets where relevant."""
    
    return query


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  LLM COUNCIL — Evaluation Analysis + Training Advice")
    print("=" * 70)
    
    # Council 1: Eval analysis
    print("\n📊 Council 1: Evaluation Results Analysis")
    print("-" * 50)
    eval_query = build_council_query()
    print(f"  Query length: {len(eval_query)} chars")
    
    eval_result = await _run_council(
        eval_query, DEFAULT_COUNCIL_MODELS, DEFAULT_CHAIRMAN_MODEL
    )
    
    eval_output = OUTPUT_DIR / "council_eval_analysis.json"
    with open(eval_output, "w") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Eval analysis saved → {eval_output}")
    
    # Print chairman's synthesis
    if "stage3" in eval_result:
        print("\n" + "=" * 70)
        print("  CHAIRMAN SYNTHESIS — EVAL ANALYSIS")
        print("=" * 70)
        print(eval_result["stage3"].get("response", "No response"))
    
    # Council 2: Training advice
    print("\n\n📝 Council 2: Training Script Improvement Advice")
    print("-" * 50)
    training_query = build_training_advice_query()
    print(f"  Query length: {len(training_query)} chars")
    
    training_result = await _run_council(
        training_query, DEFAULT_COUNCIL_MODELS, DEFAULT_CHAIRMAN_MODEL
    )
    
    training_output = OUTPUT_DIR / "council_training_advice.json"
    with open(training_output, "w") as f:
        json.dump(training_result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Training advice saved → {training_output}")
    
    # Print chairman's synthesis
    if "stage3" in training_result:
        print("\n" + "=" * 70)
        print("  CHAIRMAN SYNTHESIS — TRAINING ADVICE")
        print("=" * 70)
        print(training_result["stage3"].get("response", "No response"))
    
    # Print rankings
    for name, result in [("Eval Analysis", eval_result), ("Training Advice", training_result)]:
        rankings = result.get("metadata", {}).get("aggregate_rankings", [])
        if rankings:
            print(f"\n{name} — Model Rankings:")
            for r in rankings:
                print(f"  #{r['average_rank']:.1f} {r['model']}")


if __name__ == "__main__":
    asyncio.run(main())
