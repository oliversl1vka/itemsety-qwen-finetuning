# Evaluation Report — Qwen2.5-7B Itemset Extractor v1

**Date:** 2026-03-01  
**Model:** `OliverSlivka/qwen2.5-7b-itemset-extractor`  
**Training:** 3-phase (SFT-CoT → DPO-Real → GRPO)  
**Eval datasets:** 20 from `OliverSlivka/itemset-eval-v2` (never used in training)

---

## Executive Summary

The model **catastrophically failed** evaluation: **0% F1, 0% recall, 0% precision** across all 20 datasets. The root cause is a combination of **GRPO-induced catastrophic forgetting** and **format-unaware reward functions** that gave zero learning signal during Phase 3. The model lost both `col:value` item format and `<think>` reasoning it learned in SFT.

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Avg F1 | ≥80% | **0.0%** | ❌ Critical |
| Avg Precision | ≥80% | **0.0%** | ❌ Critical |
| Avg Recall | ≥80% | **0.0%** | ❌ Critical |
| Exact Match | ≥50% | **0.0%** | ❌ Critical |
| JSON Parse Rate | ≥90% | **95.0%** | ✅ Pass |
| Think Rate | 100% | **0.0%** | ❌ Critical |
| Hallucination Rate | ≤5% | **88.8%** | ❌ Critical |
| Avg Inference Time | ≤60s | **4.6s** | ✅ Pass |

---

## Failure Pattern Analysis (20 datasets)

### Pattern Distribution

| Pattern | Count | % | Description |
|---------|-------|---|-------------|
| no_think | 20/20 | 100% | Zero `<think>` reasoning blocks |
| column_only | 16/20 | 80% | Bare column names without values (e.g., `"age"` instead of `"age:16"`) |
| correct_col_value | 3/20 | 15% | Correct `col:value` format (but wrong values) |
| item_split | 2/20 | 10% | Split `col:value` into separate items (`["age", "16"]`) |
| below_min_support | 3/20 | 15% | Itemsets with count < min_support=3 |
| duplicate_itemsets | 2/20 | 10% | Same itemset repeated |
| json_fail | 1/20 | 5% | Could not parse JSON at all |
| repetition_loop | 1/20 | 5% | Same itemset repeated 50+ times |
| post_json_text | 1/20 | 5% | Explanatory text after JSON array |
| halluc_rows | 1/20 | 5% | Invalid row references (e.g., "Row 91", "Row  t") |

### Critical Finding: Format Mismatch

**Training data format (correct):**
```json
[{"itemset": ["age:16"], "count": 3, "rows": ["Row 1", "Row 3", "Row 5"]}]
```

**Model output (80% of cases):**
```json
[{"itemset": ["age", "traveltime"], "count": 6, "rows": ["Row 1", "Row 2"]}]
```

The model strips the `:value` suffix and treats column names as items. Since the eval ground truth uses `col:value` format, **zero items match** → 0% F1.

### Output categorization by source domain:

| Source domain | Datasets | Model behavior |
|---------------|----------|----------------|
| student-mat/por | 10 | Column names only: `["famrel", "health", "medu"]` |
| carInsurance | 5 | Column names only: `["carinsurance", "hhinsurance"]` |
| Mobile Reviews | 4 | Mixed: some `col:value`, some column-only, some value concatenation |
| superheroes | 1 | Column names only: `["has_accelerated_healinging", "has_super_strength"]` |

---

## Root Cause Analysis

### Primary Cause: GRPO Destroyed SFT+DPO Learning (Catastrophic Forgetting)

**Evidence:**
1. SFT achieved loss=0.0678 (good), model likely learned `col:value` format and `<think>` reasoning
2. DPO achieved reward_accuracy=100% (model distinguished chosen vs rejected)
3. GRPO ran 200 steps with **F1 reward ≈ 0.0 throughout** — the model got NO learning signal for correctness
4. The model was rewarded only for JSON format (0.95-1.0) and punished for everything else → it learned to output minimal JSON with bare column names

**Why F1 reward was always 0.0 during GRPO:**

The `itemset_f1_reward` function compares `frozenset(predicted_itemset)` vs `frozenset(ground_truth_itemset)`. But during GRPO generation, if the model started outputting bare column names, the frozensets would never match:
- Predicted: `frozenset({"age", "16"})` or `frozenset({"age", "traveltime"})`
- Ground truth: `frozenset({"age:16"})` or `frozenset({"age:16", "traveltime:1"})`

The reward function has **no normalization** — it doesn't lowercase, strip, or handle format variations. This means GRPO trained for 45 minutes with essentially **random exploration and zero reward signal**, which destroyed the format learned in SFT/DPO.

### Secondary Cause: Reward Function Defects

| Reward Function | Issue |
|----------------|-------|
| `json_format_reward` | Only checks JSON validity + schema keys. Does NOT verify item format matches `col:value` pattern. Gave 0.95-1.0 reward even for wrong format. |
| `itemset_f1_reward` | No normalization. Uses raw `frozenset()` comparison. Any format deviation → 0 match → 0 F1 → 0 reward. **Zero learning signal.** |
| `count_accuracy_reward` | Depends on matched itemsets from F1. Since F1=0, this also gave 0. |
| `thinking_reward` | Should have preserved `<think>` behavior, but was outweighed by the other 3 rewards pushing the model toward degenerate JSON output. |

### Tertiary Cause: Insufficient Training Data

- 348 SFT examples for a 7B model may be too few to deeply learn the `col:value` convention
- The format is unusual (most LLM training data doesn't have `col:value` tokens)
- Qwen2.5-7B's base knowledge of CSV items doesn't include `col:value` — it defaults to treating column names as items

---

## Training Phase Metrics (from training session)

| Phase | Duration | Final Loss | Key Metric | Assessment |
|-------|----------|-----------|------------|------------|
| SFT-CoT | 4m10s | 0.0678 | val_loss: 0.038 | ⚠️ OK but may need more epochs |
| DPO-Real | 14m1s | 0.0255 | reward_accuracy: 100% | ⚠️ Suspiciously perfect |
| GRPO | 45m7s | ~0.0 | F1: ~0.0, json_format: 0.95-1.0 | ❌ **Destructive** |

---

## LLM Council Analysis

Two LLM Council sessions were run on 2026-03-01 using 4 expert models + chairman synthesizer:

**Council models:** `claude-sonnet-4.6`, `gemini-3-flash-preview`, `deepseek-v3.2`, `grok-4.1-fast`  
**Chairman:** `claude-opus-4.6`  
**Full reports:** [council_eval_analysis.json](council_eval_analysis.json) | [council_training_advice.json](council_training_advice.json)

### Council 1: Evaluation Analysis

**Rankings:** Claude Sonnet #1 (avg 1.0), Grok #2 (avg 2.25), Gemini #3 (avg 2.75), DeepSeek #4 (avg 4.0)

**Unanimous diagnoses:**
1. GRPO reward functions gave zero correctness signal → model only optimized for JSON format
2. DPO phase destroyed `<think>` reasoning learned in SFT
3. Model found "shortcut": output bare column names in valid JSON = max reward

### Council 2: Training Advice

**Rankings:** Claude Sonnet #1 (avg 1.0), Grok #2 (avg 2.5), Gemini #3 (avg 2.5), DeepSeek #4 (avg 4.0)

**Unanimous agreements (4/4):**
- **Remove DPO entirely** — DPO's contrastive loss destroyed `<think>` reasoning; designed for preference not hard structural constraints
- **Stay with 7B** — sufficient capacity for CSV parsing + itemset enumeration; 3B too weak, 14B overkill
- **LoRA alpha/r ratio was critically broken** — was 16/64 = 0.25, must be 64/32 = 2.0 (weights barely updated)
- **SFT was severely undertrained** — 2 epochs insufficient; need 5 for format memorization
- **Save adapters only** between phases — `merged_4bit_forced` introduces cascading quantization noise
- **GRPO needs 5 partial-credit reward functions** — structure, col:val format, F1, recall bonus, anti-hallucination
- **Add format verification gate** between SFT and GRPO — don't proceed unless >80% format compliance
- **Set `packing=False`** in SFTConfig — cross-example contamination corrupts structured output training

---

## Recommendations (Prioritized — Council-Confirmed)

### Priority 1: Fix LoRA Configuration (CRITICAL — 3/4 consensus)

The LoRA alpha/rank ratio was **0.25** (alpha=16, r=64). Effective update = `(alpha/r) * W_update`. At 0.25, weights barely deviated from base → model couldn't override pretrained behavior.

```python
# BROKEN (original):
"lora_r": 64,      "lora_alpha": 16   # ratio = 0.25 ← barely learning

# FIXED:
"lora_r": 32,      "lora_alpha": 64   # ratio = 2.0 ← standard for fine-tuning
"lora_dropout": 0.05                    # was 0 — small dropout helps generalization
```

### Priority 2: Remove DPO Phase Entirely (4/4 unanimous)

DPO is designed for subjective preference alignment, not hard structural constraints. DPO's contrastive loss DECREASES probability of rejected format tokens, but rejected and chosen formats share tokens early in sequence → model becomes uncertain about ALL output formats.

**Evidence:** 0% think rate after DPO/GRPO = DPO killed `<think>` reasoning.

### Priority 3: Fix SFT Training (4/4 consensus)

```python
"sft_epochs": 5,         # was 2 — CRITICAL: need format memorization
"sft_lr": 1e-4,          # was 2e-4 — lower for stability with more epochs
"sft_grad_accum": 8,     # was 4 — effective batch = 16 for stability
"sft_warmup_ratio": 0.10 # was 0.05 — longer warmup for format learning
```

**Added (missing from original):**
```python
"packing": False,         # CRITICAL: never pack for structured output tasks
"weight_decay": 0.01,     # regularization
"max_grad_norm": 1.0,     # gradient clipping
```

### Priority 4: Save Adapters Only Between Phases (3/4 consensus)

```python
# BROKEN — introduces cascading quantization noise:
model.save_pretrained_merged(..., save_method="merged_4bit_forced")

# FIXED — save LoRA adapters only:
model.save_pretrained(CONFIG["sft_output_dir"])
tokenizer.save_pretrained(CONFIG["sft_output_dir"])
```

### Priority 5: Implement 5 GRPO Reward Functions (4/4 consensus)

Only add GRPO back after SFT achieves >80% format compliance via verification gate.

| Reward Function | Purpose | Range |
|----------------|---------|-------|
| `reward_structure` | Checks `<think>` + `<answer>` tags + content | [0, 1] |
| `reward_colval_format` | Fraction of items using `col:val` format | [0, 1] |
| `reward_f1` | F1 with partial credit (sqrt scaling) | [0, 1] |
| `reward_recall_bonus` | 70% recall + 30% precision weighting | [0, 1] |
| `reward_grounding` | Anti-hallucination (items must exist in CSV) | [0, 1] |

**Key design principles:**
- Every function gives **partial credit** (not binary 0/1)
- F1 uses `f1 ** 0.5` scaling so F1=0.1 → reward=0.32 (not flat 0)
- Recall bonus combats the 1–14 items vs 3–387 ground truth problem
- All functions test independently on known good/bad examples before training

### Priority 6: Add Format Verification Gate

Between SFT and GRPO, generate 20 val samples and check:
- `<think>` tag present with >20 chars content
- `<answer>` tag present
- Items use `col:val` format

**Gate threshold:** ≥80% compliance → proceed to GRPO. If <50%, increase SFT epochs.

### Priority 7: Model Size

| Size | Verdict | Reasoning |
|------|---------|-----------|
| 3B | ❌ Insufficient | Struggles with multi-step structured output + CoT for combinatorial tasks |
| **7B** | **✅ Stay here** | Sweet spot (4/4 unanimous): sufficient capacity, fits 16GB VRAM in 4-bit |
| 14B | ⚠️ Overkill | Only if 7B with all fixes still fails; requires 40GB+ VRAM |

**The failures are entirely configuration/training issues, not model capacity issues.**

---

## Complete Corrected Configuration (Council-Synthesized)

```python
CONFIG = {
    # ── Model ────────────────────────────────────────────────────
    "base_model":        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "max_seq_length":    6144,    # was 4096 — long CSV + full answers
    "load_in_4bit":      True,

    # ── LoRA — FIXED RATIO ──────────────────────────────────────
    "lora_r":            32,      # was 64
    "lora_alpha":        64,      # was 16 → ratio 0.25→2.0 (CRITICAL)
    "lora_dropout":      0.05,    # was 0
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # ── Phase 1: SFT ────────────────────────────────────────────
    "sft_epochs":        5,       # was 2
    "sft_lr":            1e-4,    # was 2e-4
    "sft_batch_size":    2,
    "sft_grad_accum":    8,       # was 4 → eff batch=16
    "sft_warmup_ratio":  0.10,    # was 0.05
    "sft_output_dir":    "./sft_checkpoint",

    # ── Phase 2: GRPO (NO DPO) ──────────────────────────────────
    "grpo_max_steps":    500,     # was 200
    "grpo_lr":           2e-6,    # was 5e-6
    "grpo_batch_size":   1,
    "grpo_grad_accum":   8,       # was 4
    "grpo_num_generations": 6,    # was 4
    "grpo_max_completion_length": 3000,  # was 2048
    "grpo_beta":         0.001,   # KL penalty
    "grpo_temperature":  0.7,
    "grpo_output_dir":   "./grpo_checkpoint",
}
```

### Parameter Change Summary

| Parameter | Original | Fixed | Impact |
|-----------|----------|-------|--------|
| lora_alpha / lora_r | 16/64 = 0.25 | 64/32 = 2.0 | Weights actually update |
| Model save between phases | merged_4bit | Adapters only | No quantization cascade |
| DPO phase | Present | **REMOVED** | Stops `<think>` loss |
| sft_epochs | 2 | 5 | Format memorization |
| sft_lr | 2e-4 | 1e-4 | Stability |
| sft_grad_accum | 4 | 8 | Eff. batch 16 |
| packing | Not set | False | No cross-example contamination |
| weight_decay | Not set | 0.01 | Regularization |
| max_grad_norm | Not set | 1.0 | Gradient clipping |
| GRPO reward functions | Broken/absent | 5 functions | Actual learning signal |
| GRPO steps | 200 | 500 | Convergence time |
| GRPO lr | 5e-6 | 2e-6 | RL stability |
| GRPO completion length | 2048 | 3000 | Full answers |
| max_seq_length | 4096 | 6144 | Long CSV support |
| Format verification gate | Absent | Added | Prevents broken GRPO |

---

## Concrete Next Steps (Action Checklist)

1. ☐ **Fix LoRA config:** r=32, alpha=64
2. ☐ **Set `packing=False`** in SFTConfig
3. ☐ **Increase SFT to 5 epochs**, lower LR to 1e-4
4. ☐ **Add `weight_decay=0.01` and `max_grad_norm=1.0`**
5. ☐ **Remove DPO phase entirely**
6. ☐ **Save adapters only** (not merged_4bit_forced)
7. ☐ **Implement all 5 reward functions** with partial credit
8. ☐ **Add format verification gate** between SFT and GRPO
9. ☐ **Test reward functions** on known good/bad examples before training
10. ☐ **Run SFT-only first**, evaluate, then add GRPO only if format compliance >80%

---

## Raw Output Examples

### Typical failure (column names only — 80% of outputs)
```
Dataset: eval_001_15x9
Expected: [{"itemset": ["failures:0"], "count": 12, ...}]
Got:      [{"itemset": ["famrel", "health", "medu"], "count": 3, ...}]
```

### Item split failure (10% of outputs)
```
Dataset: eval_005_13x9
Expected: [{"itemset": ["failures:0"], "count": 10, ...}]
Got:      [{"itemset": ["age", "15"], "count": 3, ...}]
```

### Correct format but wrong values (15% of outputs)
```
Dataset: eval_026_11x10
Expected: [{"itemset": ["fedu:2", "health:5", "studytime:2"], ...}]
Got:      [{"itemset": ["fedu:22", "health:5", "studytime:22"], ...}]
```
Note: "fedu:22" is a concatenation error — model merged "2" and "2" from adjacent tokens.

### Repetition loop (5% of outputs)
```
Dataset: eval_003_14x8
Same itemset repeated 50+ times until max_new_tokens hit, JSON never closed properly.
```

---

**Report prepared by:** Evaluation Agent  
**Council 1 (Eval Analysis):** ✅ Complete — `docs/reports/council_eval_analysis.json`  
**Council 2 (Training Advice):** ✅ Complete — `docs/reports/council_training_advice.json`  
**Next action:** Training Agent should generate corrected 2-phase (SFT→GRPO) notebook with all council-recommended fixes
