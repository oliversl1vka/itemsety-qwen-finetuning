# ADR-024: Fixed Evaluation Set

**Status:** Accepted  
**Date:** 2026-03

## Context

Models are evaluated across multiple training iterations (v1, v2, v3) and against baselines (GPT-4.1-mini, base Qwen). The evaluation data must enable fair comparison.

## Options Considered

| Strategy | Fair Comparison | Bias Risk |
|----------|----------------|-----------|
| Random sampling at eval time | No (different data each time) | Random variance |
| K-fold cross-validation | Moderate | Complex, expensive |
| **Fixed holdout set** | Yes (same data for all models) | Possible difficulty bias |

## Decision

**30 fixed evaluation datasets** in `data/eval_datasets_v2/`, versioned and never changed between model comparisons.

## Rationale

Random sampling at evaluation time means each run sees different data. A model might score higher simply by being evaluated on easier datasets. A fixed set ensures **exact apples-to-apples comparison**: every model is evaluated on precisely the same 30 datasets.

**Leakage prevention:** evaluation datasets are explicitly excluded from training data. The evaluation script checks filename overlap against `sft_cot_v3.json` and `dpo_real_v2.json`. The SHA-256 hash overlap count between training and evaluation is confirmed to be zero.

**Coverage:** the 30 datasets span the full difficulty range (5--15 rows, 3--15 columns), avoiding bias toward easy or hard instances.

## Trade-offs

- If the fixed set happens to over-represent easy or hard cases, metrics are systematically biased
- 30 datasets may be too few for statistically significant comparisons on small effect sizes

## Source Evidence

- `src/data_generation/generate_eval_datasets_v2.py` -- fixed generation
- `src/evaluation/eval_finetuned_model.py:506-526` -- leakage exclusion logic
- Published: [OliverSlivka/itemset-eval-v2](https://huggingface.co/datasets/OliverSlivka/itemset-eval-v2)
