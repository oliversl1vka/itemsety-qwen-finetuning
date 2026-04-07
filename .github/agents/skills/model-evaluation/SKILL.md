---
name: model-evaluation
description: Evaluate fine-tuned Qwen2.5-7B model performance against Apriori ground truth. Compute P/R/F1, parse rate, hallucination rate. Supports two-phase generation and Council multi-model analysis.
---

# Model Evaluation

Comprehensive evaluation of fine-tuned models vs Apriori baseline.

## Overview

Evaluation measures:
- **Precision:** Correct itemsets / Predicted itemsets
- **Recall:** Correct itemsets / Ground truth itemsets
- **F1 Score:** Harmonic mean of P and R
- **Exact Match:** Perfect prediction rate
- **JSON Parse Rate:** Valid output percentage
- **Hallucination Rate:** Invented items percentage
- **Inference Time:** Seconds per dataset

## Current Tools

| Tool | Purpose | Location |
|------|---------|----------|
| `eval_finetuned_model.py` | Main evaluation script | `src/evaluation/` |
| `inference_utils.py` | Two-phase generation, StoppingCriteria | `src/evaluation/` |
| `council_advisor.py` | LLM Council multi-model analysis | `src/evaluation/` |
| Eval datasets | 50 fixed unseen datasets | `data/eval_datasets_v2/` |

## Quick Start

### Evaluate Model
```bash
python src/evaluation/eval_finetuned_model.py \
  --model-path OliverSlivka/qwen2.5-7b-itemset-v3 \
  --eval-dir data/eval_datasets_v2 \
  --count 50
```

### Council Analysis (Multi-Model)
```bash
python src/evaluation/council_advisor.py \
  --eval-results eval_results.json \
  --models gpt-4.1-mini,claude-sonnet,gemini-2.5-flash
```

## Evaluation Process

### 1. Fixed Evaluation Datasets
Eval datasets are **versioned and never change** between model versions:
```
data/eval_datasets_v2/
├── eval_ds_0001_8x10.csv
├── eval_ds_0002_12x6.csv
├── ...
└── eval_ds_0050_15x8.csv
```

### 2. Run Apriori Ground Truth
```bash
python pipeline.py --data-dir data/eval_datasets_v2 --min-support 3 --max-size 3
```

### 3. Two-Phase Model Inference (v3)
```python
from src.evaluation.inference_utils import generate_two_phase, ItemsetStoppingCriteria

FastLanguageModel.for_inference(model)

# Phase 1: Generate <think> reasoning
# Phase 2: Generate JSON output after </think>
output = generate_two_phase(
    model=model,
    tokenizer=tokenizer,
    prompt=formatted_prompt,
    max_think_tokens=512,
    max_json_tokens=1024,
    temperature=0.3,
    repetition_penalty=1.2,
)
```

### 4. Compare and Score
```python
def compute_metrics(predicted, ground_truth):
    """Compare predicted itemsets against Apriori ground truth."""
    pred_set = {frozenset(i["itemset"]) for i in predicted}
    true_set = {frozenset(i["itemset"]) for i in ground_truth}
    
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(true_set) if true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}
```

## Inference Utilities (v3)

### StoppingCriteria
Prevents infinite generation and repetition loops:
```python
from src.evaluation.inference_utils import ItemsetStoppingCriteria

stopping = ItemsetStoppingCriteria(
    tokenizer=tokenizer,
    stop_tokens=["]"],          # Stop after JSON array closes
    max_repetitions=3,          # Detect repetition loops
)
```

### Two-Phase Generation
Separates reasoning from output generation:
1. **Phase 1:** Generate up to `</think>` (reasoning)
2. **Phase 2:** Generate JSON array (structured output)

Benefits: Better token budget allocation, prevents think-bleed into JSON.

### Dynamic Token Budget
```python
from src.training.training_utils import estimate_token_budget

budget = estimate_token_budget(
    csv_text=csv_text,
    n_rows=15,
    n_cols=8,
    max_size=3,
)
# Returns estimated max_new_tokens needed
```

## Metrics Definitions

### Core Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Precision | TP / (TP + FP) | High = Few false positives |
| Recall | TP / (TP + FN) | High = Few false negatives |
| F1 Score | 2×P×R / (P+R) | Balanced measure |
| Exact Match | Perfect / Total | Strictest measure |

### Reliability Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Parse Rate | Valid JSON / Total | ≥ 95% |
| Hallucination | Invented items / All predicted | ≤ 5% |
| Repetition Rate | Looped outputs / Total | ≤ 5% |

## Performance Targets

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Precision | 60% | 80% | 90%+ |
| Recall | 60% | 80% | 90%+ |
| F1 Score | 60% | 80% | 90%+ |
| Exact Match | 30% | 50% | 70%+ |
| Parse Rate | 80% | 95% | 99%+ |
| Hallucination | <10% | <5% | <1% |
| Inference Time | <120s | <60s | <30s |

## Evaluation History

| Version | F1 | Parse Rate | Notes |
|---------|-----|-----------|-------|
| v1 (3B, merged 4-bit) | 17% | 20% | Merge corruption, repetition loops |
| v2 (7B, SFT only) | — | — | Baseline improvement |
| v3 (7B, SFT+DPO) | — | — | Council-tuned hyperparams |

## LLM Council Evaluation

Multi-model analysis for deeper failure understanding:

```python
from src.evaluation.council_advisor import CouncilAdvisor

council = CouncilAdvisor(
    models=["gpt-4.1-mini", "claude-sonnet-4-20250514", "gemini-2.5-flash"],
    provider="openrouter",
)

# Analyze failure patterns across models
analysis = council.analyze_failures(
    eval_results=results,
    focus="json_parse_failures",
)
```

### Council Use Cases
- Identify systematic failure patterns
- Compare reasoning quality across models
- Get diverse perspectives on improvement strategies
- Validate hyperparameter recommendations

## Raw Capture Workflow

For debugging model outputs during evaluation:

```
eval_raw_capture/
├── ds_0013_18x12_4d718b95_raw.txt    # Raw model output
├── ds_0045_6x12_2802645a_raw.txt
├── ...
└── summary.json                       # Aggregate metrics
```

```
eval_reppenalty/                         # With repetition_penalty=1.2
├── ds_0013_18x12_4d718b95_raw.txt
├── ...
```

## Failure Analysis

### Common Patterns

| Pattern | Cause | Fix |
|---------|-------|-----|
| Empty output | Model stops early | Increase max_new_tokens |
| Invalid JSON | Text before/after | Strip to `[` ... `]` |
| Truncated | Context too long | Use dynamic token budget |
| Repetition loop | Training issue | `repetition_penalty=1.2` + DPO |
| Hallucination | Weak grounding | More diverse training data |
| Think-bleed | No phase separation | Use two-phase generation |
| Wrong format | Support as % not decimal | Fix training examples |

### Per-Dataset Debugging
```bash
# Inspect raw output
cat eval_raw_capture/ds_0013_18x12_4d718b95_raw.txt

# Compare with Apriori
cat artifacts/apriori_outputs/apriori_output_ds_0013_*.json | jq '.'
```

## Visualization

Generate comparison charts:
```bash
python src/utils/visualization.py --db runs.db --outdir visuals
```

Produces:
- `f1_distribution.png` — F1 score histogram
- `precision_recall_scatter.png` — P vs R scatter
- `model_comparison.png` — Base vs Fine-tuned vs Apriori bars

## ⚠️ Evaluation Best Practices

1. **Always use fixed eval datasets** — Never change between model versions
2. **Run Apriori first** — Ground truth must exist before model inference
3. **Save raw outputs** — Keep `eval_raw_capture/` for debugging
4. **Test with repetition_penalty** — Compare with/without
5. **Record everything** — Log to Obsidian `Experiments/` folder
6. **Council for hard cases** — Use multi-model analysis for persistent failures
