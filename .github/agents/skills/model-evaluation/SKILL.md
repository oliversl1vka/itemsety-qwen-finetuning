---
name: model-evaluation
description: Evaluate fine-tuned model performance against Apriori ground truth. Compute P/R/F1, parse rate, hallucination rate, and inference time.
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

## Quick Start

### Evaluate Model
```bash
python src/evaluation/eval_finetuned_model.py --model-path OliverSlivka/qwen2.5-3b-itemset-extractor --count 20
```

### Generate Eval Datasets
```bash
python src/data_generation/generate_eval_datasets_v2.py --count 50 --output data/eval_datasets
```

## Evaluation Process

### 1. Generate Unseen Datasets
```bash
python src/data_generation/generate_eval_datasets_v2.py --count 50
```

### 2. Run Apriori (Ground Truth)
```bash
python pipeline.py --data-dir data/eval_datasets --min-support 3 --max-size 3
```

### 3. Run Model Inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("OliverSlivka/qwen2.5-3b-itemset-extractor")
tokenizer = AutoTokenizer.from_pretrained("OliverSlivka/qwen2.5-3b-itemset-extractor")

# Generate for each dataset
output = model.generate(input_ids, max_new_tokens=1024)
response = tokenizer.decode(output[0])
```

### 4. Compare and Score
```python
def compute_metrics(predicted, ground_truth):
    pred_set = {frozenset(i["itemset"]) for i in predicted}
    true_set = {frozenset(i["itemset"]) for i in ground_truth}
    
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(true_set) if true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}
```

## Metrics Definitions

### Precision
```
Precision = |Correct Predictions| / |All Predictions|
```
High precision = Few false positives

### Recall
```
Recall = |Correct Predictions| / |Ground Truth|
```
High recall = Few false negatives

### F1 Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Balanced measure of both

### JSON Parse Rate
```
Parse Rate = |Valid JSON Outputs| / |Total Outputs|
```
Critical for usability

### Hallucination Rate
```
Hallucination = |Invented Items| / |All Predicted Items|
```
Should be < 5%

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

## Current Results (V2 Model)

| Metric | Value | Status |
|--------|-------|--------|
| Precision | 18% | ❌ Below target |
| Recall | 15% | ❌ Below target |
| F1 Score | 17% | ❌ Below target |
| Parse Rate | 20% | ❌ Critical issue |
| Inference Time | 120s | ⚠️ Slow |

**Primary Issue:** Low JSON parse rate causing cascade failures

## Output Format

### Evaluation Report
```json
{
  "model": "OliverSlivka/qwen2.5-3b-itemset-extractor",
  "eval_date": "2026-02-01",
  "datasets_evaluated": 50,
  "metrics": {
    "precision": {"mean": 0.18, "std": 0.12},
    "recall": {"mean": 0.15, "std": 0.10},
    "f1": {"mean": 0.17, "std": 0.11},
    "parse_rate": 0.20,
    "hallucination_rate": 0.08,
    "avg_inference_time_s": 120
  },
  "failure_patterns": [
    {"pattern": "empty_output", "count": 15},
    {"pattern": "invalid_json", "count": 25},
    {"pattern": "truncated", "count": 5}
  ]
}
```

## Failure Analysis

### Common Patterns

| Pattern | Cause | Fix |
|---------|-------|-----|
| Empty output | Model stops early | Increase max_new_tokens |
| Invalid JSON | Text before/after | Strip to [ ... ] |
| Truncated | Context too long | Reduce input size |
| Repetition | Training issue | Add repetition_penalty |
| Hallucination | Weak grounding | More diverse training data |

## Comparison Script

Compare two models:
```bash
python src/evaluation/eval_finetuned_model.py \
  --model-a OliverSlivka/qwen2.5-3b-v1 \
  --model-b OliverSlivka/qwen2.5-3b-v2 \
  --count 50
```

## Visualization

Generate charts:
```bash
python src/utils/visualization.py --db runs.db --outdir visuals --eval-mode
```

Produces:
- `f1_distribution.png` — F1 score histogram
- `precision_recall_scatter.png` — P vs R scatter
- `model_comparison.png` — Side-by-side bars
