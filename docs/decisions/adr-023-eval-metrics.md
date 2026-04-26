# ADR-023: Seven Evaluation Metrics

**Status:** Accepted  
**Date:** 2026-03

## Context

A single accuracy metric is insufficient to understand model behavior on itemset extraction. Different failure modes (over-prediction, under-prediction, hallucination, format errors) require different metrics to diagnose.

## Decision

**Seven metrics** capturing complementary aspects of model quality.

## Metric Selection Rationale

| Metric | Why Included |
|--------|-------------|
| **Precision** | Distinguishes over-prediction (many wrong itemsets) from conservative models |
| **Recall** | Distinguishes under-prediction (missing itemsets) from precision-focused models |
| **F1** | Primary aggregate metric; harmonic mean prevents gaming by extreme P or R |
| **Exact Match** | The production-readiness bar: is the output perfectly correct? |
| **Count Accuracy** | Separately measures count estimation (within +/-1) for matched itemsets |
| **Hallucination Rate** | Directly tracks the dominant failure mode (`item_missing_in_row`) |
| **JSON Parse Rate** | Tracks structural format compliance; a model with high F1 but low parse rate is unusable |

## Why Not a Single Metric

F1 alone masks qualitatively different failures:

- A model with P=100%, R=10%, F1=18% is very conservative -- it reports few but correct itemsets
- A model with P=10%, R=100%, F1=18% has the same F1 but is over-predicting wildly

Hallucination Rate specifically tracks whether the model invents evidence (items not in the CSV), which is the most dangerous failure mode for production use.

JSON Parse Rate catches format compliance issues that F1 ignores: a model that produces invalid JSON scores 0% on all other metrics, but the reason is format failure, not reasoning failure.

## Source Evidence

- `src/evaluation/eval_finetuned_model.py:417-500` -- metric computation
