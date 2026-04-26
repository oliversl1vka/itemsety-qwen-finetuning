# ADR-001: Apriori as Ground-Truth Oracle

**Status:** Accepted  
**Date:** 2026-02

## Context

Training a model for itemset extraction requires ground-truth labels: the correct set of frequent itemsets for each dataset. Manually labeling 500 datasets -- each potentially containing hundreds of itemsets across singles, pairs, and triples -- is infeasible for a bachelor's thesis.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Human labeling | Gold standard quality | Infeasible at 500 datasets; subjective edge cases |
| Apriori algorithm | Deterministic, exact, zero cost | Limited to exact Apriori semantics; max_size cap |
| FP-Growth | Faster than Apriori on large datasets | Same output quality; adds implementation complexity |
| LLM-as-judge | Flexible, handles ambiguity | Circular reasoning; introduces label noise |

## Decision

**Apriori algorithm** as the sole ground-truth oracle for training, validation, and evaluation.

## Rationale

Apriori is fully deterministic: given the same CSV file and the same parameters (min_support, max_size), it always produces exactly the same set of itemsets. This eliminates three problems simultaneously:

1. **No annotation cost** -- ground truth is generated automatically for any number of datasets
2. **No label noise** -- the output is mathematically exact, not subject to human error or disagreement
3. **Consistent measurement** -- the same algorithm provides ground truth for training data, pipeline validation, and model evaluation, ensuring that all measurements use the same definition of "correct"

The self-validating nature of this design is the architectural key to the entire project: correctness is measured against mathematics, not against AI or human judgment.

## Trade-offs

- Capped at max_size=3 (no 4-item or larger itemsets)
- No fuzzy matching -- an itemset is either exactly correct or wrong
- The model can only learn Apriori's definition of "frequent itemset," which may differ from domain-specific definitions

## Source Evidence

- `pipeline.py:401-466` -- `apriori_frequent_itemsets` implementation
- `pipeline.py:64-141` -- validation uses Apriori output as reference
- `src/evaluation/eval_finetuned_model.py:175-249` -- evaluation oracle is the same algorithm
