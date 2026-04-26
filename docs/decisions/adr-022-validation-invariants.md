# ADR-022: 13 Validation Invariants

**Status:** Accepted  
**Date:** 2026-02

## Context

LLM outputs for itemset extraction can fail in many ways: malformed JSON, hallucinated items, wrong counts, invalid row references. A comprehensive validation system is needed to classify these failures precisely.

## Decision

**13 invariant checks** covering parse, structure, canonicalization, and semantic correctness.

## Invariant Categories

**Parse invariants:**

1. Output is valid JSON
2. Output is a JSON array
3. Each element is an object with required keys

**Structural invariants:**

4. `itemset` is a non-empty list of strings
5. `count` is a positive integer
6. `rows` is a non-empty list of strings

**Canonicalization invariants:**

7. Items are lowercase and trimmed
8. Items within each itemset are sorted alphabetically
9. Row references follow "Row N" format (1-based)

**Semantic invariants:**

10. Each cited item exists in the referenced row of the CSV
11. Each cited row actually exists in the dataset
12. The count matches the number of evidence rows
13. The itemset size does not exceed max_size

## Rationale

Simple validation (JSON parse + count check) misses the dominant failure mode. The critical invariant is #10: **item existence in row**. This catches `item_missing_in_row` errors -- the model cites an item as appearing in a row where it does not exist. This error type accounts for 99.5% of all real LLM failures.

The 13 invariants were determined empirically by observing all failure modes encountered across 1600+ pipeline runs. Each invariant corresponds to a real failure type that occurred at least once in production runs.

## Consequences

The validation system serves three purposes:

1. **Pipeline quality gate**: every pipeline run gets a `validation_passed` flag in runs.db
2. **DPO data source**: runs with `validation_passed=0 AND llm_itemset_count > 0` become rejected examples
3. **Evaluation ground truth**: the same invariants validate model outputs during evaluation

## Source Evidence

- `pipeline.py:64-141` -- `validate_all()` and individual check functions
