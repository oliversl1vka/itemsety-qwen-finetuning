# ADR-009: Real LLM Failures for DPO

**Status:** Accepted  
**Date:** 2026-03

## Context

DPO requires "rejected" completions -- outputs the model should learn to avoid. How should these be generated?

## Options Considered

| Strategy | Pros | Cons |
|----------|------|------|
| Synthetic corruption (random flip/delete) | Easy to generate at scale | Doesn't match real LLM error patterns |
| Model self-samples at high temperature | Matches model behavior | Requires expensive generation runs |
| **Real LLM failures from pipeline runs** | Matches actual failure modes | Requires running pipeline on multiple models |

## Decision

**Real LLM failures** collected from `pipeline.py` runs on 4 different models.

## Rationale

Analysis of 1600+ pipeline runs revealed that **99.5% of real LLM errors are `item_missing_in_row`**: the model confidently asserts that an item appears in a specific row when it does not.

This is qualitatively different from what synthetic corruption produces:

- **Random item flipping** generates uniformly distributed errors across error types
- **Random row deletion** generates count-mismatch errors
- **Neither matches the real pattern**: confident hallucination of evidence rows

Training against real failures teaches the model to recognize and avoid its actual behavioral failure mode, not arbitrary noise patterns.

**Source models for rejected outputs:**

- GPT-4.1-mini
- GPT-4.1-nano
- GPT-4o
- o4-mini

## Trade-offs

- Requires running a costly extraction pipeline across multiple models to collect failures
- Failure patterns may be partially model-specific (GPT-4.1-mini's failures might differ from smaller models)
- Maximum 3 rejected outputs per dataset prevents over-representation but limits data volume

## Source Evidence

- `src/training/export_real_dpo_data.py:11-14` -- docstring documenting the 99.5% error distribution
- `pipeline.py:64-141` -- validation logic that classifies `item_missing_in_row`

## Consequences

This decision directly contributed to the zero hallucination rate achieved by both SFT and DPO models: training against real evidence-hallucination failures effectively taught the model to ground its outputs in the actual CSV data.
