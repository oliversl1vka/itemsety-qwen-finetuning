# ADR-007: SFT Before DPO (Phase Ordering)

**Status:** Accepted  
**Date:** 2026-02

## Context

Should the model be trained with DPO directly from the base weights, or should SFT establish a foundation first?

## Options Considered

| Strategy | Pros | Cons |
|----------|------|------|
| DPO only (from base) | Single phase, simpler | Base model doesn't know the output format |
| **SFT then DPO** | Solid foundation + refinement | Two phases, more wall time |
| Joint training | Potentially optimal | Complex, less understood |
| RLHF (PPO) | Flexible reward signal | Requires reward model training |

## Decision

**SFT (Phase 1) then DPO (Phase 2)**, using the SFT checkpoint as both the initial policy and frozen reference model for DPO.

## Rationale

The base Qwen2.5-7B model has no knowledge of:

1. The JSON output schema (`[{"itemset": [...], "count": N, "rows": [...]}]`)
2. The `<think>` tag reasoning format
3. The concept of scanning a CSV for co-occurring items

Running DPO on a model that doesn't understand the task format causes catastrophic divergence. The model cannot meaningfully compare "chosen" vs "rejected" completions when it doesn't know what either should look like.

SFT first establishes:

- The output format (JSON array with specific fields)
- The reasoning strategy (column-grouped CoT in `<think>` tags)
- Basic task comprehension (scanning rows, counting co-occurrences)

DPO then operates as a refinement step on an already-capable model, with a much smaller distribution shift.

## Trade-offs

- Two-phase training doubles wall time
- The SFT model's quality ceiling constrains what DPO can achieve
- If SFT overfits, DPO starts from a worse reference point

## Source Evidence

- `notebooks/training_3phase_2026-03-09_v3.ipynb` -- SFT cells before DPO cells
- `obsidian-brain/Experiments/2026-03-09 Council v3 Plan.md` -- Council analysis confirming this approach
- Evaluation results: Base (1.0% F1) vs SFT (12.6% F1) demonstrates SFT's critical role
