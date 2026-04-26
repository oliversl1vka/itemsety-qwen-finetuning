# ADR-004: LoRA Rank 32

**Status:** Accepted  
**Date:** 2026-03-09 (v3 change)

## Context

LoRA rank (`r`) and scaling factor (`alpha`) control the expressivity of the adapter. Higher rank means more trainable parameters and more capacity to learn, but also more risk of overfitting on a small dataset.

## Options Considered

| r | alpha | alpha/r | Trainable Params | Risk |
|---|-------|---------|-----------------|------|
| 8 | 16 | 2.0 | Very few | Under-capacity |
| 16 | 32 | 2.0 | Few | Possible under-capacity |
| **32** | **64** | **2.0** | Moderate | Balanced |
| 64 | 16 | 0.25 | Many | v2 showed overfitting |
| 128 | 256 | 2.0 | Very many | Severe overfitting risk |

## Decision

**r=32, alpha=64** (ratio 2.0). Changed from r=64, alpha=16 in v2.

## Rationale

v2 used r=64 with alpha=16 (ratio 0.25). With only 348 training examples (v2), the model showed signs of overfitting: training loss dropped rapidly while validation metrics stagnated.

The v3 change:

- **r=32** reduces the adapter's capacity, acting as implicit regularization
- **alpha=64** increases the scaling factor to maintain the standard alpha/r=2.0 ratio, ensuring that the smaller adapter has proportionally stronger updates per step
- Combined with other v3 regularization changes (dropout=0.05, weight_decay=0.01), this addresses the overfitting observed in v2

This change was recommended by the LLM Council analysis on 2026-03-09.

## Trade-offs

- Slightly less capacity to learn complex patterns
- If the training dataset grows significantly, r=32 might become the bottleneck

## Source Evidence

- `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 2 -- v3 config with change comments
- `obsidian-brain/Experiments/2026-03-09 Council v3 Plan.md` -- Council recommendation
