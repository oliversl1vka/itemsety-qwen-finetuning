# ADR-014: DPO Hyperparameters

**Status:** Accepted  
**Date:** 2026-03-09

## Context

DPO training is more sensitive to hyperparameters than SFT because it optimizes a relative preference between two completions, not an absolute target.

## Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 5e-5 | Standard DPO recommendation; lower than SFT to prevent overshooting |
| Beta | 0.1 | KL penalty weight -- controls how far the policy can deviate from the SFT reference |
| Epochs | 1 | Reduced from 2 in v2; DPO can overfit to zero loss without genuine generalization |
| Batch size | 1 | DPO processes chosen+rejected pairs, effectively doubling memory per step |
| Gradient accumulation | 4 | Effective batch = 4 |

## Key Decisions

**Beta = 0.1**: This is the TRL default and represents a moderate constraint. Higher beta (e.g., 0.5) would keep the model very close to the SFT policy, limiting DPO's ability to learn preferences. Lower beta (e.g., 0.01) would allow larger policy shifts, risking mode collapse. At 0.1, the model has room to learn preferences while staying grounded in SFT behavior.

**1 epoch**: DPO loss can reach near-zero without the model actually generalizing -- it simply memorizes which completion is preferred for each training prompt. With 606 pairs and 1 epoch, each example is seen exactly once, minimizing this memorization risk.

**Learning rate = 5e-5**: DPO's loss landscape is steeper than SFT's because small policy changes create large probability ratio swings. A lower learning rate prevents the optimizer from overshooting.

## Source Evidence

- `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 2 -- DPO config section
