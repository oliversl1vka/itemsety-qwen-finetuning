# ADR-013: SFT Hyperparameters

**Status:** Accepted  
**Date:** 2026-03-09 (v3 changes from v2)

## Context

Each hyperparameter in the SFT configuration was adjusted in v3 based on v2 training observations and Council analysis.

## Changes from v2 to v3

| Parameter | v2 | v3 | Rationale |
|-----------|----|----|-----------|
| Learning rate | 2e-4 | **1e-4** | 2e-4 caused rapid loss drop with validation stagnation (overfitting) |
| Epochs | 2 | **3** | More passes help the model converge on a small dataset (272 examples) |
| Warmup ratio | 0.05 | **0.10** | Larger warmup stabilizes the learning rate schedule on small datasets |
| Weight decay | 0 | **0.01** | L2 regularization added as anti-overfitting measure |
| Sequence length | 4096 | **4096** | Initially reduced to 2048 in v3, restored in v3.7 to avoid truncating CoT+JSON targets |
| LoRA r | 64 | **32** | Reduced adapter capacity (see [ADR-004](adr-004-lora-rank-32.md)) |
| LoRA alpha | 16 | **64** | Maintain alpha/r=2.0 scaling ratio |
| LoRA dropout | 0 | **0.05** | Stochastic regularization during training |

## Rationale

The v2 configuration exhibited clear overfitting symptoms: training loss decreased rapidly while per-dataset evaluation metrics showed no corresponding improvement. This is expected with only ~300 training examples -- the model memorizes rather than generalizes.

The v3 changes collectively address this through multiple regularization strategies:

1. **Lower learning rate** (1e-4) slows the optimization, giving the model more gradient steps to converge
2. **More epochs** (3) compensates for the lower learning rate with more total updates
3. **Larger warmup** (0.10) prevents the first few batches from pushing weights too far
4. **Weight decay** (0.01) adds explicit L2 regularization
5. **LoRA dropout** (0.05) adds stochastic regularization
6. **Smaller LoRA rank** (32) reduces the adapter's raw capacity to memorize

These changes work in concert: individually each is minor, but together they shift the training dynamics from "fast memorization" to "slower generalization."

## Source Evidence

- `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 2 -- all values with change comments
- `obsidian-brain/Experiments/2026-03-09 Council v3 Plan.md` -- Council analysis
