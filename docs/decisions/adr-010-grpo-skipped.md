# ADR-010: GRPO Skipped in v3

**Status:** Accepted  
**Date:** 2026-03-09

## Context

The training pipeline was designed with three phases: SFT, DPO, and GRPO (Group Relative Policy Optimization). Should Phase 3 be executed in v3?

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **Skip GRPO** | Focus on SFT+DPO baseline; avoid over-optimization | Miss potential improvement |
| Run GRPO | Additional refinement | Reward function design non-trivial; risk of degrading DPO checkpoint |

## Decision

**Skip GRPO in v3.** Focus on establishing and evaluating the SFT+DPO baseline first.

## Rationale

The LLM Council analysis on 2026-03-09 recommended against running GRPO at this stage for three reasons:

1. **No established baseline**: Without a properly evaluated SFT+DPO result, adding GRPO conflates multiple variables. If the combined result is poor, it's unclear which phase is responsible.

2. **Reward function design**: GRPO requires a reward function mapping model outputs to scalar scores. F1 score is non-differentiable and discrete -- approximating it as a reward signal risks over-optimization artifacts (the model might learn to exploit reward function quirks rather than genuinely improve).

3. **Catastrophic forgetting risk**: GRPO's RL-based optimization can degrade an otherwise-good DPO checkpoint if the reward signal is noisy or misaligned.

## Trade-offs

- Potentially missing F1 improvements that GRPO could provide
- The GRPO infrastructure (TRL `GRPOTrainer`, reward function, generation sampling) was implemented and is available in the training notebook for future work

## Source Evidence

- `obsidian-brain/Experiments/2026-03-09 Council v3 Plan.md` -- Council recommendation
- `notebooks/training_3phase_2026-03-09_v3.ipynb` -- GRPO cells present but skipped
- `data/hf_dataset_v3/grpo/` -- GRPO-formatted data available for future use
