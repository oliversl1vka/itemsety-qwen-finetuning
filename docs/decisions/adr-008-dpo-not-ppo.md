# ADR-008: DPO Algorithm Choice

**Status:** Accepted  
**Date:** 2026-02

## Context

Select a preference alignment algorithm for Phase 2 training.

## Options Considered

| Algorithm | Reward Model | Data Format | Stability | Complexity |
|-----------|-------------|-------------|-----------|------------|
| PPO (RLHF) | Required (expensive) | Reward signal | Unstable | High |
| **DPO** | Not needed | Preference pairs | Stable | Low |
| KTO | Not needed | Binary signal | Moderate | Low |
| ORPO | Not needed | Preferences + SFT | Moderate | Medium |
| IPO | Not needed | Preference pairs | Moderate | Low |

## Decision

**Standard DPO** (Direct Preference Optimization) via TRL `DPOTrainer`.

## Rationale

- **No reward model needed**: PPO requires training a separate reward model on preference data, then using it to generate rewards during RL training. This doubles the training infrastructure and introduces instability from reward model errors. DPO eliminates this entirely.

- **Data naturally fits DPO format**: Our data is already paired -- each DPO example has one "chosen" (Apriori correct) and one "rejected" (real LLM failure) completion for the same prompt. This is exactly what DPO is designed for.

- **Stable training**: DPO loss is a binary cross-entropy variant, well-understood and numerically stable. PPO has notoriously unstable training dynamics (reward hacking, KL divergence explosions).

- **Mature implementation**: TRL's `DPOTrainer` is well-tested and compatible with Unsloth's optimizations.

## Trade-offs

- DPO can be unstable when chosen/rejected completions are very similar in probability
- Cannot optimize arbitrary reward functions (only binary preferences)
- Requires paired data (not all real-world preference data is naturally paired)

## Source Evidence

- `src/training/export_real_dpo_data.py` -- generates DPO-formatted pairs
- `notebooks/training_3phase_2026-03-09_v3.ipynb` DPO training section
