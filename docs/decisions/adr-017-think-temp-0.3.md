# ADR-017: Thinking Temperature 0.3

**Status:** Accepted  
**Date:** 2026-03

## Context

The temperature for Phase 1 (reasoning in `<think>` tags) must balance coherent reasoning against repetition loop avoidance.

## Options Considered

| Temperature | Behavior | F1 Impact |
|-------------|----------|-----------|
| 0.0 (greedy) | Deterministic; severe repetition loops | Unusable |
| 0.1 | Near-greedy; frequent attractor states | Poor |
| **0.3** | Mild stochasticity; escapes loops | < 3% F1 loss |
| 0.5 | More varied reasoning; occasional drift | Moderate |
| 0.7+ | High variety; unfocused reasoning | Significant |

## Decision

**Temperature = 0.3** for the thinking phase.

## Rationale

At temperature 0.0 and 0.1, the model enters **deterministic attractor states** -- it picks the same high-probability next token at each step, causing the reasoning to loop indefinitely. For example, the model might generate "checking row 5... checking row 5... checking row 5..." because "checking row 5" has the highest probability at each position.

Temperature 0.3 adds sufficient stochasticity to escape these attractors while keeping the reasoning focused on the task. The model occasionally generates a slightly different reasoning step, breaking the loop and progressing through the analysis.

Council analysis found that **the F1 cost is less than 3%** compared to higher temperatures -- a small price for eliminating infinite loops.

## Trade-offs

- Slight non-determinism: two inference runs on the same input may produce different reasoning traces
- Rare cases where temp=0.3 generates slightly incorrect intermediate reasoning but still produces correct final JSON

## Source Evidence

- `src/evaluation/inference_utils.py:177-179` -- temperature configuration
- Council analysis records in `obsidian-brain/Experiments/`
