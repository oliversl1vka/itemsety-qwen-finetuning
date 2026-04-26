# ADR-015: Sequence Length 2048

**Status:** Accepted  
**Date:** 2026-03-09 (v3 change)

## Context

The maximum sequence length determines how much text the model can process per training example. Longer sequences require quadratically more memory in the attention mechanism.

## Options Considered

| Length | VRAM (attention) | Coverage | v3 CoT fit |
|--------|-----------------|----------|------------|
| 1024 | ~25% of 4096 | Too short for larger datasets | Some truncation |
| **2048** | ~25% of 4096 | Good coverage | ~70% headroom |
| 4096 (v2) | Baseline | Full coverage | Wasteful |
| 8192 | 4x of 4096 | Unnecessary | Impossible on T4 |

## Decision

**2048 tokens** in v3 (reduced from 4096 in v2).

## Rationale

The v3 concise CoT format (column-grouped scanning) generates approximately 800--1200 tokens per training example, as estimated from the `est_tokens` metadata in `sft_cot_v3.json`. A 2048 token budget provides over 70% headroom.

Halving the sequence length from 4096 to 2048 provides significant VRAM savings: attention memory scales quadratically with sequence length, so 2048 uses approximately 4x less attention memory than 4096.

The `--max-tokens 3500` filter in SFT data generation ensures no example exceeds the token budget. This filter reduced the v3 dataset from 348 (v2 count) to 272 examples by excluding examples that were too long even for the concise format.

## Trade-offs

- Examples requiring more than 2048 tokens (large datasets with many itemsets) are excluded from training
- The model may struggle at inference time with inputs longer than what it was trained on

## Source Evidence

- `src/training/generate_cot_sft_data.py:69` -- `--max-tokens 3500` filter
- `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 2 -- `max_seq_length=2048`
