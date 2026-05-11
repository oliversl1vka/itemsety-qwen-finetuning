# ADR-015: Sequence Length 4096 (Restored)

**Status:** Accepted (amended v3.7)  
**Date:** 2026-03-09 (initial v3 change); amended 2026-03-18 (v3.7 restoration)

## Context

The maximum sequence length determines how much text the model can process per training example. Longer sequences require quadratically more memory in the attention mechanism.

## Options Considered

| Length | VRAM (attention) | Coverage | v3 CoT fit |
|--------|-----------------|----------|------------|
| 1024 | ~25% of 4096 | Too short for larger datasets | Some truncation |
| 2048 | ~25% of 4096 | Good coverage | ~70% headroom |
| **4096** | Baseline | Full coverage | Avoids truncation |
| 8192 | 4x of 4096 | Unnecessary | Impossible on T4 |

## Decision

**4096 tokens** in v3 (restored from the initial v3 reduction to 2048).

The initial v3 decision reduced `max_seq_length` from 4096 to 2048, reasoning that the concise CoT format (800--1200 tokens per example) made 4096 wasteful. However, this was reversed in v3.7: some SFT examples with CoT reasoning *plus* the full JSON target exceeded 2048 tokens and were being truncated during training, corrupting the learning signal. The final v3 notebook (`training_3phase_2026-03-09_v3.ipynb` Cell 2, v3.7 comment) sets `max_seq_length=4096`.

## Rationale

The v3 concise CoT format (column-grouped scanning) generates approximately 800--1200 tokens for the reasoning trace alone, but the full training example also includes the system prompt, user message, and JSON output array. For datasets with many frequent itemsets, the combined sequence exceeds 2048 tokens. Restoring 4096 prevents silent truncation of training targets.

The SFT data generation path includes a `--max-tokens 3500` filter for excluding examples that are too long for the training budget. The final exported v3 JSON contains 272 examples. A stale sidecar report next to the JSON records 334 examples from an earlier generation state, so the report must not be used as the final v3 dataset count.

## Trade-offs

- Higher VRAM usage than 2048 due to quadratic attention scaling
- Some headroom is unused for smaller examples, but prevents truncation on larger ones

## Source Evidence

- `src/training/generate_cot_sft_data.py` -- `--max-tokens 3500` filter and report writer
- `data/sft_cot_v3.json` -- final exported SFT-CoT v3 JSON with 272 examples
- `data/sft_cot_v3.report.json` -- stale sidecar report with 334 examples from an earlier generation state
- `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 2 -- `max_seq_length=4096` (v3.7: restored from 2048)
