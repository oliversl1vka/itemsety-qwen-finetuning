# ADR-002: Qwen2.5-7B Model Selection

**Status:** Accepted  
**Date:** 2026-02

## Context

A base model must be selected for fine-tuning, subject to a hardware constraint: training must fit on a single GPU (T4 16GB or A100 40GB) with 4-bit quantization.

## Options Considered

| Option | Size | 4-bit VRAM | Pros | Cons |
|--------|------|------------|------|------|
| Qwen2.5-3B-Instruct | 3B | ~3 GB | Fast, fits anywhere | Insufficient capacity for structured JSON |
| **Qwen2.5-7B-Instruct** | 7B | ~6 GB | Strong reasoning, native think tags | -- |
| Qwen2.5-14B-Instruct | 14B | ~10 GB | Better quality | Too large for T4 with LoRA overhead |
| Llama-3-8B-Instruct | 8B | ~6 GB | Strong community | No native think tag support |
| Mistral-7B-Instruct | 7B | ~6 GB | Fast inference | Weaker on structured output |

## Decision

**`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`** -- the Unsloth pre-quantized 4-bit NF4 variant.

## Rationale

- **7B is the efficiency frontier** for single-GPU deployment. 14B would require more VRAM headroom than T4 provides with LoRA training overhead.
- **Qwen2.5-Instruct** has native `<think>` tag support from its instruct tuning, matching our CoT training format.
- **Unsloth pre-quantized format** enables 2x training speedup through optimized CUDA kernels.
- **Qwen2.5-7B benchmarks** show stronger reasoning performance than Llama-3 and Mistral at the same parameter count on structured output tasks.

## Trade-offs

- 14B would produce higher-quality outputs but doesn't fit the hardware constraint
- Qwen's tokenizer (152K vocab) is larger than Llama's, adding slight overhead
- Unsloth lock-in: the pre-quantized format works best with the Unsloth framework

## Source Evidence

- `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 2 -- model_id configuration
- `requirements.txt` -- unsloth dependency
