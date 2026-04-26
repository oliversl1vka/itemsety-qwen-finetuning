# ADR-005: 4-bit NF4 Quantization

**Status:** Accepted  
**Date:** 2026-02

## Context

The 7B model must fit in GPU memory during training alongside LoRA adapters, gradients, and optimizer states. Quantization reduces the base model's memory footprint.

## Options Considered

| Format | Bits | VRAM (7B) | Quality Loss | Notes |
|--------|------|-----------|-------------|-------|
| bf16 (full precision) | 16 | ~14 GB | None | Doesn't fit with training overhead |
| INT8 | 8 | ~7 GB | Minimal | Less VRAM savings |
| **NF4 (Normal Float 4)** | 4 | ~3.5 GB | Small | Optimal for normally-distributed weights |
| FP4 | 4 | ~3.5 GB | Small | Worse than NF4 for neural network weights |

## Decision

**4-bit NF4** via BitsAndBytes with bfloat16 compute dtype.

## Rationale

NF4 (Normal Float 4) is information-theoretically optimal for quantizing values drawn from a normal distribution. Neural network weights are approximately normally distributed, making NF4 the best 4-bit format for model compression.

At 4-bit, the base model occupies ~3.5 GB, leaving ample headroom for LoRA adapters (~200 MB), gradients, and optimizer states on a 16 GB T4 or 40 GB A100. The Unsloth framework provides optimized CUDA kernels for 4-bit operations, achieving 2x training speedup over standard implementations.

## Trade-offs

- 4-bit base model cannot be cleanly merged with LoRA adapters (see [ADR-020](adr-020-adapter-only-push.md))
- Slight precision loss in the frozen base layers
- Requires BitsAndBytes library (CUDA-only, no CPU or MPS support)

## Source Evidence

- `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 2 -- `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`
- Model ID: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (pre-quantized)
