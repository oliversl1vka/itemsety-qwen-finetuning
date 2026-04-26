# ADR-006: Unsloth Framework

**Status:** Accepted  
**Date:** 2026-02

## Context

Choose the training framework layer above HuggingFace Transformers and TRL.

## Options Considered

| Framework | Speed | VRAM | Key Feature | Limitation |
|-----------|-------|------|------------|------------|
| Raw Transformers + PEFT | 1x | Baseline | Full control | Manual optimization |
| Axolotl | ~1.5x | ~80% | Config-driven | Less flexible |
| LLaMA-Factory | ~1.5x | ~80% | Web UI | Python API is secondary |
| **Unsloth** | **2x** | **30%** | Custom CUDA kernels | Black-box internals |

## Decision

**Unsloth** for all training phases.

## Rationale

- **2x training speed** via custom triton/CUDA kernels for attention and LoRA operations
- **70% VRAM reduction** compared to standard PEFT, critical for fitting 7B model on T4
- **`train_on_responses_only`** helper masks the training loss on system/user turns in ChatML format, ensuring only the assistant response (CoT + JSON) is trained. This is non-trivial to implement correctly with raw TRL.
- **`FastLanguageModel.for_inference`** provides an optimized inference path
- **Native Qwen2.5 + BitsAndBytes 4-bit support** with pre-quantized model variants

## Trade-offs

- Black-box CUDA kernels are harder to debug than standard PyTorch operations
- Unsloth-specific API (not standard HuggingFace) creates some framework lock-in
- Less community support than raw Transformers for edge cases

## Source Evidence

- `requirements.txt` -- `unsloth` dependency
- `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 1 -- `from unsloth import FastLanguageModel`
- `src/evaluation/eval_finetuned_model.py:84-146` -- `FastLanguageModel.from_pretrained()` for inference
