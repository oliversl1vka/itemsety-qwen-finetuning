# ADR-003: LoRA vs Full Fine-tuning

**Status:** Accepted  
**Date:** 2026-02

## Context

Choose how to adapt the 7B model weights to the itemset extraction task.

## Options Considered

| Option | VRAM Required | Storage | Quality |
|--------|--------------|---------|---------|
| Full fine-tuning | >100 GB (7B params x 4 bytes x optimizer states) | ~14 GB | Highest ceiling |
| **QLoRA (4-bit + LoRA)** | ~6-8 GB | ~65 MB | Near full-FT quality |
| LoRA (16-bit base) | ~20 GB | ~65 MB | Same as QLoRA |

## Decision

**QLoRA** -- 4-bit NF4 quantized base model with LoRA adapter layers.

## Rationale

Full fine-tuning requires storing the full model weights, gradients, and optimizer states (Adam stores 2 momentum tensors per parameter). For a 7B model this exceeds 100 GB of VRAM -- impossible on any single consumer or cloud GPU.

QLoRA trains small rank-decomposed weight matrices (LoRA adapters) while keeping the base model frozen in 4-bit. This reduces trainable parameters to ~0.05% of total, VRAM to ~6 GB, and adapter storage to ~65 MB. Research shows LoRA quality approaches full fine-tuning for instruction-following and structured output tasks.

## Trade-offs

- Quality ceiling slightly below full fine-tuning for complex tasks
- Requires loading the base model separately during inference
- 4-bit quantization introduces small precision loss in frozen layers

## Source Evidence

- `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 7 -- `FastLanguageModel.get_peft_model()`
- LoRA config: r=32, alpha=64, target_modules=[q,k,v,o,gate,up,down]
