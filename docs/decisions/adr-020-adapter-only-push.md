# ADR-020: Adapter-Only Model Push

**Status:** Accepted  
**Date:** 2026-03

## Context

How should the fine-tuned model be published to HuggingFace Hub?

## Options Considered

| Strategy | Size | Reproducibility | Ease of Use |
|----------|------|----------------|-------------|
| Merge LoRA into base, push full weights | ~14 GB | Lossy (merge artifact) | Load directly |
| **Push LoRA adapter only** | ~65 MB | Exact | Requires base model + adapter load |
| Push GGUF quantized | ~4 GB | Lossy (double quantization) | Easy with llama.cpp |

## Decision

**Push LoRA adapter weights only** (~65 MB) to HuggingFace Hub.

## Rationale

Unsloth's `merged_4bit_forced` method -- which merges LoRA adapters back into a 4-bit NF4 base model -- **destroys the adapter structure**. The merge operation with NF4 quantization produces weights that are not cleanly invertible: the merged model cannot be decomposed back into base + adapter, and the merged weights may differ from the "true" merged model due to quantization artifacts.

Adapter-only push is **exact and deterministic**: loading `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` as the base and applying the adapter produces the exact same model every time. This is critical for reproducibility -- anyone can recreate the exact inference setup.

The model is published at [OliverSlivka/qwen2.5-7b-itemset-extractor](https://huggingface.co/OliverSlivka/qwen2.5-7b-itemset-extractor).

## Trade-offs

- Users must load the base model separately (additional download)
- Requires Unsloth or PEFT library for adapter application
- Cannot be used directly with tools that expect full model weights (e.g., vLLM without adapter support)

## Source Evidence

- `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 20 -- comment about `merged_4bit_forced` failure
- `src/evaluation/eval_finetuned_model.py:84-146` -- adapter loading implementation
