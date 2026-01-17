---
title: Qwen Fine-Tuning
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.13.0"
app_file: app.py
pinned: false
license: mit
hardware: zero-a10g
---

# Qwen2.5 Fine-Tuning for Itemset Extraction

This Space fine-tunes Qwen2.5-0.5B-Instruct on the [itemset-extraction-v2](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2) dataset.

## What it does

Trains a language model to extract frequent itemsets from transaction data using:
- **Dataset**: 488 training examples with real-world column names
- **Model**: Qwen2.5-3B-Instruct (high quality results)
- **Method**: Supervised Fine-Tuning (SFT) with 4-bit LoRA
- **Hardware**: Zero GPU A10G (free GPU access)

## How to use

1. Select training mode (test or full)
2. Click "Submit" to start training
3. Watch logs stream in real-time
4. Trained model will be pushed to HuggingFace Hub

## Training Configuration

### Test Mode (50 examples)
- **Model**: Qwen2.5-3B-Instruct
- **LoRA rank**: 16
- **Batch size**: 2 (effective 16 with gradient accumulation)
- **Duration**: ~10-15 minutes
- **Output**: `OliverSlivka/qwen2.5-3b-itemset-test`
Notes

This Space supports two training modes:

- **Test Mode**: Quick validation with 50 examples (~10-15 min)
  - Verifies setup works on Zero GPU
  - Pushes to test repo for inspection
  
- **Full Mode**: Production training with 439 examples, 3 epochs (~40-60 min)
  - Target: 80-90% valid JSON (vs 6.7% from 0.5B baseline)
  - Final model for real-world use

Both modes use **Qwen2.5-3B with 4-bit quantization** - fits perfectly in Zero GPU's 16GB memory!
## Notes

This is a **test run** with 50 training examples to verify the setup works with Zero GPU. 

For production training:
- Use full 439-example training set
- Train for 2-3 epochs (~200 steps)
- Consider using Qwen2.5-3B or 7B for better results (requires paid GPU)

## Dataset

Training data: https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2

## Project

Full pipeline: https://github.com/OliverSlivka/itemsety_real_training
