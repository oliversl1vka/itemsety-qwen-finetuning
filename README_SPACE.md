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
hardware: t4-small
---

# Qwen2.5 Fine-Tuning for Itemset Extraction

Fine-tune Qwen2.5-3B on the [itemset-extraction-v2](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2) dataset.

## What it does

Trains a language model to extract frequent itemsets from transaction data using:
- **Dataset**: 488 training examples with real-world column names
- **Model**: Qwen2.5-3B-Instruct (high quality results)
- **Method**: Supervised Fine-Tuning (SFT) with 4-bit LoRA
- **Hardware**: NVIDIA T4 Small (paid GPU, 16GB VRAM)

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
## Training Modes

### Test Mode (50 examples)
- **Duration**: ~10-15 minutes
- **Output**: `OliverSlivka/qwen2.5-3b-itemset-test`
- **Purpose**: Quick validation before full training
  
### Full Mode (439 examples, 3 epochs)
- **Duration**: ~40-60 minutes  
- **Output**: `OliverSlivka/qwen2.5-3b-itemset-extractor`
- **Target**: 80-90% valid JSON (vs 6.7% from 0.5B baseline)
- **Cost**: ~$0.60 on T4 Small

**Technical Details:**
- LoRA rank 16, alpha 32
- Batch size 2, gradient accumulation 8 (effective batch 16)
- 4-bit quantization (QLoRA) - efficient training, proven results
- FP16 precision (T4 compatible)

## Notes

Both modes use **4-bit quantization** for:
- ✅ Faster training (lower memory = faster iteration)
- ✅ Lower cost (~30% faster = ~30% cheaper)
- ✅ Proven effective for LoRA fine-tuning
- ✅ No quality loss vs full precision LoRA

Paid T4 GPU ($0.60/hour) provides consistent performance without time limits.

## Dataset

Training data: https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2

## Project

Full pipeline: https://github.com/OliverSlivka/itemsety_real_training
