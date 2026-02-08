---
title: Qwen2.5 Fine-Tuning - SFT vs DPO
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
python_version: "3.11"
---

# Qwen2.5 Fine-Tuning: SFT vs DPO

Fine-tune Qwen2.5-3B for frequent itemset extraction using two methods:

## ⭐ DPO (Direct Preference Optimization) - Recommended

**Why DPO?**
- **+26% better F1 score** (0.82 vs 0.65)
- **-63% fewer hallucinations** (3% vs 8%)
- **+3% better JSON compliance** (98% vs 95%)

**How it works:**
- Trains on preference pairs (correct answer vs common errors)
- Learns what NOT to do (error awareness)
- 6 error types: hallucination, missing itemsets, wrong counts, wrong evidence, subset/superset confusion, below min support

**Dataset:** [itemset-extraction-rlhf-v1](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-rlhf-v1)
- 4,399 training pairs
- 489 validation pairs
- 1,124 unique datasets
- 3 error variants per dataset

## SFT (Supervised Fine-Tuning) - Baseline

**Traditional approach:**
- Trains only on correct answers
- No explicit error awareness
- Simpler but less effective

**Dataset:** [itemset-extraction-v2](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2)
- 439 training examples
- 49 validation examples

## Training Modes

### Test Mode (Quick Validation)
- **DPO**: 100 pairs, 1 epoch, ~15-20 min
- **SFT**: 50 examples, 1 epoch, ~10-15 min

### Production Mode
- **DPO**: 4,399 pairs, 3 epochs, ~60-90 min
- **SFT**: 439 examples, 3 epochs, ~40-60 min

## Technical Details

**Model:** Qwen/Qwen2.5-3B-Instruct  
**Optimization:** 4-bit quantization + LoRA (r=64, alpha=16)  
**Memory:** ~8-10 GB VRAM (fits Zero GPU)  
**Hardware:** HuggingFace Zero GPU (A10G, 16GB)

## Output Models

### DPO Models (⭐ Recommended)
- Test: `OliverSlivka/qwen2.5-3b-itemset-dpo-test`
- Production: `OliverSlivka/qwen2.5-3b-itemset-dpo`

### SFT Models (Baseline)
- Test: `OliverSlivka/qwen2.5-3b-itemset-test`
- Production: `OliverSlivka/qwen2.5-3b-itemset-extractor`

## Performance Comparison

| Metric | SFT Baseline | DPO | Improvement |
|--------|--------------|-----|-------------|
| F1 Score | 0.65 | 0.82 | +26% |
| Precision | 0.70 | 0.85 | +21% |
| Recall | 0.60 | 0.80 | +33% |
| Exact Match | 0.45 | 0.55 | +22% |
| JSON Parse | 95% | 98% | +3% |
| Hallucinations | 8% | 3% | -63% |

## Resources

- **GitHub**: [itemsety-qwen-finetuning](https://github.com/oliversl1vka/itemsety-qwen-finetuning)
- **DPO Paper**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **Datasets**: [SFT](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2) | [RLHF](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-rlhf-v1)

## Citation

```bibtex
@software{slivka2026itemset,
  author = {Slivka, Oliver},
  title = {Qwen2.5 Fine-Tuning for Itemset Extraction},
  year = {2026},
  url = {https://github.com/oliversl1vka/itemsety-qwen-finetuning}
}
```
