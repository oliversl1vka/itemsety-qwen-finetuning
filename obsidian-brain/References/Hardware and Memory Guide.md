# Hardware and Memory Guide

**Date:** 2026-03-18  
**Source:** HuggingFace Skills `hardware_guide.md` + project training experience  
**Tags:** #reference #hardware #vram #gpu #memory

---

## Memory Estimation Formulas

### Full Fine-Tuning
```
Memory (GB) ≈ (Model params in billions) × 20
```

### LoRA/QLoRA Fine-Tuning
```
Memory (GB) ≈ (Model params in billions) × 4
```

### Examples
| Model | Full FT | LoRA | QLoRA (4-bit) |
|-------|---------|------|---------------|
| Qwen2.5-0.5B | ~10GB | ~2GB | ~2GB |
| Qwen2.5-1.5B | ~30GB | ~6GB | ~4GB |
| Qwen2.5-3B | ~60GB | ~12GB | ~8GB |
| **Qwen2.5-7B** | ~140GB ❌ | ~28GB | **~16GB** ✅ |
| Qwen2.5-14B | ~280GB ❌ | ~56GB | ~32GB |

---

## GPU Selection Guide

### By Model Size (LoRA/QLoRA)

| Model Size | Minimum GPU | Recommended GPU | VRAM |
|-----------|-------------|-----------------|------|
| < 1B | T4 | T4 / L4 | 16GB |
| 1-3B | T4 | A10G | 16-24GB |
| **7-8B** | **A10G** | **A100 / H200** | **24-80GB** |
| 13B | A10G-large | A100 | 24-40GB |
| 30B+ | A100-80GB | H100 / H200 | 80GB+ |

### Our Server (TLJH Jupyter)
- **GPU:** NVIDIA H200 NVL (141GB HBM3e × 2)
- **Selection:** `CUDA_VISIBLE_DEVICES=1` (avoid device 0 conflicts with other users)
- **Capability:** Qwen2.5-7B in 4-bit = ~16GB (well within H200's 141GB)

### Unsloth VRAM Requirements (4-bit)

| Model | Min VRAM (Unsloth 4-bit) | Recommended GPU |
|-------|--------------------------|-----------------|
| 2B-4B | 8GB | T4, L4 |
| **7B-8B** | **16GB** | **A10G, L4×4** |
| 13B | 24GB | A10G-large |
| 30B+ | 48GB+ | A100 |

---

## Memory Optimization Techniques

### Priority Order (try these in sequence)

1. **Use LoRA/PEFT** (most impactful)
   ```python
   from peft import LoraConfig
   peft_config = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear")
   ```

2. **Reduce batch size**
   ```python
   per_device_train_batch_size = 1
   ```

3. **Increase gradient accumulation** (maintain effective batch size)
   ```python
   gradient_accumulation_steps = 8  # Effective batch = 1 × 8
   ```

4. **Enable gradient checkpointing**
   ```python
   gradient_checkpointing = True
   # With Unsloth:
   use_gradient_checkpointing = "unsloth"  # 30% VRAM savings
   ```

5. **Use mixed precision**
   ```python
   bf16 = True  # preferred on modern GPUs (A10G+)
   # fp16 = True  # fallback for older GPUs (T4)
   ```

6. **4-bit quantization** (QLoRA)
   ```python
   from transformers import BitsAndBytesConfig
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.bfloat16,
       bnb_4bit_use_double_quant=True,
   )
   ```

7. **Unsloth optimizations** (~60% less VRAM, 2× faster)
   ```python
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="unsloth/Qwen2.5-7B-Instruct",
       max_seq_length=2048,
       load_in_4bit=True,
   )
   ```

8. **Disable evaluation during training** (saves ~40% memory)
   ```python
   eval_strategy = "no"  # Enable only when you have enough VRAM
   ```

---

## Memory Cleanup Between Training Phases

Critical when running SFT → DPO sequentially:
```python
import gc, torch

# After each phase
del model, trainer
gc.collect()
torch.cuda.empty_cache()

# Verify cleanup
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
```

---

## Common OOM Scenarios and Fixes

| Scenario | Symptom | Fix |
|----------|---------|-----|
| Large batch size | CUDA OOM on first step | `per_device_train_batch_size=1` |
| Long sequences | OOM mid-epoch | Reduce `max_seq_length` (2048→1024) |
| Eval during training | OOM during eval | `eval_strategy="no"` |
| Multiple phases | OOM on phase 2 | Memory cleanup between phases |
| Merged model push | OOM during merge | Save adapter-only, merge on CPU later |
| Flash Attention missing | Slow + high memory | `pip install flash-attn` or use Unsloth |

---

## Our Project Configuration

### Training Setup (v3.2)
```
Model: Qwen2.5-7B-Instruct (Unsloth 4-bit)
GPU: H200 NVL (CUDA_VISIBLE_DEVICES=1)
Peak VRAM: ~16GB (SFT), ~18GB (DPO)
Training time: ~40 min SFT, ~20 min DPO
```

### TLJH Server Checklist
From Training Agent memory — critical steps:
1. `CUDA_VISIBLE_DEVICES=1` — always select GPU 1
2. `pip install unsloth` — if not in base environment
3. Verify: `torch.cuda.is_available()` and `torch.cuda.device_count()`
4. **Keras 3 crash workaround:** `os.environ["TF_USE_LEGACY_KERAS"] = "1"` if tensorflow imported
5. **Never** use `merge_and_unload()` on 4-bit models — save adapter only

---

## Cost Estimation (HuggingFace Jobs)

For reference — our training runs on TLJH, but these costs help understand scale:

| Hardware | Cost/hr | Qwen2.5-7B LoRA (3 epochs) |
|----------|---------|---------------------------|
| T4 (16GB) | $0.60 | ❌ Won't fit 7B |
| A10G (24GB) | $1.05 | ~3-4h, ~$4 |
| A100 (40GB) | $4.00 | ~1-2h, ~$6 |
| A100 (80GB) | $6.50 | ~1h, ~$6.50 |

---

## See Also

- [[Unsloth Notebook Patterns]] — Specific Unsloth optimization patterns
- [[Training Methods Guide]] — Method-specific VRAM considerations
- [[Training Agent]] — Real training logs with VRAM measurements
- [[Training Troubleshooting]] — OOM and hardware issue fixes
