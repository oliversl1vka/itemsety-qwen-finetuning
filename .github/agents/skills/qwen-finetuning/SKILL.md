---
name: qwen-finetuning
description: Fine-tune Qwen2.5-7B models using Unsloth + LoRA for frequent itemset extraction. Supports 3-phase training (SFT-CoT → DPO-Real). Use for model training configuration and execution.
---

# Qwen Model Fine-Tuning

Train Qwen2.5-7B-Instruct to extract frequent itemsets using Unsloth + LoRA.

## Overview

Current training pipeline (v3):
1. Load base model (Qwen2.5-7B-Instruct via Unsloth, 4-bit)
2. Apply LoRA adapters (r=32, alpha=64)
3. **Phase 1: SFT-CoT** — Train on ground truth with `<think>` reasoning
4. **Phase 2: DPO-Real** — Align using real LLM failures as rejected
5. ~~Phase 3: GRPO~~ — Skipped (Council decision, produces F1=0)
6. Save adapter only (NEVER merge 4-bit) and push to HuggingFace Hub

## Prerequisites

### Hardware
- GPU with 16GB+ VRAM (A10G, A100, H200)
- For Qwen2.5-7B with Unsloth 4-bit: ~16GB VRAM
- Our server: H200 NVL (`CUDA_VISIBLE_DEVICES=1`)

### Dependencies
```bash
pip install unsloth torch transformers peft trl bitsandbytes datasets accelerate
```

### HuggingFace Token
```bash
export HF_TOKEN=hf_your_token_here
```

## Model Selection

| Model | VRAM (4-bit LoRA) | Training Time | Quality |
|-------|-------------------|---------------|---------|
| Qwen2.5-0.5B | ~2GB | 15 min | Low |
| Qwen2.5-1.5B | ~4GB | 25 min | Medium |
| Qwen2.5-3B | ~8GB | 45 min | Good |
| **Qwen2.5-7B** | **~16GB** | **40-60 min** | **Best** ✅ |

## Configuration (v3.2 — Council + Diamond Knowledge)

### LoRA Configuration
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,                        # Higher rank for structured extraction (Council)
    lora_alpha=64,               # Ratio 2.0 (Council: original 0.25 was too low)
    lora_dropout=0.05,           # Regularization against repetition loops
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_gradient_checkpointing="unsloth",  # 30% VRAM savings
    random_state=42,
)
```

### SFT Training Arguments
```python
from trl import SFTConfig

sft_config = SFTConfig(
    output_dir="./sft-output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,          # Half of Unsloth default (Council: prevent repetition)
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_seq_length=2048,
    packing=False,               # Protect label masking boundaries (Council)
    weight_decay=0.01,
    seed=42,
)
```

### DPO Training Arguments
```python
from trl import DPOConfig

dpo_config = DPOConfig(
    output_dir="./dpo-output",
    num_train_epochs=1,          # DPO needs fewer epochs
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,          # DPO uses MUCH lower LR than SFT
    beta=0.1,                    # KL penalty coefficient
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    optim="paged_adamw_8bit",
    max_length=2048,
    seed=42,
)
```

## Training Workflow

### 1. Load Model with Unsloth
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)
```

### 2. Phase 1: SFT-CoT
```python
from trl import SFTTrainer
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=sft_dataset,
    args=sft_config,
)

# Mask system/user tokens — train only on assistant responses
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# ⚠️ VERIFY label masking before training
labels = trainer.train_dataset[0]["labels"]
assert any(l == -100 for l in labels), "Masking not working!"

trainer.train()
```

### 3. Memory Cleanup Between Phases
```python
import gc, torch
del trainer
gc.collect()
torch.cuda.empty_cache()
```

### 4. Phase 2: DPO-Real
```python
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    train_dataset=dpo_dataset,
    args=dpo_config,
)
dpo_trainer.train()
```

### 5. Save Adapter Only
```python
# ✅ CORRECT: Save adapter only
model.save_pretrained("./final-adapter")
tokenizer.save_pretrained("./final-adapter")

# Push to Hub (versioned repo — NEVER overwrite old versions)
model.push_to_hub("OliverSlivka/qwen2.5-7b-itemset-v3")
tokenizer.push_to_hub("OliverSlivka/qwen2.5-7b-itemset-v3")

# ❌ NEVER DO THIS on 4-bit models:
# model.merge_and_unload()  → corrupts weights (v1 catastrophic bug)
```

## Inference

### Two-Phase Generation (v3)
```python
FastLanguageModel.for_inference(model)  # 2× speedup

output = model.generate(
    input_ids,
    max_new_tokens=1024,
    temperature=0.3,
    repetition_penalty=1.2,
    do_sample=True,
)
```

## Training Notebook

The versioned training notebook is at:
- `notebooks/training_3phase_7b.ipynb` (template)
- Generated versioned copies via `training-agent /export`

## Monitoring

### Key Metrics During Training
- **SFT loss < 0.1** → Good convergence
- **DPO loss decreasing** → Alignment working
- **No NaN/Inf** → Stable training

### Warning Signs
- Loss spike > 2× → Reduce learning rate
- Loss flat → Check data, verify masking
- VRAM usage growing → Memory leak, add cleanup

## ⚠️ Critical Lessons (from Training Agent memory)

1. **Never `merge_and_unload()` on 4-bit** — Save adapter only
2. **Always verify label masking** — Check for -100 in labels
3. **Memory cleanup between phases** — `del model; gc.collect(); torch.cuda.empty_cache()`
4. **DPO LR must be much lower** than SFT (5e-7 vs 1e-4)
5. **GRPO is skipped** — Produced F1=0 in v1, not worth the risk
6. **Use `CUDA_VISIBLE_DEVICES=1`** on TLJH server
7. **R-shorthand in CoT** — Use "R1-R4" not "Row 1, Row 2, Row 3, Row 4" (tokenization)
8. **Reliability over performance** — Use proven defaults, don't enable torch_compile

## Troubleshooting

### CUDA OOM (priority order)
1. `per_device_train_batch_size=1`
2. `gradient_accumulation_steps=8` (maintain effective batch)
3. `use_gradient_checkpointing="unsloth"` (30% VRAM savings)
4. Reduce `max_seq_length` (2048 → 1024)
5. Disable evaluation: `eval_strategy="no"`
6. Use smaller LoRA rank: `r=16` instead of `r=32`

### Repetition Loops
- Lower learning rate: `1e-4` → `5e-5`
- Add `repetition_penalty=1.2` at inference
- Add `lora_dropout=0.05`
- Use DPO to teach what NOT to output

### Poor Results
- Verify label masking is working
- Check SFT data quality (CoT + correct JSON)
- Increase training data (300+ examples recommended)
- Try DPO after SFT (real failures as negatives)

### Model Merge Issues
- **Never merge 4-bit** — save adapter only
- For GGUF conversion: reload base in float16, then merge on CPU
