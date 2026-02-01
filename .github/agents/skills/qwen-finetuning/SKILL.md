---
name: qwen-finetuning
description: Fine-tune Qwen2.5 models using LoRA/QLoRA for frequent itemset extraction. Use for model training with SFT (Supervised Fine-Tuning).
---

# Qwen Model Fine-Tuning

Train Qwen2.5 models to extract frequent itemsets using LoRA/QLoRA.

## Overview

Fine-tuning pipeline:
1. Load base model (Qwen2.5-3B-Instruct)
2. Apply QLoRA (4-bit quantization + LoRA adapters)
3. Train on ChatML examples with CoT
4. Merge and push to HuggingFace Hub

## Prerequisites

### Hardware
- GPU with 16GB+ VRAM (A10G, RTX 4090, A100)
- For 3B model: 10GB VRAM with QLoRA

### Dependencies
```bash
pip install torch transformers peft trl bitsandbytes datasets accelerate
```

### HuggingFace Token
```bash
export HF_TOKEN=hf_your_token_here
# Or in .env file
```

## Quick Start

### Test Mode (10-15 min)
```bash
python src/training/run_sft_test.py
```

### Production Mode (40-60 min)
```bash
python src/training/run_sft_full.py
```

## Configuration

### Model Selection

| Model | VRAM | Training Time | Quality |
|-------|------|---------------|---------|
| Qwen2.5-0.5B | 4GB | 15 min | Low |
| Qwen2.5-1.5B | 6GB | 25 min | Medium |
| Qwen2.5-3B | 10GB | 45 min | Good |
| Qwen2.5-7B | 20GB | 90 min | Best |

### LoRA Configuration
```python
peft_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Alpha scaling
    lora_dropout=0.05,       # Dropout
    target_modules=[         # Layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM"
)
```

### Training Arguments
```python
training_args = SFTConfig(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_seq_length=2048,
)
```

## Training Workflow

### 1. Load Dataset
```python
from datasets import load_dataset

dataset = load_dataset("OliverSlivka/itemset-extraction-dataset")
```

### 2. Initialize Model
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### 3. Apply LoRA
```python
from peft import get_peft_model

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Output: trainable params: 20M || all params: 3B || trainable%: 0.66%
```

### 4. Train
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    args=training_args,
)

trainer.train()
```

### 5. Merge & Push
```python
# Merge LoRA with base
merged_model = model.merge_and_unload()

# Push to Hub
merged_model.push_to_hub("OliverSlivka/qwen2.5-3b-itemset-extractor")
tokenizer.push_to_hub("OliverSlivka/qwen2.5-3b-itemset-extractor")
```

## Monitoring Training

### Loss Curve
```python
# In training logs
Step 100: loss=1.234, lr=2e-4
Step 200: loss=0.987, lr=1.8e-4
```

### Validation Metrics
```python
# Compute during eval
eval_loss, perplexity = trainer.evaluate()
```

## Output Artifacts

| Artifact | Location |
|----------|----------|
| Checkpoints | `outputs/checkpoint-*` |
| Final model | `outputs/final` |
| Training logs | `outputs/runs/` |
| Hub model | `OliverSlivka/qwen2.5-3b-itemset-extractor` |

## Troubleshooting

### CUDA OOM
```python
# Reduce batch size
per_device_train_batch_size=1
gradient_accumulation_steps=16

# Enable gradient checkpointing
gradient_checkpointing=True

# Use 8-bit optimizer
optim="paged_adamw_8bit"
```

### Slow Training
- Enable bf16/fp16
- Use Flash Attention 2
- Reduce max_seq_length

### Poor Results
- Increase epochs (3 → 5)
- Add more training data
- Try larger base model
- Adjust LoRA rank (16 → 32)

## Model Card

Push with model card:
```python
model_card = """
---
language: en
tags:
  - frequent-itemset-mining
  - qwen2.5
  - lora
---

# Qwen2.5-3B Itemset Extractor

Fine-tuned to extract frequent itemsets from CSV data.

## Usage
...
"""

with open("README.md", "w") as f:
    f.write(model_card)
```
