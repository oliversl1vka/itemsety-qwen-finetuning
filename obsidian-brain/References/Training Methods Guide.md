# Training Methods Guide

**Date:** 2026-03-18  
**Source:** HuggingFace Skills `hugging-face-model-trainer` + project experience  
**Tags:** #reference #training #sft #dpo #grpo

---

## Method Overview

| Method | Complexity | Data Required | Use Case |
|--------|-----------|---------------|----------|
| **SFT** | Low | Demonstrations (messages) | Initial fine-tuning, teach new capabilities |
| **DPO** | Medium | Paired preferences (chosen/rejected) | Post-SFT alignment, improve quality |
| **GRPO** | Medium-High | Prompts + reward function | Online RL with automatic rewards |
| **Reward Modeling** | Medium | Paired preferences | Build reward model for RLHF pipeline |

---

## Supervised Fine-Tuning (SFT)

**What it is:** Standard instruction tuning with supervised learning on demonstration data.

**When to use:**
- Initial fine-tuning of base models on task-specific data
- Teaching new capabilities or domains
- Most common starting point for fine-tuning

**Dataset format:** Conversational format with "messages" field (ChatML)

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="sft-model",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,        # Council recommendation for structured extraction
        eval_strategy="no",
    )
)
trainer.train()
```

### SFT with CoT (Our Approach)
Our SFT uses Chain-of-Thought reasoning in `<think>` tags. The assistant response contains reasoning then the JSON output:
```
<think>
R1-R7 have {Apple, Banana} → count 7, support 7/15=0.467
...
</think>
[{"itemset": ["Apple","Banana"], "count": 7, "rows": ["R1",...], "support": 0.467}]
```

**Key SFT settings for our project (v3.2):**
- `learning_rate=1e-4` (half of Unsloth default, prevents repetition loops)
- `lora_r=32`, `lora_alpha=64` (higher capacity for structured extraction)
- `packing=False` (protect label masking boundaries)
- `train_on_responses_only()` with label verification

---

## Direct Preference Optimization (DPO)

**What it is:** Alignment method using preference pairs (chosen vs rejected) without requiring a reward model.

**When to use:**
- Aligning models to human preferences after SFT
- Improving response quality
- Have paired preference data

**Dataset format:** Preference pairs with "chosen" and "rejected" fields

```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model="path/to/sft-model",   # Use SFT-trained model as base
    train_dataset=dataset,
    args=DPOConfig(
        output_dir="dpo-model",
        beta=0.1,                 # KL penalty coefficient
        num_train_epochs=1,       # DPO typically needs fewer epochs than SFT
        per_device_train_batch_size=1,
        learning_rate=5e-7,       # DPO uses MUCH lower LR than SFT
        eval_strategy="no",
    )
)
trainer.train()
```

### DPO with Real Failures (Our Approach)
Our DPO uses real LLM extraction failures as rejected responses:
- **Chosen:** Apriori ground truth (correct itemsets)
- **Rejected:** Actual GPT model failures (wrong counts, hallucinated items, format errors)
- 606 preference pairs from runs where LLM ≠ Apriori

**Key DPO considerations:**
- Lower learning rate than SFT (5e-7 vs 1e-4)
- `beta=0.1` controls how much the model can deviate from reference
- Higher beta = stay closer to SFT model (more conservative)
- Lower beta = allow more change (risk instability)

---

## Group Relative Policy Optimization (GRPO)

**What it is:** Online RL training that generates multiple responses and uses relative rewards within each group.

**When to use:**
- Have a clear reward function (e.g., format correctness, accuracy)
- Want online exploration (model discovers solutions)
- After SFT+DPO for further refinement

**Dataset format:** Prompts only (model generates responses, reward function scores them)

```python
from trl import GRPOTrainer, GRPOConfig

def reward_fn(completions, **kwargs):
    """Score each completion."""
    scores = []
    for completion in completions:
        try:
            data = json.loads(completion)
            scores.append(1.0 if validate(data) else 0.0)
        except:
            scores.append(-1.0)
    return scores

trainer = GRPOTrainer(
    model="path/to/dpo-model",
    reward_funcs=[reward_fn],
    train_dataset=prompts_dataset,
    args=GRPOConfig(
        output_dir="grpo-model",
        learning_rate=5e-6,
        num_generations=4,
        max_grad_norm=0.1,
    )
)
trainer.train()
```

### GRPO Status in Our Project
⚠️ **GRPO is currently SKIPPED** (Council Decision, 2026-03-09):
- v1 GRPO produced F1=0 (complete collapse)
- Reward function design is fragile for structured JSON extraction
- SFT+DPO provides sufficient quality improvement
- May revisit if DPO plateaus

---

## Method Selection for Our Project

### Decision Tree
```
Start → SFT-CoT (always first)
  ↓
  SFT loss < 0.1?
  ├── Yes → DPO with real failures
  │         ↓
  │         F1 ≥ 80%?
  │         ├── Yes → Done ✅
  │         └── No → Analyze failure patterns, more data, or try GRPO
  └── No → Check data quality, increase epochs, adjust LR
```

### Our 3-Phase Pipeline
1. **Phase 1: SFT-CoT** — Train on ground truth with `<think>` reasoning
2. **Phase 2: DPO-Real** — Align using real LLM failures as negatives
3. ~~Phase 3: GRPO~~ — **Skipped** (Council decision)

---

## Reliability Principles

From HuggingFace production experience:

### Choose reliability over performance
```python
# ❌ RISKY: Aggressive optimization that may fail
SFTConfig(
    torch_compile=True,        # Can fail on T4, A10G GPUs
    optim="adamw_bnb_8bit",    # Requires specific setup
)

# ✅ SAFE: Proven defaults
SFTConfig(
    optim="paged_adamw_8bit",  # Standard, works with bitsandbytes
    bf16=True,                 # Stable and fast on modern GPUs
    gradient_checkpointing=True,
)
```

### Always verify label masking
```python
# After train_on_responses_only(), verify labels
labels = trainer.train_dataset[0]["labels"]
assert -100 in labels, "System/user tokens should be masked (-100)"
```

---

## See Also

- [[Unsloth Notebook Patterns]] — Diamond knowledge defaults
- [[Training Agent]] — Full training history and lessons
- [[Decisions/Diamond Knowledge Integration 2026-03-17]]
- [[Hardware and Memory Guide]] — VRAM requirements per method
