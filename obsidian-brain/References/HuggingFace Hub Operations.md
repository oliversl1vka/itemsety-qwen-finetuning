# HuggingFace Hub Operations

**Date:** 2026-03-18  
**Source:** HuggingFace Skills `hub_saving.md`, `hf-cli` + project experience  
**Tags:** #reference #huggingface #hub #deployment #versioning

---

## Authentication

### Token Setup
```bash
# Environment variable (preferred)
export HF_TOKEN=hf_your_write_token

# Or login via CLI
huggingface-cli login

# Or in Python
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN"))
```

### Token Permissions
| Operation | Permission Needed |
|-----------|------------------|
| Download public models | None |
| Download private models | Read |
| Push models/datasets | Write |
| Create repos | Write |
| Manage organization repos | Write + Org membership |

---

## Model Upload Patterns

### ✅ Adapter-Only Upload (Recommended for LoRA)
```python
# Save locally first
model.save_pretrained("./adapter-output")
tokenizer.save_pretrained("./adapter-output")

# Push adapter + tokenizer
model.push_to_hub("OliverSlivka/qwen2.5-7b-itemset-adapter-v3")
tokenizer.push_to_hub("OliverSlivka/qwen2.5-7b-itemset-adapter-v3")
```

### ❌ Never Merge 4-bit Then Push
```python
# THIS CORRUPTS WEIGHTS — our v1 catastrophic bug
model = model.merge_and_unload()  # On 4-bit → garbage
```

### Merged Model Upload (when needed)
```python
# Reload base model in full precision first
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu",  # Merge on CPU to save VRAM
)
merged = PeftModel.from_pretrained(base, "./adapter-output").merge_and_unload()
merged.push_to_hub("OliverSlivka/qwen2.5-7b-itemset-merged-v3")
```

---

## Dataset Upload Patterns

### From Local Directory
```python
from datasets import load_from_disk

dataset = load_from_disk("data/hf_dataset_v3")
dataset.push_to_hub("OliverSlivka/itemset-extraction-v3")
```

### Using Upload Script
```bash
python src/training/upload_dataset_to_hf.py \
  --dataset-path data/hf_dataset_v3 \
  --repo OliverSlivka/itemset-extraction-v3
```

### Dataset Card
```yaml
---
language: en
task_categories:
  - text-generation
tags:
  - frequent-itemset-mining
  - csv-analysis
  - chain-of-thought
size_categories:
  - 100<n<1K
configs:
  - sft
  - dpo
  - grpo
---
```

---

## Hub Saving During Training

### Strategy Options

| Strategy | Behavior | When to Use |
|----------|----------|-------------|
| `"end"` | Push only at end | Short training, save bandwidth |
| `"every_save"` | Push at each checkpoint | Long training, need recovery |
| `"checkpoint"` | Push checkpoints | Want to resume from any point |
| `"all_checkpoints"` | Push all, don't delete | Need full training history |

### Production Training Config
```python
config = SFTConfig(
    output_dir="model-output",
    
    # Hub settings
    push_to_hub=True,
    hub_model_id="username/model-name",
    hub_strategy="every_save",
    
    # Checkpoint settings
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,  # Keep only last 3 checkpoints
)
```

---

## Versioning Rules

### ⚠️ CRITICAL: Never Overwrite Repos

Each dataset/model version gets its **own repo**:

| Version | Dataset Repo | Model Repo |
|---------|-------------|------------|
| v2 | `OliverSlivka/itemset-extraction-v2` | `OliverSlivka/qwen2.5-7b-itemset-v2` |
| v3 | `OliverSlivka/itemset-extraction-v3` | `OliverSlivka/qwen2.5-7b-itemset-v3` |

**Why:** Old repos may be referenced by training logs, experiment reports, and evaluation comparisons. Overwriting breaks reproducibility.

**Decision:** See [[Decisions/HF Dataset Versioning]] for full rationale.

### Current Repos
See [[HF Dataset Repos]] for the complete registry.

---

## GGUF Conversion

For local deployment (llama.cpp, Ollama, LM Studio):

### Process
1. Load base model + LoRA adapter
2. Merge in full precision (CPU)
3. Convert to GGUF format
4. Upload quantized files

### Quantization Options

| Quant | Size (7B) | Quality | Use Case |
|-------|-----------|---------|----------|
| F16 | ~14GB | Best | Full precision |
| Q8_0 | ~7GB | High | Good balance |
| Q5_K_M | ~5GB | Good | Recommended default |
| Q4_K_M | ~4GB | Acceptable | Memory-constrained |
| Q3_K_S | ~3GB | Lower | Edge devices |

### HuggingFace conversion script pattern
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load and merge
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16)
model = PeftModel.from_pretrained(base, ADAPTER_MODEL)
merged = model.merge_and_unload()

# 2. Save merged model
merged.save_pretrained("/tmp/merged")

# 3. Convert with llama.cpp (external tool)
# llama-quantize /tmp/merged/model.gguf output.gguf Q5_K_M
```

---

## CLI Quick Reference

### Model Operations
```bash
# Download model
hf download username/model-name

# Upload file
hf upload username/model-name ./local-file.safetensors

# Create repo
hf repos create username/model-name

# Tag release
hf repos tag username/model-name v1.0.0
```

### Dataset Operations
```bash
# Download dataset
hf download username/dataset-name --repo-type dataset

# Upload dataset files
hf upload username/dataset-name ./data/ --repo-type dataset
```

### Info Queries
```python
from huggingface_hub import model_info, dataset_info

# Check model details
info = model_info("OliverSlivka/qwen2.5-7b-itemset-v3")
print(f"Downloads: {info.downloads}")
print(f"Tags: {info.tags}")

# Check dataset details
info = dataset_info("OliverSlivka/itemset-extraction-v3")
print(f"Size: {info.card_data.get('size_categories', 'unknown')}")
```

---

## Model Card Best Practices

### Required Sections
1. **Model Description** — What it does, base model, training method
2. **Training Data** — Dataset link, size, format
3. **Usage** — Code example for inference
4. **Metrics** — Evaluation results table
5. **Limitations** — Known weaknesses

### Model-Index Format (for HF leaderboards)
```yaml
model-index:
  - name: qwen2.5-7b-itemset-v3
    results:
      - task:
          type: text-generation
        dataset:
          name: Itemset Extraction Eval
          type: custom
        metrics:
          - name: F1 Score
            type: f1
            value: 0.786
          - name: JSON Parse Rate
            type: parse_rate
            value: 0.95
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| 403 Forbidden | Token lacks write permission | Regenerate token with write access |
| 409 Conflict | Repo already exists | Use different name or add version suffix |
| Large file error | File > 10GB without LFS | `git lfs install && git lfs track "*.safetensors"` |
| Slow upload | Large model files | `export HF_HUB_ENABLE_HF_TRANSFER=1` for 5× speedup |
| Push fails mid-upload | Network timeout | Use `hub_strategy="checkpoint"` for resumable uploads |
| Missing tokenizer | Forgot to push tokenizer | Always push both model and tokenizer |

---

## See Also

- [[HF Dataset Repos]] — Registry of all versioned repos
- [[Decisions/HF Dataset Versioning]] — Why we never overwrite repos
- [[Deployment Agent]] — Agent-specific deployment procedures
- [[Training Methods Guide]] — Training configuration that feeds into deployment
