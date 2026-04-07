---
name: huggingface-deployment
description: Deploy adapters, datasets, and notebooks to HuggingFace Hub. Manage versioned repos, GGUF conversion, and model cards. No Spaces/Gradio (not used in this project).
---

# HuggingFace Hub Deployment

Push adapters, datasets, and training notebooks to HuggingFace Hub.

## Overview

Deployment targets for this project:
- **Adapters:** LoRA adapters (NEVER merged 4-bit models)
- **Datasets:** Versioned HF datasets (sft/dpo/grpo configs)
- **Notebooks:** Versioned training notebooks
- **GGUF:** Optional quantized exports for local inference

⚠️ **No Spaces/Gradio** — This project uses direct model inference, not web demos.

## Prerequisites

### HuggingFace Token
```bash
# Environment variable (preferred)
export HF_TOKEN=hf_your_write_token

# Or login via CLI
huggingface-cli login

# Or from hf.env file (gitignored)
source hf.env
```

### Required Permissions
- Write access to `OliverSlivka/` namespace
- For large models: Git LFS quota

## Adapter Deployment (Primary)

### Push Adapter Only
```python
# ✅ CORRECT: Save and push adapter (from training notebook)
model.save_pretrained("./final-adapter")
tokenizer.save_pretrained("./final-adapter")

model.push_to_hub("OliverSlivka/qwen2.5-7b-itemset-v3")
tokenizer.push_to_hub("OliverSlivka/qwen2.5-7b-itemset-v3")
```

### ❌ NEVER Do This
```python
# NEVER merge 4-bit models — corrupts weights (v1 catastrophic bug)
model.merge_and_unload()  # → BROKEN ON 4-BIT
merged_model.push_to_hub(...)  # → GARBAGE WEIGHTS
```

### Adapter Loading (for inference)
```python
from unsloth import FastLanguageModel

# Load base + adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="OliverSlivka/qwen2.5-7b-itemset-v3",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

## Dataset Deployment

### Upload Script
```bash
python src/training/upload_dataset_to_hf.py \
  --dataset-path data/hf_dataset_v3 \
  --repo OliverSlivka/itemset-extraction-v3
```

### Programmatic Upload
```python
from datasets import load_from_disk

dataset = load_from_disk("data/hf_dataset_v3")
dataset.push_to_hub("OliverSlivka/itemset-extraction-v3")
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
  - sft
  - dpo
size_categories:
  - 100<n<1K
configs:
  - sft    # SFT-CoT training examples
  - dpo    # DPO preference pairs
  - grpo   # GRPO prompts (format ready)
---
```

## Versioning Rules (CRITICAL)

### ⚠️ NEVER Overwrite Old Repos
Each training version gets its own repo:

| Version | Model Repo | Dataset Repo |
|---------|-----------|--------------|
| v2 | `OliverSlivka/qwen2.5-7b-itemset-v2` | `OliverSlivka/itemset-extraction-v2` |
| **v3** | **`OliverSlivka/qwen2.5-7b-itemset-v3`** | **`OliverSlivka/itemset-extraction-v3`** |
| v4 | `OliverSlivka/qwen2.5-7b-itemset-v4` | `OliverSlivka/itemset-extraction-v4` |

### Naming Convention
- Models: `OliverSlivka/qwen2.5-{size}-itemset-{version}`
- Datasets: `OliverSlivka/itemset-extraction-{version}`
- Each version is immutable after push

### Version Tracking
Keep track of all versions in Obsidian:
- `obsidian-brain/References/HF Dataset Repos.md`
- `obsidian-brain/Experiments/` (per-training reports)

## GGUF Conversion (Optional)

For local inference with llama.cpp or Ollama:

### Using Unsloth (Recommended)
```python
# Must reload base in float16 first!
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct",
    load_in_4bit=False,       # Full precision for merge
    dtype=torch.float16,
)

# Load adapter
model = PeftModel.from_pretrained(model, "./final-adapter")
model = model.merge_and_unload()  # Safe on float16!

# Save GGUF
model.save_pretrained_gguf(
    "model-gguf",
    tokenizer,
    quantization_method="q4_k_m",  # Good balance
)

# Push GGUF to Hub
model.push_to_hub_gguf(
    "OliverSlivka/qwen2.5-7b-itemset-v3-GGUF",
    tokenizer,
    quantization_method="q4_k_m",
)
```

### GGUF Quantization Options
| Method | Size (7B) | Quality | Use Case |
|--------|-----------|---------|----------|
| q4_k_m | ~4.5GB | Good | Default choice |
| q5_k_m | ~5.5GB | Better | Quality-focused |
| q8_0 | ~8GB | Best | Max quality |
| q2_k | ~3GB | Fair | Size-constrained |

## Hub Operations

### Check Repository
```python
from huggingface_hub import model_info, HfApi

info = model_info("OliverSlivka/qwen2.5-7b-itemset-v3")
print(f"Downloads: {info.downloads}")
print(f"Last modified: {info.lastModified}")
print(f"Files: {[s.rfilename for s in info.siblings]}")
```

### List All Versions
```python
api = HfApi()
models = api.list_models(author="OliverSlivka", search="itemset")
for m in models:
    print(f"{m.modelId} — {m.lastModified}")
```

### Upload Additional Files
```python
api = HfApi()

# Upload eval results
api.upload_file(
    path_or_fileobj="eval_results.json",
    path_in_repo="eval_results.json",
    repo_id="OliverSlivka/qwen2.5-7b-itemset-v3",
)

# Upload training notebook
api.upload_file(
    path_or_fileobj="notebooks/training_3phase_7b.ipynb",
    path_in_repo="training_notebook.ipynb",
    repo_id="OliverSlivka/qwen2.5-7b-itemset-v3",
)
```

### CLI Operations
```bash
# Download model
huggingface-cli download OliverSlivka/qwen2.5-7b-itemset-v3

# Check repo info
huggingface-cli repo info OliverSlivka/qwen2.5-7b-itemset-v3

# Upload folder
huggingface-cli upload OliverSlivka/qwen2.5-7b-itemset-v3 ./final-adapter
```

## Model Card Template

```markdown
---
language: en
license: apache-2.0
tags:
  - qwen2.5
  - frequent-itemset-mining
  - lora
  - unsloth
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-7B-Instruct
---

# Qwen2.5-7B Itemset Extractor (v3)

LoRA adapter fine-tuned on Qwen2.5-7B-Instruct for extracting frequent itemsets from CSV data.

## Training
- **Method:** SFT-CoT → DPO-Real (3-phase, GRPO skipped)
- **Framework:** Unsloth + LoRA (r=32, alpha=64)
- **Data:** 348 SFT examples + 606 DPO preference pairs
- **Hardware:** NVIDIA H200 NVL

## Usage
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "OliverSlivka/qwen2.5-7b-itemset-v3",
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

## Metrics
| Metric | Score |
|--------|-------|
| F1 | TBD |
| Parse Rate | TBD |
```

## Deployment Checklist

Before pushing any version:

- [ ] Training completed successfully (no NaN loss)
- [ ] Adapter saved (NOT merged 4-bit)
- [ ] Eval results recorded in Obsidian
- [ ] Version number incremented (check existing repos)
- [ ] Model card written with training details
- [ ] Dataset repo matches model version
- [ ] Training notebook included
- [ ] Pushed with descriptive commit message

## Troubleshooting

### Push Failed (403)
- Check token has write permission: `huggingface-cli whoami`
- Verify HF_TOKEN is set: `echo $HF_TOKEN`
- Check repo ownership

### Large File Errors
```bash
# Enable Git LFS
git lfs install
git lfs track "*.safetensors" "*.bin" "*.gguf"
```

### Adapter vs Full Model Confusion
- **Adapter repo:** Contains `adapter_config.json` + `adapter_model.safetensors` (~100MB)
- **Full model repo:** Contains `model.safetensors` (~15GB for 7B)
- This project ONLY pushes adapters (except GGUF conversions)

### Wrong Version Pushed
- HuggingFace repos are immutable by convention in this project
- If wrong content pushed: Create new version repo instead of overwriting
- Contact HF support to delete repo only if absolutely necessary
| F1 | 0.XX |
| Parse Rate | XX% |

## Limitations
- Optimized for 5-25 row datasets
- May struggle with >20 columns
```
