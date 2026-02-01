---
name: huggingface-deployment
description: Deploy models and datasets to HuggingFace Hub. Manage Gradio Spaces for interactive demos.
---

# HuggingFace Hub Deployment

Deploy models, datasets, and Spaces to HuggingFace Hub.

## Overview

Deployment targets:
- **Models:** Fine-tuned Qwen models
- **Datasets:** Training and evaluation datasets
- **Spaces:** Interactive Gradio demos

## Prerequisites

### HuggingFace Token
```bash
# Set environment variable
export HF_TOKEN=hf_your_write_token

# Or login via CLI
huggingface-cli login
```

### Required Permissions
- Write access to target repositories
- For Spaces: GPU quota (if using Zero GPU)

## Model Deployment

### Push Merged Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./outputs/final")
tokenizer = AutoTokenizer.from_pretrained("./outputs/final")

model.push_to_hub("OliverSlivka/qwen2.5-3b-itemset-extractor")
tokenizer.push_to_hub("OliverSlivka/qwen2.5-3b-itemset-extractor")
```

### Push with Model Card
```python
# Create README.md in model directory first
model.push_to_hub(
    "OliverSlivka/qwen2.5-3b-itemset-extractor",
    commit_message="Add fine-tuned model v2.0"
)
```

### Push Adapter Only
```python
# For LoRA adapter without merging
model.save_pretrained("./adapter")
model.push_to_hub("OliverSlivka/qwen2.5-3b-itemset-adapter")
```

## Dataset Deployment

### Push Dataset
```bash
python src/training/upload_dataset_to_hf.py --dataset-path data/hf_dataset_v2
```

### Programmatic Upload
```python
from datasets import load_from_disk

dataset = load_from_disk("data/hf_dataset_v2")
dataset.push_to_hub("OliverSlivka/itemset-extraction-dataset")
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
size_categories:
  - 100<n<1K
---

# Itemset Extraction Dataset

Training data for frequent itemset extraction from CSV files.
```

## Gradio Space Deployment

### Deploy Script
```powershell
./deploy_to_hf_space.ps1
```

### Manual Deployment
```bash
# Clone Space repo
git clone https://huggingface.co/spaces/OliverSlivka/itemset-extractor
cd itemset-extractor

# Copy files
cp ../app_v2.py app.py
cp ../requirements.txt .

# Push
git add .
git commit -m "Update app"
git push
```

### Space Configuration (README.md)
```yaml
---
title: Itemset Extractor
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-gpu  # or cpu-basic, t4-small, a10g-small
---
```

### GPU Options

| Hardware | VRAM | Cost | Use Case |
|----------|------|------|----------|
| cpu-basic | - | Free | Testing |
| zero-gpu | 16GB (shared) | Free | Demo |
| t4-small | 16GB | $0.60/hr | Production |
| a10g-small | 24GB | $1.50/hr | Large models |

## Health Checks

### Check Model
```python
from huggingface_hub import model_info

info = model_info("OliverSlivka/qwen2.5-3b-itemset-extractor")
print(f"Downloads: {info.downloads}")
print(f"Last modified: {info.lastModified}")
```

### Check Space
```python
from huggingface_hub import space_info

info = space_info("OliverSlivka/itemset-extractor")
print(f"Status: {info.runtime.stage}")
print(f"Hardware: {info.runtime.hardware}")
```

### Test Inference
```python
from huggingface_hub import InferenceClient

client = InferenceClient("OliverSlivka/qwen2.5-3b-itemset-extractor")
response = client.text_generation("Extract itemsets from...")
```

## Version Management

### Tag Release
```bash
# Using git
cd model_repo
git tag v2.0.0
git push --tags

# Using API
from huggingface_hub import HfApi
api = HfApi()
api.create_tag("OliverSlivka/qwen2.5-3b-itemset-extractor", tag="v2.0.0")
```

### Rollback
```python
# Download specific revision
model = AutoModelForCausalLM.from_pretrained(
    "OliverSlivka/qwen2.5-3b-itemset-extractor",
    revision="v1.0.0"
)
```

## Troubleshooting

### Push Failed (403)
- Check token has write permission
- Verify repository exists
- Check you own the repo

### Space Build Failed
- Check requirements.txt syntax
- Verify Python version compatibility
- Check app.py has valid Gradio interface

### Zero GPU Timeout
- Optimize model loading
- Use smaller model
- Reduce max_new_tokens

### Large Files
```bash
# Enable Git LFS
git lfs install
git lfs track "*.bin" "*.safetensors"
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
pipeline_tag: text-generation
---

# Qwen2.5-3B Itemset Extractor

## Model Description
Fine-tuned Qwen2.5-3B-Instruct for extracting frequent itemsets from CSV data.

## Training Data
- 439 validated examples from Apriori + GPT-4 pipeline
- ChatML format with Chain-of-Thought reasoning

## Usage
\`\`\`python
from transformers import pipeline

pipe = pipeline("text-generation", model="OliverSlivka/qwen2.5-3b-itemset-extractor")
result = pipe("Extract frequent itemsets from: ...")
\`\`\`

## Metrics
| Metric | Score |
|--------|-------|
| F1 | 0.XX |
| Parse Rate | XX% |

## Limitations
- Optimized for 5-25 row datasets
- May struggle with >20 columns
```
