#!/usr/bin/env python3
"""
Push versioned datasets to HuggingFace Hub.
Restores v2 to original state and creates new v3 repo.

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/push_versioned_datasets.py
"""
import os
import sys
from pathlib import Path
from datasets import DatasetDict, load_from_disk
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

login(token=HF_TOKEN)
api = HfApi()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Restore v2 to original state (348 SFT examples, verbose Row N format)
# ═══════════════════════════════════════════════════════════════════════════════
V2_REPO = "OliverSlivka/itemset-extraction-v2"
V2_LOCAL = "data/hf_dataset_v2"

print(f"\n{'='*60}")
print(f"📦 RESTORING v2: {V2_REPO}")
print(f"{'='*60}")

for config in ["sft", "dpo", "grpo"]:
    ds = load_from_disk(f"{V2_LOCAL}/{config}")
    print(f"  [{config}] train={len(ds['train'])}, val={len(ds['validation'])}")
    ds.push_to_hub(V2_REPO, config_name=config, private=False,
                   commit_message=f"Restore v2 {config}: original 348-example verbose Row N format")
    print(f"  ✅ Pushed {config}")

# Push v2 notebook
V2_NOTEBOOK = "notebooks/training_3phase_2026-03-07_v2.ipynb"
if Path(V2_NOTEBOOK).exists():
    api.upload_file(
        path_or_fileobj=V2_NOTEBOOK,
        path_in_repo="notebooks/training_3phase_7b.ipynb",
        repo_id=V2_REPO,
        repo_type="dataset",
        commit_message="Restore v2 notebook (training_3phase_2026-03-07_v2.ipynb)",
    )
    print(f"  ✅ Pushed v2 notebook")
else:
    print(f"  ⚠️ v2 notebook not found at {V2_NOTEBOOK}, skipping")

print(f"\n🔗 v2 live at: https://huggingface.co/datasets/{V2_REPO}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Create and push v3 (272 SFT examples, concise spaced R-ref format)
# ═══════════════════════════════════════════════════════════════════════════════
V3_REPO = "OliverSlivka/itemset-extraction-v3"
V3_LOCAL = "data/hf_dataset_v3"

print(f"\n{'='*60}")
print(f"📦 CREATING v3: {V3_REPO}")
print(f"{'='*60}")

# Create repo if it doesn't exist
try:
    api.create_repo(repo_id=V3_REPO, repo_type="dataset", exist_ok=True, private=False)
    print(f"  ✅ Repo ready: {V3_REPO}")
except Exception as e:
    print(f"  ℹ️ Repo creation: {e}")

for config in ["sft", "dpo", "grpo"]:
    ds = load_from_disk(f"{V3_LOCAL}/{config}")
    print(f"  [{config}] train={len(ds['train'])}, val={len(ds['validation'])}")
    ds.push_to_hub(V3_REPO, config_name=config, private=False,
                   commit_message=f"v3 {config}: 272 concise SFT examples with spaced R-refs, tokenizer-verified ≤4096 tokens")
    print(f"  ✅ Pushed {config}")

# Push v3 notebook
V3_NOTEBOOK = "notebooks/training_3phase_2026-03-09_v3.ipynb"
if Path(V3_NOTEBOOK).exists():
    api.upload_file(
        path_or_fileobj=V3_NOTEBOOK,
        path_in_repo="notebooks/training_3phase_7b.ipynb",
        repo_id=V3_REPO,
        repo_type="dataset",
        commit_message="v3.10 notebook: spaced R-refs, inlined inference utils, tokenizer-verified data",
    )
    print(f"  ✅ Pushed v3 notebook")
else:
    print(f"  ⚠️ v3 notebook not found at {V3_NOTEBOOK}, skipping")

# Push dataset card for v3
V3_README = """---
dataset_info:
  - config_name: sft
    features:
      - name: messages
        list:
          - name: content
            dtype: string
          - name: role
            dtype: string
    splits:
      - name: train
        num_examples: 245
      - name: validation
        num_examples: 27
  - config_name: dpo
    features:
      - name: prompt
        list:
          - name: content
            dtype: string
          - name: role
            dtype: string
      - name: chosen
        list:
          - name: content
            dtype: string
          - name: role
            dtype: string
      - name: rejected
        list:
          - name: content
            dtype: string
          - name: role
            dtype: string
    splits:
      - name: train
        num_examples: 546
      - name: validation
        num_examples: 60
  - config_name: grpo
    features:
      - name: prompt
        list:
          - name: content
            dtype: string
          - name: role
            dtype: string
      - name: ground_truth
        dtype: string
    splits:
      - name: train
        num_examples: 245
      - name: validation
        num_examples: 27
configs:
  - config_name: sft
    data_files:
      - split: train
        path: sft/train-*
      - split: validation
        path: sft/validation-*
  - config_name: dpo
    data_files:
      - split: train
        path: dpo/train-*
      - split: validation
        path: dpo/validation-*
  - config_name: grpo
    data_files:
      - split: train
        path: grpo/train-*
      - split: validation
        path: grpo/validation-*
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - frequent-itemset-mining
  - fine-tuning
  - chain-of-thought
  - v3
---

# Itemset Extraction Training Dataset — v3

**Version:** v3.10 (2026-03-18)
**Model target:** Qwen2.5-7B-Instruct

## What's New in v3 (vs v2)

| Aspect | v2 | v3 |
|--------|----|----|
| SFT format | Verbose `Row N` in think block | **Concise column-grouped**, spaced `R1, R10, R2` |
| SFT examples | 348 (314/34 split) | **272** (245/27 split, tokenizer-verified ≤4096) |
| R-ref format | N/A (Row N) | **Spaced** `R1, R10` (clean tokenization) |
| Token filter | chars/4 estimate | **Actual Qwen tokenizer** (0 examples >4096) |
| DPO pairs | 606 (546/60) | 606 (546/60) — unchanged |

## Configs

### `sft` — Supervised Fine-Tuning with Chain-of-Thought
- 245 train / 27 val examples
- Format: `{messages: [{role, content}]}` with `<think>` reasoning

### `dpo` — Direct Preference Optimization
- 546 train / 60 val pairs
- Chosen = Apriori ground truth, Rejected = real LLM failures from 4 models

### `grpo` — Group Relative Policy Optimization
- 245 train / 27 val (reuses SFT prompts with ground_truth JSON)

## Usage

```python
from datasets import load_dataset

sft = load_dataset("OliverSlivka/itemset-extraction-v3", "sft")
dpo = load_dataset("OliverSlivka/itemset-extraction-v3", "dpo")
grpo = load_dataset("OliverSlivka/itemset-extraction-v3", "grpo")
```

## Training Notebook

Download from this repo: `notebooks/training_3phase_7b.ipynb`

## Version History

- **v3** (2026-03-18): Fixed R-shorthand tokenization, concise CoT format, tokenizer-verified lengths
- **v2** (2026-03-07): Original verbose format, 348 SFT examples → [itemset-extraction-v2](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2)
"""

import tempfile
with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
    f.write(V3_README)
    readme_path = f.name

api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=V3_REPO,
    repo_type="dataset",
    commit_message="Add v3 dataset card with version history",
)
os.unlink(readme_path)
print(f"  ✅ Pushed v3 README/dataset card")

print(f"\n🔗 v3 live at: https://huggingface.co/datasets/{V3_REPO}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("✅ VERSIONING COMPLETE")
print(f"{'='*60}")
print(f"  v2 (frozen): https://huggingface.co/datasets/{V2_REPO}")
print(f"    SFT: 314/34 (verbose Row N, from sft_cot_v2.json)")
print(f"    DPO: 546/60")
print(f"    GRPO: 314/34")
print(f"  v3 (current): https://huggingface.co/datasets/{V3_REPO}")
print(f"    SFT: 245/27 (concise spaced R-refs, from sft_cot_v3.json)")
print(f"    DPO: 546/60")
print(f"    GRPO: 245/27")
