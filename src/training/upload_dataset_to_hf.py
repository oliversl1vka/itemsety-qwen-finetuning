#!/usr/bin/env python3
"""
Upload pre-built HuggingFace dataset (3 configs: sft/dpo/grpo) to HuggingFace Hub.

Usage:
    python src/training/upload_dataset_to_hf.py
    python src/training/upload_dataset_to_hf.py --dataset-path data/hf_dataset_v2 --repo OliverSlivka/itemset-extraction-v2
"""

import argparse
import os
from pathlib import Path
from datasets import load_from_disk
from huggingface_hub import HfApi, login, create_repo

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_DATASET_PATH = Path("data/hf_dataset_v2")
DEFAULT_REPO = "OliverSlivka/itemset-extraction-v2"


def main():
    parser = argparse.ArgumentParser(
        description="Upload HF dataset (sft/dpo/grpo) to HuggingFace Hub"
    )
    parser.add_argument(
        "--dataset-path", type=Path, default=DEFAULT_DATASET_PATH,
        help=f"Path to local HF dataset directory (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--repo", type=str, default=DEFAULT_REPO,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--private", action="store_true", default=False,
        help="Make the dataset private",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    repo = args.repo

    # ── Resolve HF token ────────────────────────────────────────────────────
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN not set. Run `source hf.env` or set the env variable."
        )
    login(token=token, add_to_git_credential=False)
    print(f"✅ Logged in to HuggingFace (token: {token[:10]}...)")

    # ── Ensure repo exists ───────────────────────────────────────────────────
    create_repo(repo, repo_type="dataset", exist_ok=True, token=token)
    print(f"✅ Repo ready: https://huggingface.co/datasets/{repo}")
    print()

    # ── Validate local dataset ───────────────────────────────────────────────
    configs = ["sft", "dpo", "grpo"]
    for cfg in configs:
        cfg_path = dataset_path / cfg
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Missing config directory: {cfg_path}\n"
                f"Run build_hf_dataset_v2.py first to create the dataset."
            )

    print(f"📂 Dataset path: {dataset_path}")
    print(f"📤 Target repo:  {repo}")
    print()

    # ── Push each config ─────────────────────────────────────────────────────
    for cfg in configs:
        cfg_path = dataset_path / cfg
        ds = load_from_disk(str(cfg_path))

        print(f"  [{cfg}] Train: {len(ds['train']):,} | Val: {len(ds['validation']):,}")

        ds.push_to_hub(
            repo,
            config_name=cfg,
            private=args.private,
            token=token,
            commit_message=f"Upload {cfg} config ({len(ds['train'])} train, {len(ds['validation'])} val)",
        )
        print(f"  ✅ Pushed '{cfg}' config")

    # ── Upload dataset card ──────────────────────────────────────────────────
    print("\n📝 Uploading dataset card...")

    sft_ds = load_from_disk(str(dataset_path / "sft"))
    dpo_ds = load_from_disk(str(dataset_path / "dpo"))
    grpo_ds = load_from_disk(str(dataset_path / "grpo"))

    dataset_card = f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - frequent-itemset-mining
  - data-extraction
  - json-generation
  - fine-tuning
  - chain-of-thought
  - dpo
  - grpo
  - rlhf
size_categories:
  - n<1K
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
---

# Itemset Extraction Training Data v2

3-phase training dataset for fine-tuning LLMs to extract frequent itemsets from CSV transaction data.

## Overview

| Config | Purpose | Train | Val | Format |
|--------|---------|-------|-----|--------|
| **sft** | SFT with Chain-of-Thought | {len(sft_ds['train'])} | {len(sft_ds['validation'])} | `messages` (ChatML) |
| **dpo** | DPO with real LLM failures | {len(dpo_ds['train'])} | {len(dpo_ds['validation'])} | `prompt` / `chosen` / `rejected` |
| **grpo** | GRPO with Apriori rewards | {len(grpo_ds['train'])} | {len(grpo_ds['validation'])} | `prompt` / `ground_truth` |

## Training Pipeline (v2 — council-corrected)

```
Phase 1: SFT-CoT (5 epochs)  → Teach <think> reasoning + JSON with col:val format
Phase 2: GRPO (500 steps)    → Optimize with 5 programmatic reward functions
```

> **Note:** DPO data is included for reference but the v2 notebook does NOT use it.
> DPO was removed after LLM Council analysis found it destroyed `<think>` reasoning.

## Data Sources

- **SFT**: Validated pipeline runs with Chain-of-Thought reasoning generated from Apriori ground truth
- **DPO chosen**: Apriori ground truth + CoT reasoning
- **DPO rejected**: Real extraction failures from GPT-4.1-mini (25%), GPT-4.1-nano (44.5%), o4-mini (26.6%), GPT-4o (3.8%)
- **GRPO**: Same prompts as SFT, with serialized Apriori ground truth for reward computation

## Usage

```python
from datasets import load_dataset

# Load each config
sft  = load_dataset("{repo}", "sft")
dpo  = load_dataset("{repo}", "dpo")
grpo = load_dataset("{repo}", "grpo")
```

## Key Features

- **`<think>` reasoning**: SFT examples include step-by-step reasoning inside `<think>` tags
- **Real failures**: DPO rejected samples are actual LLM extraction errors (not synthetic)
- **Compact system prompt**: ~150 tokens, optimized for small models
- **Validated**: All ground truth verified against Apriori algorithm (13 invariants)
- **500 diverse datasets**: 5-25 rows, 4-20 columns, 25 domains

## Recommended Model

- **Qwen2.5-7B-Instruct** with LoRA (r=32, alpha=64 — ratio 2.0)
- 4-bit quantization via Unsloth
- 6144 max sequence length
- SFT: 5 epochs, lr=1e-4, packing=False
- GRPO: 500 steps, 5 reward functions, lr=2e-6
- Target: F1 >= 0.80 vs Apriori

## Training Notebook

See `notebooks/training_3phase_7b.ipynb` (v2) in this repo for the full training script.

## License

Apache 2.0
"""

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=dataset_card.encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        commit_message="Update dataset card for v2 (3-phase: sft/dpo/grpo)",
    )
    print("  ✅ Dataset card uploaded")

    print(f"\n🎉 Done! Dataset at: https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
