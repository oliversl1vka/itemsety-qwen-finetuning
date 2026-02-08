#!/usr/bin/env python3
"""
Upload training dataset to HuggingFace Hub for fine-tuning.
"""

import argparse
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

# Configuration
REPO_NAME = "OliverSlivka/itemset-extraction-v2"
DEFAULT_TRAINING_DATA_DIR = Path("data/training_v2")

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def main():
    parser = argparse.ArgumentParser(description="Upload training dataset to HuggingFace Hub")
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DATA_DIR,
                       help="Path to training data directory (default: data/training_v2)")
    parser.add_argument("--dataset-path", type=Path, help="Alias for --training-dir (for backward compat)")
    args = parser.parse_args()
    
    # Use dataset-path if provided, otherwise training-dir
    training_dir = args.dataset_path if args.dataset_path else args.training_dir
    
    print("📂 Loading training data...")
    print(f"   Directory: {training_dir}")
    
    # Load simple format (clean JSON output - better for fine-tuning)
    simple_examples = load_jsonl(training_dir / "train_simple.jsonl")
    print(f"   Loaded {len(simple_examples)} simple examples")
    
    # Load CoT format (with reasoning - for comparison)
    cot_examples = load_jsonl(training_dir / "train_cot.jsonl")
    print(f"   Loaded {len(cot_examples)} CoT examples")
    
    # Split into train/validation (90/10)
    split_idx = int(len(simple_examples) * 0.9)
    
    train_simple = simple_examples[:split_idx]
    val_simple = simple_examples[split_idx:]
    
    train_cot = cot_examples[:split_idx]
    val_cot = cot_examples[split_idx:]
    
    print(f"\n📊 Split statistics:")
    print(f"   Train: {len(train_simple)} examples")
    print(f"   Validation: {len(val_simple)} examples")
    
    # Convert to HuggingFace format - extract messages for SFT training
    def convert_to_hf_format(examples: list[dict]) -> dict:
        """Convert to format expected by HF training."""
        return {
            "messages": [ex["messages"] for ex in examples],
            "dataset_id": [ex["metadata"]["dataset_id"] for ex in examples],
            "num_itemsets": [ex["metadata"]["num_itemsets"] for ex in examples],
        }
    
    # Create datasets
    print("\n🔨 Creating HuggingFace datasets...")
    
    # Simple format dataset (recommended for fine-tuning)
    ds_simple = DatasetDict({
        "train": Dataset.from_dict(convert_to_hf_format(train_simple)),
        "validation": Dataset.from_dict(convert_to_hf_format(val_simple)),
    })
    
    # CoT format dataset (alternative with reasoning)
    ds_cot = DatasetDict({
        "train": Dataset.from_dict(convert_to_hf_format(train_cot)),
        "validation": Dataset.from_dict(convert_to_hf_format(val_cot)),
    })
    
    print(f"   Simple dataset: {ds_simple}")
    print(f"   CoT dataset: {ds_cot}")
    
    # Upload to Hub
    print(f"\n📤 Uploading to HuggingFace Hub: {REPO_NAME}")
    
    # Upload simple format (main dataset)
    ds_simple.push_to_hub(
        REPO_NAME,
        private=False,
        commit_message="Upload itemset extraction training data v2 (simple format)"
    )
    print(f"   ✅ Uploaded simple format to {REPO_NAME}")
    
    # Upload CoT format as separate config
    ds_cot.push_to_hub(
        f"{REPO_NAME}",
        config_name="chain_of_thought",
        private=False,
        commit_message="Add chain-of-thought format"
    )
    print(f"   ✅ Uploaded CoT format to {REPO_NAME} (config: chain_of_thought)")
    
    # Create dataset card
    print("\n📝 Creating dataset card...")
    
    dataset_card = f"""---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - frequent-itemset-mining
  - data-extraction
  - json-generation
  - fine-tuning
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: validation  
        path: data/validation-*
  - config_name: chain_of_thought
    data_files:
      - split: train
        path: chain_of_thought/train-*
      - split: validation
        path: chain_of_thought/validation-*
---

# Itemset Extraction Training Data v2

Training dataset for fine-tuning LLMs to extract frequent itemsets from transaction data.

## Dataset Description

This dataset contains {len(simple_examples)} examples of:
- **Input**: Transaction data in CSV format + minimum support threshold
- **Output**: JSON array of frequent itemsets with support counts and evidence rows

### Configs

1. **default** - Clean JSON output format (recommended for fine-tuning)
2. **chain_of_thought** - Includes reasoning steps before JSON output

### Statistics

| Metric | Value |
|--------|-------|
| Total examples | {len(simple_examples)} |
| Train split | {len(train_simple)} |
| Validation split | {len(val_simple)} |
| Avg itemsets per example | 17.2 |
| Source datasets | 500 semi-human CSVs |

### Data Format

Each example has a `messages` field with ChatML format:

```json
{{
  "messages": [
    {{"role": "system", "content": "You are an expert data mining assistant..."}},
    {{"role": "user", "content": "Analyze the following transaction data..."}},
    {{"role": "assistant", "content": "[{{\\"itemset\\": [...], \\"count\\": N, ...}}]"}}
  ]
}}
```

### Key Features

- **Real column names** from 25 diverse domains (retail, healthcare, education, etc.)
- **Anti-hallucination rules** in system prompt
- **Validated outputs** - all examples passed strict validation against Apriori ground truth
- **Support evidence** - each itemset includes row IDs where it appears

### Usage

```python
from datasets import load_dataset

# Load default (simple) format
ds = load_dataset("OliverSlivka/itemset-extraction-v2")

# Load chain-of-thought format
ds_cot = load_dataset("OliverSlivka/itemset-extraction-v2", "chain_of_thought")
```

### Fine-tuning

Recommended for SFT with:
- Qwen2.5-3B or Qwen2.5-7B
- LoRA/QLoRA for efficiency
- 2-3 epochs

## License

MIT
"""
    
    # Upload README
    api = HfApi()
    api.upload_file(
        path_or_fileobj=dataset_card.encode(),
        path_in_repo="README.md",
        repo_id=REPO_NAME,
        repo_type="dataset",
        commit_message="Add dataset card"
    )
    print("   ✅ Dataset card uploaded")
    
    print(f"\n🎉 Done! Dataset available at: https://huggingface.co/datasets/{REPO_NAME}")

if __name__ == "__main__":
    main()
