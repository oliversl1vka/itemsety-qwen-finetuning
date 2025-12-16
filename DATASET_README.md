---
license: mit
task_categories:
- text-generation
language:
- en
size_categories:
- n<1K
---

# Itemsety Real Training Dataset

Frequent itemset extraction training dataset with 88 train and 10 validation examples.

## Dataset Description

This dataset contains chat-formatted examples for supervised fine-tuning (SFT) of language models on frequent itemset mining tasks.

## Dataset Structure

### Data Fields

- `messages`: List of conversation messages with `role` and `content` fields
  - `role`: Either "system", "user", or "assistant"
  - `content`: Text content of the message
- `metadata`: Dictionary containing:
  - `dataset_id`: Unique identifier for the source dataset
  - `dataset_name`: Original CSV filename
  - `itemset_count`: Number of frequent itemsets
  - `min_support`: Minimum support threshold

### Data Splits

- `train`: 88 examples
- `validation`: 10 examples

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("OliverSlivka/itemsety-real-training")

# Access train split
train_data = dataset["train"]
print(train_data[0]["messages"])
```

## Dataset Creation

Created from runs.db using export_training_data.py and enhanced with real-world context via enhance_training_data.py.

## Citation

Internal research dataset for frequent itemset mining with LLMs.
