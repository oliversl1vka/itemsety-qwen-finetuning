# Data (`data/`)

All datasets and training data.

## Structure

```
data/
├── datasets_v2/       # Generated CSV datasets (500 files)
├── training_v2/       # Training examples (JSONL format)
├── hf_dataset_v2/     # HuggingFace Dataset format (current)
├── hf_dataset_v1/     # Legacy HF dataset (archived)
└── training_v1/       # Legacy training data (archived)
```

## Datasets

### datasets_v2/
500 synthetic CSV datasets with:
- 5-25 rows each
- 5-20 columns
- Real-world item names (grocery, electronics, etc.)
- Pattern: `ds_NNNN_ROWSxCOLS.csv`

### training_v2/
Training data in JSONL format:
- `train_combined.jsonl` - Full training set
- `train_cot.jsonl` - Chain-of-thought examples
- `train_simple.jsonl` - Simple examples
- `training_metadata.json` - Generation metadata

### hf_dataset_v2/
HuggingFace Dataset format ready for training:
- 439 training examples
- 49 validation examples
- ChatML conversation format

## Usage

```python
from datasets import load_from_disk

# Load local dataset
dataset = load_from_disk("data/hf_dataset_v2")

# Or from Hub
from datasets import load_dataset
dataset = load_dataset("OliverSlivka/itemset-extraction-v2")
```
