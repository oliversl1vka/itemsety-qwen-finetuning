---
name: training-data-export
description: Export validated pipeline runs as training examples in ChatML format with Chain-of-Thought reasoning. Use before fine-tuning to prepare training dataset.
---

# Training Data Export

Convert validated pipeline runs into fine-tuning training examples.

## Overview

Export creates training data with:
- ChatML conversation format (system/user/assistant)
- Chain-of-Thought (CoT) reasoning steps
- CSV input → JSON output pairs
- Quality filtering (validation_passed only)

## Quick Start

### Export Validated Runs
```bash
python src/training/export_training_data.py --validation-passed --min-itemsets 5
```

### Export to HuggingFace Format
```bash
python src/training/create_hf_dataset.py --input data/training_v2 --output data/hf_dataset_v2
```

### Upload to Hub
```bash
python src/training/upload_dataset_to_hf.py --dataset-path data/hf_dataset_v2
```

## Export Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--validation-passed` | true | Only export validated runs |
| `--min-itemsets` | 5 | Minimum itemsets per example |
| `--output-dir` | data/training_v2 | Output directory |
| `--db-path` | runs.db | Database path |

## Training Example Format

### ChatML Structure
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert at extracting frequent itemsets from CSV data..."
    },
    {
      "role": "user", 
      "content": "Extract frequent itemsets from this CSV:\n\nproduct,category,brand\nApple,Fruit,FreshCo\n..."
    },
    {
      "role": "assistant",
      "content": "<think>\nLet me analyze this CSV data...\n1. First, I'll identify all items...\n2. Then count co-occurrences...\n</think>\n\n[{\"itemset\": [\"Apple\", \"Banana\"], \"count\": 5, ...}]"
    }
  ]
}
```

### Chain-of-Thought Format
```
<think>
Let me analyze this CSV data step by step.

1. **Item Identification**: The dataset has 15 rows and 10 columns...
2. **Frequency Counting**: Looking for items appearing 3+ times...
3. **Co-occurrence Analysis**: Finding pairs that appear together...
4. **Itemset Construction**: Building itemsets with counts and row evidence...
</think>

[{"itemset": [...], "count": N, "rows": [...], "support": X.XXX}]
```

## Output Structure

```
data/training_v2/
├── ds_0001_15x10.json    # Individual example
├── ds_0002_20x12.json
├── ...
└── manifest.json          # Index of all examples
```

## HuggingFace Dataset Format

After `src/training/create_hf_dataset.py`:

```
data/hf_dataset_v2/
├── dataset_dict.json
├── train/
│   ├── data-00000-of-00001.arrow
│   └── state.json
└── validation/
    ├── data-00000-of-00001.arrow
    └── state.json
```

### Dataset Statistics
```python
from datasets import load_from_disk

ds = load_from_disk("data/hf_dataset_v2")
print(f"Train: {len(ds['train'])} examples")
print(f"Validation: {len(ds['validation'])} examples")
```

## Quality Criteria

### Include Example If:
- ✅ `validation_passed = 1`
- ✅ `apriori_itemsets_count >= 5`
- ✅ `llm_itemsets_count >= 1`
- ✅ Valid JSON in both Apriori and LLM output

### Exclude Example If:
- ❌ Validation failed
- ❌ Too few itemsets (not enough signal)
- ❌ Empty LLM response
- ❌ Missing artifacts

## Training Split

Default: 90% train, 10% validation

```python
# In create_hf_dataset.py
train_ratio = 0.9
```

## Verify Export

```bash
# Check example count
ls data/training_v2/*.json | wc -l

# Inspect single example
cat data/training_v2/ds_0001_15x10.json | jq '.messages[2].content' | head -20

# Validate JSON format
python -c "import json; [json.load(open(f)) for f in glob.glob('data/training_v2/*.json')]"
```

## Integration with Training

```bash
# Full workflow
python src/training/export_training_data.py --validation-passed
python src/training/create_hf_dataset.py --input data/training_v2 --output data/hf_dataset_v2
python src/training/upload_dataset_to_hf.py --dataset-path data/hf_dataset_v2
python src/training/run_sft_full.py  # Uses dataset from Hub
```

## Troubleshooting

### No examples exported
- Check validation pass rate: `sqlite3 runs.db "SELECT AVG(validation_passed) FROM runs"`
- Lower `--min-itemsets` threshold
- Verify artifacts exist in `artifacts/` directories

### Malformed CoT
- Check `extractor_system_prompt.md` for prompt structure
- Verify LLM output has `<think>` tags
- Post-process to add CoT if missing

### Dataset too small
- Generate more datasets
- Run more pipeline batches
- Lower quality thresholds (carefully)
