---
name: csv-dataset-generation
description: Generate synthetic CSV datasets optimized for frequent itemset mining. Use when creating training data, test datasets, or expanding dataset collection for LLM fine-tuning.
---

# CSV Dataset Generation

Generate high-quality synthetic CSV datasets for frequent itemset extraction training.

## Overview

This skill creates CSV datasets with:
- Configurable size (5-25 rows, 5-20 columns optimal for LLM context)
- Semi-realistic item distributions (real-world column names, meaningful patterns)
- Reproducible generation via seeding
- Automatic quality validation and metadata logging

## Workflow

### 1. Generate Single Dataset
```bash
python src/data_generation/generate_datasets_v2.py --count 1 --rows 15 --cols 10 --seed 42
```

### 2. Generate Batch (500 datasets)
```bash
python src/data_generation/generate_datasets_v2.py --count 500 --output-dir data/datasets_v2
```

### 3. Verify Generation
```bash
# Check generation log
cat data/datasets_v2/generation_log.json | jq '.[-1]'

# Validate dataset
python -c "import pandas as pd; print(pd.read_csv('data/datasets_v2/ds_0001_15x10.csv').shape)"
```

## Dataset Naming Convention

```
ds_{ID:04d}_{rows}x{cols}.csv
```

Examples:
- `ds_0001_5x8.csv` — Dataset #1, 5 rows, 8 columns
- `ds_0042_20x15.csv` — Dataset #42, 20 rows, 15 columns

## Quality Criteria

A valid dataset must have:
- **Rows:** 5-25 (optimal for LLM context window)
- **Columns:** 5-20 (prevents item explosion)
- **No empty cells** (all values populated)
- **No duplicate rows** (unique transactions)
- **Categorical values** (no pure numeric columns)
- **Sufficient pattern density** (itemsets extractable at min_support=3)

## Generation Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `row_subsample` | Random row selection | Smaller datasets from large |
| `col_subsample` | Random column selection | Reduce item space |
| `combined` | Both row and col subsample | Balanced reduction |
| `shuffle` | Randomize row/col order | Augmentation |
| `noise` | Add random perturbations | Robustness testing |

## Output Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| CSV files | `data/datasets_v2/ds_*.csv` | Generated datasets |
| Generation log | `data/datasets_v2/generation_log.json` | Metadata for all datasets |
| Hash index | Computed on-demand | SHA256 first 12 chars |

## Common Issues

### Issue: Too many items (>100 unique)
**Solution:** Reduce columns or use categorical binning

### Issue: No frequent itemsets found
**Solution:** Increase row count or reduce min_support threshold

### Issue: Duplicate dataset hashes
**Solution:** Change seed or generation strategy

## Integration

After generation, run pipeline:
```bash
python pipeline.py --data-dir datasets_v2 --min-support 3 --max-size 3 --llm-full
```
