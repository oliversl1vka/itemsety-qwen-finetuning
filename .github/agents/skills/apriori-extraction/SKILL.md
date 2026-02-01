---
name: apriori-extraction
description: Run Apriori algorithm to extract frequent itemsets from CSV transactions. Use as ground truth for LLM training and validation.
---

# Apriori Frequent Itemset Extraction

Extract frequent itemsets using the deterministic Apriori algorithm.

## Overview

The Apriori algorithm provides **ground truth** for:
- Training data labels (what the LLM should learn to output)
- Validation (comparing LLM output against correct answer)
- Performance benchmarking

## Quick Start

### Single Dataset
```bash
python pipeline.py --data data/datasets_v2/ds_0001_15x10.csv --min-support 3 --max-size 3
```

### Batch Processing
```bash
python pipeline.py --data-dir data/datasets_v2 --min-support 3 --max-size 3
```

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-support` | 3 | Minimum occurrence count |
| `--max-size` | 3 | Maximum itemset size (1=singletons, 2=pairs, 3=triples) |
| `--data` | — | Single CSV file path |
| `--data-dir` | — | Directory of CSV files |

## Output Format

```json
[
  {
    "itemset": ["Apple", "Banana"],
    "count": 5,
    "rows": ["Row 1", "Row 3", "Row 5", "Row 8", "Row 10"],
    "support": 0.333
  }
]
```

### Fields

- **itemset:** Sorted list of items (alphabetical)
- **count:** Number of transactions containing all items
- **rows:** List of row labels where itemset appears
- **support:** count / total_rows

## Invariants (Must Hold True)

1. `count == len(rows)` — Count matches row list length
2. `support == count / total_rows` — Support correctly calculated
3. All items in itemset exist in original dataset
4. Row labels are valid ("Row N" format)
5. No duplicate items within itemset
6. Itemsets are sorted alphabetically
7. `count >= min_support` — All itemsets meet threshold

## Output Artifacts

| File | Location | Description |
|------|----------|-------------|
| Apriori output | `artifacts/apriori_outputs/gpt_4_1_apriori_output_{stem}_{hash}.json` | Full itemset list |
| Log | `logs/apriori/gpt_4_1_apriori_generation_log_{stem}_{hash}.json` | Execution metadata |

## Performance

| Dataset Size | Expected Duration |
|--------------|-------------------|
| 5x5 | < 0.1s |
| 15x10 | < 0.5s |
| 25x20 | < 2s |
| 50x30 | < 10s |

## Troubleshooting

### No itemsets found
- Lower `--min-support` (try 2)
- Check dataset has repeated patterns
- Verify CSV format (categorical, not numeric)

### Too many itemsets (>1000)
- Increase `--min-support`
- Reduce `--max-size`
- Filter dataset columns

### Memory issues
- Reduce `--max-size` to 2
- Process datasets individually, not batch

## Code Reference

Core function in `pipeline.py`:
```python
def apriori_frequent_itemsets(transactions, min_support, max_size):
    """
    Level-wise Apriori with row tracking.
    Returns list of {itemset, count, rows, support}.
    """
```
