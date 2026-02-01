---
name: validation-pipeline
description: Validate extracted itemsets against 13 invariants ensuring data quality. Use after any extraction (Apriori or LLM) to verify correctness.
---

# Itemset Validation Pipeline

Comprehensive validation ensuring extracted itemsets meet all quality invariants.

## Overview

Validation is **critical** for:
- Training data quality (bad labels = bad model)
- Identifying LLM extraction errors
- Ensuring Apriori implementation correctness
- Filtering valid runs for training export

## Quick Start

Validation runs automatically in pipeline:
```bash
python pipeline.py --data data/datasets_v2/ds_0001_15x10.csv --llm-full
# Validation happens after extraction
```

## The 13 Invariants

### Format Invariants
| # | Invariant | Description |
|---|-----------|-------------|
| 1 | JSON_PARSEABLE | Response is valid JSON |
| 2 | ARRAY_FORMAT | Top-level is array, not object |
| 3 | REQUIRED_FIELDS | Each itemset has: itemset, count, rows, support |

### Consistency Invariants
| # | Invariant | Description |
|---|-----------|-------------|
| 4 | COUNT_MATCHES_ROWS | count == len(rows) |
| 5 | SUPPORT_CORRECT | support == count / total_rows |
| 6 | ITEMS_EXIST | All items in original dataset |
| 7 | ROWS_VALID | Row labels are "Row N" format |

### Integrity Invariants
| # | Invariant | Description |
|---|-----------|-------------|
| 8 | NO_DUPLICATE_ITEMS | No repeated items in single itemset |
| 9 | ITEMSET_SORTED | Items alphabetically sorted |
| 10 | NO_DUPLICATE_ROWS | No repeated rows in evidence |

### Threshold Invariants
| # | Invariant | Description |
|---|-----------|-------------|
| 11 | MIN_SUPPORT_MET | count >= min_support |
| 12 | MAX_SIZE_MET | len(itemset) <= max_size |
| 13 | NO_HALLUCINATIONS | No items invented by LLM |

## Validation Report Format

```json
{
  "dataset_id": "ds_0001_15x10",
  "validation_passed": true,
  "total_itemsets": 25,
  "valid_itemsets": 25,
  "invalid_itemsets": 0,
  "failures": [],
  "invariants_checked": 13,
  "invariants_passed": 13,
  "timestamp": "2026-02-01T10:30:00Z"
}
```

### Failure Entry
```json
{
  "invariant": "COUNT_MATCHES_ROWS",
  "itemset": ["Apple", "Banana"],
  "expected": 5,
  "actual": 4,
  "message": "Count (5) does not match row list length (4)"
}
```

## Output Artifacts

| File | Location |
|------|----------|
| Validation report | `artifacts/validation_reports/gpt_4_1_validation_report_{stem}_{hash}.json` |
| DB-prepared summary | `artifacts/db_prepared/gpt_4_1_db_prepared_{stem}_{hash}.json` |

## Database Persistence

Validation results stored in `runs.db`:
```sql
SELECT dataset_id, validation_passed, 
       apriori_itemsets_count, llm_itemsets_count
FROM runs 
WHERE validation_passed = 1
ORDER BY timestamp DESC;
```

## Filtering for Training

Only export validated runs:
```bash
python src/training/export_training_data.py --validation-passed --min-itemsets 5
```

## Common Validation Failures

### COUNT_MATCHES_ROWS
**Cause:** LLM duplicated or omitted row labels
**Fix:** Deduplicate rows in post-processing

### NO_HALLUCINATIONS
**Cause:** LLM invented items not in dataset
**Fix:** Filter itemsets to only include known items

### SUPPORT_CORRECT
**Cause:** LLM calculated support incorrectly
**Fix:** Recalculate support from count and total_rows

### ITEMSET_SORTED
**Cause:** LLM returned unsorted items
**Fix:** Sort itemset alphabetically in post-processing

## Validation Statistics

Query validation pass rate:
```sql
SELECT 
  COUNT(*) as total_runs,
  SUM(validation_passed) as passed,
  AVG(CAST(validation_passed AS FLOAT)) * 100 as pass_rate
FROM runs;
```

## Integration with Training

1. Run pipeline with validation
2. Export only validated runs
3. Create HF dataset from validated data
4. Train model on clean data

```bash
# Full workflow
python pipeline.py --data-dir data/datasets_v2 --llm-full
python src/training/export_training_data.py --validation-passed
python src/training/create_hf_dataset.py --input data/training_v2 --output data/hf_dataset_v2
```
