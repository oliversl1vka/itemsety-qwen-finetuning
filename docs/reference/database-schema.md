# Database Schema

`runs.db` is the SQLite database that records every pipeline run. It serves as the single source of truth for the entire project.

## Table: `runs`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key, auto-increment |
| `timestamp` | TEXT | ISO 8601 run timestamp |
| `python_version` | TEXT | Python version used for the run |
| `data_path` | TEXT | Path to the input CSV file |
| `dataset_id` | TEXT | `<name>:<hash>` identifier |
| `dataset_name` | TEXT | CSV filename without path |
| `dataset_hash` | TEXT | SHA-256[:12] of CSV content |
| `dataset_size_rows` | INTEGER | Number of rows in the dataset |
| `dataset_size_bytes` | INTEGER | File size in bytes |
| `min_support` | INTEGER | Apriori min_support parameter |
| `max_size` | INTEGER | Apriori max itemset size |
| `llm_full` | INTEGER | 1 if LLM extraction was run, 0 otherwise |
| `llm_chunk_size` | INTEGER | Rows per LLM API chunk |
| `apriori_itemset_count` | INTEGER | Number of itemsets found by Apriori |
| `llm_itemset_count` | INTEGER | Number of itemsets found by LLM |
| `validation_passed` | INTEGER | 1 if all invariants passed, 0 otherwise |
| `apriori_valid_ratio` | REAL | Fraction of Apriori itemsets passing validation |
| `llm_valid_ratio` | REAL | Fraction of LLM itemsets passing validation |
| `apriori_errors` | INTEGER | Number of validation errors in Apriori output |
| `llm_errors` | INTEGER | Number of validation errors in LLM output |
| `run_duration_ms` | INTEGER | Total run time in milliseconds |
| `invariants` | TEXT | JSON string of invariant check results |
| `apriori_output_path` | TEXT | Path to Apriori JSON output artifact |
| `llm_output_path` | TEXT | Path to LLM JSON output artifact |
| `validation_report_path` | TEXT | Path to validation report |
| `error_message` | TEXT | Error message if the run failed |
| `llm_model` | TEXT | Model ID (e.g., `gpt-4.1-mini`) |

## Dynamic Schema

The schema supports runtime evolution via `ALTER TABLE`. New columns are added automatically when the pipeline introduces new metadata fields (`pipeline.py:282-301`). Existing rows receive NULL for new columns.

## Key Queries

**SFT data selection** (one valid Apriori run per dataset):
```sql
SELECT * FROM runs
WHERE apriori_itemset_count > 0
GROUP BY dataset_id
```

**DPO data selection** (real LLM failures):
```sql
SELECT * FROM runs
WHERE validation_passed = 0
  AND llm_itemset_count > 0
```

**Run count by model**:
```sql
SELECT llm_model, COUNT(*) as runs, AVG(validation_passed) as pass_rate
FROM runs
WHERE llm_full = 1
GROUP BY llm_model
```

## Source

- `pipeline.py:252-336` -- `persist_run_to_sqlite` function
