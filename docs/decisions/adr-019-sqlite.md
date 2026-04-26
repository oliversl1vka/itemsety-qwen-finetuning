# ADR-019: SQLite Persistence

**Status:** Accepted  
**Date:** 2026-02

## Context

Pipeline run metadata (1600+ runs) must be persisted for querying during training data generation and analysis.

## Options Considered

| Storage | Setup | Query Support | Dependencies | Portability |
|---------|-------|--------------|-------------|-------------|
| CSV logs | None | Limited (pandas) | None | High |
| Parquet | None | Good (pandas/pyarrow) | pyarrow | High |
| PostgreSQL | Server required | Full SQL | psycopg2 + server | Low |
| **SQLite** | None (file-based) | Full SQL | Standard library | High |

## Decision

**SQLite** (`runs.db`) as the sole persistence layer.

## Rationale

- **Zero setup**: Single file, no server, no network dependency. `python -c "import sqlite3"` works everywhere.
- **Standard library**: Python's `sqlite3` module requires no additional dependencies
- **Full SQL queries**: Both training data generation scripts use complex WHERE clauses:
    - SFT: `SELECT * FROM runs WHERE apriori_itemset_count > 0 GROUP BY dataset_id`
    - DPO: `SELECT * FROM runs WHERE validation_passed = 0 AND llm_itemset_count > 0`
- **Dynamic schema**: `ALTER TABLE runs ADD COLUMN` enables evolving the schema without migration scripts (see `pipeline.py:282-301` for runtime column addition)
- **Gitignored**: Contains local paths, is regenerable from the pipeline, no size or privacy concerns

## Trade-offs

- Single-writer only (fine for sequential pipeline execution)
- Not suitable for cloud/distributed workloads
- No built-in replication or backup

## Source Evidence

- `pipeline.py:252-336` -- `persist_run_to_sqlite` with 27-column schema
- `src/training/generate_cot_sft_data.py` -- queries runs.db for SFT data
- `src/training/export_real_dpo_data.py` -- queries runs.db for DPO data
