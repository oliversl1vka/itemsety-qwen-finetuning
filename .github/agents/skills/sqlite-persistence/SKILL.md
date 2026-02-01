---
name: sqlite-persistence
description: Persist run metadata to SQLite database for tracking, querying, and analysis. Use for all database operations on runs.db.
---

# SQLite Run Persistence

Store and query pipeline run metadata in SQLite database.

## Overview

`runs.db` is the **single source of truth** for:
- Run history and metadata
- Validation status tracking
- Performance metrics over time
- Training data selection

## Database Location

```
/Users/oliver/itemsety-qwen-finetuning/runs.db
```

## Schema

```sql
CREATE TABLE runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    validation_passed INTEGER DEFAULT 0,
    llm_model TEXT,
    apriori_itemsets_count INTEGER,
    llm_itemsets_count INTEGER,
    apriori_duration_s REAL,
    llm_duration_s REAL,
    validation_duration_s REAL,
    total_duration_s REAL,
    min_support INTEGER,
    max_size INTEGER,
    total_rows INTEGER,
    total_items INTEGER,
    hash TEXT
);

CREATE INDEX idx_timestamp ON runs(timestamp);
CREATE INDEX idx_validation ON runs(validation_passed);
CREATE INDEX idx_dataset ON runs(dataset_id);
CREATE INDEX idx_model ON runs(llm_model);
```

## Common Queries

### Recent Runs
```sql
SELECT dataset_id, validation_passed, llm_itemsets_count, timestamp
FROM runs 
ORDER BY timestamp DESC 
LIMIT 10;
```

### Validation Statistics
```sql
SELECT 
  COUNT(*) as total,
  SUM(validation_passed) as passed,
  ROUND(AVG(CAST(validation_passed AS FLOAT)) * 100, 1) as pass_rate
FROM runs;
```

### Performance by Model
```sql
SELECT 
  llm_model,
  COUNT(*) as runs,
  AVG(llm_duration_s) as avg_duration,
  AVG(CAST(validation_passed AS FLOAT)) * 100 as pass_rate
FROM runs
GROUP BY llm_model;
```

### Failed Validations
```sql
SELECT dataset_id, timestamp, llm_model
FROM runs 
WHERE validation_passed = 0
ORDER BY timestamp DESC;
```

### Training Data Candidates
```sql
SELECT dataset_id, apriori_itemsets_count, llm_itemsets_count
FROM runs 
WHERE validation_passed = 1 
  AND apriori_itemsets_count >= 5
ORDER BY timestamp DESC;
```

## Auto-Migration

New columns are added automatically:
```python
def persist_run_to_sqlite(run_data, db_path="runs.db"):
    # Automatically adds missing columns via ALTER TABLE
```

To add new column manually:
```bash
python migrate_add_model_column.py --db runs.db
```

## Python Access

### Query
```python
import sqlite3

conn = sqlite3.connect("runs.db")
conn.row_factory = sqlite3.Row

cursor = conn.execute("""
    SELECT * FROM runs 
    WHERE validation_passed = 1 
    LIMIT 10
""")

for row in cursor:
    print(dict(row))

conn.close()
```

### Insert
```python
def insert_run(run_data):
    conn = sqlite3.connect("runs.db")
    columns = ", ".join(run_data.keys())
    placeholders = ", ".join(["?" for _ in run_data])
    
    conn.execute(f"""
        INSERT INTO runs ({columns})
        VALUES ({placeholders})
    """, list(run_data.values()))
    
    conn.commit()
    conn.close()
```

## Backup

```bash
# Create backup
cp runs.db runs_backup_$(date +%Y%m%d).db

# Or use SQLite backup command
sqlite3 runs.db ".backup runs_backup.db"
```

## Integrity Checks

```bash
# Check database integrity
sqlite3 runs.db "PRAGMA integrity_check;"

# Check for orphaned records
sqlite3 runs.db "SELECT COUNT(*) FROM runs WHERE dataset_id IS NULL;"
```

## GUI Access

Use DB Browser for SQLite or VS Code SQLite extension:
```bash
# macOS
brew install --cask db-browser-for-sqlite
```

Or use included editor:
```bash
python db_editor.py
```

## Data Export

### To CSV
```bash
sqlite3 -header -csv runs.db "SELECT * FROM runs" > runs_export.csv
```

### To JSON
```python
import sqlite3
import json

conn = sqlite3.connect("runs.db")
conn.row_factory = sqlite3.Row
cursor = conn.execute("SELECT * FROM runs")
rows = [dict(row) for row in cursor]
print(json.dumps(rows, indent=2))
```

## Troubleshooting

### Database locked
**Cause:** Another process has write lock
**Fix:** Close other connections, use WAL mode

```sql
PRAGMA journal_mode=WAL;
```

### Missing columns
**Cause:** Schema outdated
**Fix:** Run migration script

### Slow queries
**Cause:** Missing indices
**Fix:** Create index on frequently queried columns
