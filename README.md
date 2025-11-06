# Frequent Itemset Extraction Pipeline (Apriori + Azure OpenAI LLM)

Simplified, validation‑focused pipeline for frequent itemset extraction. It mines deterministic Apriori itemsets and (with valid Azure OpenAI credentials) performs LLM extraction over either the full dataset (chunked) or an initial sample.

Removed: classification metrics, TP/FP/FN, overlap reporting. Added: batch processing, hashed artifact naming, robust validation invariants, SQLite persistence with auto migration and per‑run metadata.

## Components

- `pipeline.py` – Main script (supports single file or batch directory processing, validation, persistence, cleanup).
- `dataset.csv` – Example source data. Loader treats every non-empty cell as an item in wide format (including numeric values).
- `extractor_system_prompt.md` – Strict JSON extraction system prompt used for LLM calls.
- `runs.db` – SQLite database storing one row per run (auto-migration of new columns + indices).
- `azure.env` – Example environment variable file (DO NOT COMMIT real credentials).
- Generated artifacts (single run, hashed naming) organized by subdirectories:
   - `apriori_outputs/apriori_output_<stem>_<hashprefix>.json`
   - `extractor_outputs/extractor_output_<stem>_<hashprefix>.json`
   - `validation_reports/validation_report_<stem>_<hashprefix>.json`
   - `db_prepared/db_prepared_<stem>_<hashprefix>.json` (run summary)
   - (Optional) legacy generic files removed if `--cleanup-old` used.

## Data Considerations

- Every non-empty cell (including numeric values) becomes an item when using wide format, which inflates low-information singleton itemsets.
- Row identifiers are normalized as `Row N` internally; LLM evidence rows are reconciled to the same format.
- Using sample mode (`--llm-full` omitted) limits LLM visibility and reduces recall vs Apriori. Enable full mode for better coverage.

## Core Functions

### `load_transactions_csv(path)`
Detects CSV format (long / wide / single-column) and builds transaction lists. (Future improvement: filter numeric columns when wide.)

### `apriori_frequent_itemsets(transactions, min_support, max_size)`
Standard Apriori (join + prune). Produces itemsets with count ≥ `min_support` up to `max_size`.

### Validation
Enforces invariants: count equals unique evidence rows; each evidence row contains all items; Apriori support matches count/total_rows; no itemset below min_support; row labels canonical.

### LLM Extraction
Chunked prompting over transactions using Azure OpenAI (LangChain). Counts recomputed from unique canonical evidence rows to prevent inflation. If credentials are missing, script aborts (no singleton fallback).

## Recommended Improvements

1. Transaction Normalization
   - When wide format detected, include only object/string columns OR an allowlist like: `States`, `Geopolitical`.
   - Exclude numeric-like tokens using a regex or dtype check.
2. Row Identifier Consistency
   - Use common row IDs (e.g., integer index or "Row N") in BOTH Apriori and LLM extraction.
3. Full-Data LLM Extraction (Implemented)
   - Use `--llm-full` flag plus `--llm-chunk-size` to stream entire dataset to the LLM.
4. Enhanced Validation (Partial)
   - Apriori ensures counts and row lists; LLM aggregation recomputes counts from evidence.
5. Modularity
   - Extract notebook logic into `pipeline.py` for scriptable runs and unit testing.
6. Secrets Hygiene
   - Remove/rotate exposed API key in `azure.env`. Add `.gitignore` entry (provided) to prevent committing.
   - Use environment variable injection (`setx` or `.env` not tracked by VCS).
7. Requirements & Reproducibility
   - Provide `requirements.txt` for consistent environments.
8. Testing
   - Add unit tests for: CSV loader format detection; Apriori counts; canonicalization.
9. Prompt Robustness
   - Consider adding explicit instruction to reference all provided transactions (not just sample) or add an iterative multi-turn approach.
10. Performance / Scaling
   - For larger datasets, consider FP-Growth or transaction compression to reduce candidate enumeration cost.

## Setup
Install dependencies (Python 3.10+ recommended):
```powershell
pip install -r requirements.txt
```
Set Azure OpenAI environment variables (do not commit real secrets):
```powershell
$Env:AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
$Env:AZURE_OPENAI_API_KEY="<your-key>"
$Env:AZURE_OPENAI_API_VERSION="2024-08-01-preview"
$Env:AZURE_OPENAI_CHAT_DEPLOYMENT="<deployment-name>"
```

## Running the Script (`pipeline.py`)
Single file – full LLM:
```powershell
python pipeline.py --data dataset.csv --min-support 3 --max-size 3 --llm-full --llm-chunk-size 50
```

Single file – sample (first chunk only):
```powershell
python pipeline.py --data dataset.csv --min-support 3 --max-size 3 --llm-chunk-size 20
```

Batch directory (all CSV):
```powershell
python pipeline.py --data-dir datasets --min-support 3 --max-size 3 --llm-full --llm-chunk-size 50
```

Batch + cleanup generic artifacts:
```powershell
python pipeline.py --data-dir datasets --min-support 3 --max-size 3 --llm-full --cleanup-old
```

Flags:
- `--data` single CSV path (ignored if `--data-dir` provided)
- `--data-dir` directory containing multiple CSV files (batch mode)
- `--min-support` minimum transaction count threshold
- `--max-size` Apriori maximum itemset cardinality
- `--llm-full` run LLM over all transactions (else only first chunk)
- `--llm-chunk-size` rows per LLM prompt
- `--system-prompt` custom system prompt file
- `--sqlite-db` SQLite database file (default `runs.db`)
- `--disable-db` skip persistence (still writes JSON summaries)
- `--cleanup-old` remove generic non-hash artifact files after each run

## Run Summary & Persistence

Each run produces a hashed summary file (`db_prepared_<stem>_<hashprefix>.json`). Unless `--disable-db`, a row is inserted into `runs.db`.

Tracked fields:
- `timestamp` (UTC ISO8601)
- `python_version`
- `data_path`, `dataset_id`, `dataset_name`, `dataset_hash`, `dataset_size_rows`, `dataset_size_bytes`
- `min_support`, `max_size`, `llm_full`, `llm_chunk_size`
- `apriori_itemset_count`, `llm_itemset_count`
- `validation_passed`, `apriori_valid_ratio`, `llm_valid_ratio`, `apriori_errors`, `llm_errors`
- `run_duration_ms`
- `invariants` (validation rule list)
- `apriori_output_path`, `llm_output_path`, `validation_report_path`, `summary_path` (all point to subdirectories)
- `error_message` (NULL if success; populated on exception)

Schema Migration:
New columns are added transparently via `ALTER TABLE` if missing. Indices exist on `timestamp`, `validation_passed`, and `dataset_id` for efficient querying.

Query examples (PowerShell):
```powershell
sqlite3 runs.db "SELECT dataset_id, COUNT(*) AS runs, AVG(run_duration_ms) FROM runs GROUP BY dataset_id;"
sqlite3 runs.db "SELECT * FROM runs ORDER BY timestamp DESC LIMIT 5;"
sqlite3 runs.db "SELECT id, dataset_id, error_message FROM runs WHERE error_message IS NOT NULL ORDER BY id DESC LIMIT 10;"
```

Exit codes:
- 0: At least one successful run (even if some failures)
- 1: Batch completed, but all runs failed
- 2: Invalid `--data-dir` path
- 3: Missing Azure OpenAI credentials (LLM required; no singleton fallback)

## Security Notice
The `azure.env` file currently contains a real-looking API key. Remove or rotate it immediately. NEVER commit real secrets. Consider using Azure Key Vault or local secret management.

## Next Steps (Optional)
- Implement improved loader filtering (exclude numeric-like tokens).
- Add unit tests & CI.
- Add FP-Growth alternative.
- Integrate summarization for top itemsets.

## License
Specify a license here if distributing (e.g., MIT). Currently unspecified.

---
README aktualizovaný podľa aktuálnej funkcionality (hashované artefakty, batch mód, validačné invarianty, povinné LLM kredity).
