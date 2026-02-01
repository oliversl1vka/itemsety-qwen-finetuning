# AI Coding Agent Instructions

Concise guidance for working productively in this frequent itemset mining + LLM fine-tuning project.

## Repository Structure (Updated 2026-02-01)

```
itemsety-qwen-finetuning/
├── pipeline.py                    # Core extraction pipeline (Apriori + LLM)
├── src/
│   ├── training/                  # Fine-tuning scripts
│   │   ├── run_sft_full.py        # Production training
│   │   ├── run_sft_test.py        # Test training  
│   │   ├── export_training_data.py
│   │   ├── create_hf_dataset.py
│   │   └── upload_dataset_to_hf.py
│   ├── evaluation/
│   │   └── eval_finetuned_model.py
│   ├── data_generation/
│   │   ├── generate_datasets_v2.py
│   │   └── generate_eval_datasets_v2.py
│   └── utils/
│       ├── visualization.py
│       ├── compute_stats.py
│       └── inspect_training_data.py
├── data/
│   ├── datasets_v2/               # CSV datasets (500)
│   ├── training_v2/               # Training examples (ChatML)
│   └── hf_dataset_v2/             # HuggingFace format
├── docs/
│   ├── guides/
│   └── reports/
├── scripts/
│   ├── deployment/                # HF deployment scripts
│   ├── colab/                     # Google Colab scripts
│   └── db_maintenance/            # SQLite utilities
├── .github/
│   ├── agents/                    # 9 agent definition files
│   ├── agents_log/                # Agent activity logs
│   ├── agents_memory/             # Agent persistent memory
│   └── copilot-instructions.md    # This file
├── artifacts/                     # Pipeline outputs (gitignored)
├── logs/                          # Execution logs (gitignored)
└── runs.db                        # SQLite database (gitignored)
```

## Project Essence
- Pipeline (`pipeline.py`) runs Apriori (deterministic) + LLM extraction (Azure OpenAI via LangChain) over CSV datasets (single file or batch directory).
- Artifacts are hash-suffixed JSONs in `artifacts/` directory: `apriori_outputs/`, `extractor_outputs/`, `validation_reports/`, `db_prepared/` plus generation logs in `logs/<kind>/`.
- SQLite persistence (`runs.db`) stores run metadata; auto-migrations add new columns (e.g. `llm_model`).

## Key Files
- `pipeline.py` – Orchestrates load → Apriori → LLM → validation → persistence.
- `src/data_generation/generate_datasets_v2.py` – Semi-human dataset factory (500 datasets); logs to `data/datasets_v2/generation_log.json`.
- `src/utils/visualization.py` – Comparative plots Apriori vs LLM (uses `runs.db`).
- `src/training/run_sft_full.py` – Production fine-tuning script.
- `src/evaluation/eval_finetuned_model.py` – Model evaluation script.
- `requirements.txt` – Core runtime deps (pandas, langchain, matplotlib, sqlite via stdlib).

## Naming & IDs
- Dataset file pattern: `ds_<NNNN>_<rows>x<cols>.csv` → hash (SHA256 first 12 chars) used in artifact filenames.
- Dataset generation log: `logs/generation_log.csv` contains metadata for all generated datasets (id, file, rows, cols, hash, timestamp).
- Pipeline generation log pattern: `gpt_<model>_<kind>_generation_log_<dataset>_<hash>.json` (only gpt_4_1 currently real); synthetic gpt_5_mini logs flagged by creation scripts.
- Row labels normalized to `Row N` for validation invariants.

## Workflow Patterns
1. Loading: `load_transactions_csv` auto-detects format (long/wide/single-column); wide = all columns treated as items (numeric included).
2. Apriori: `apriori_frequent_itemsets` builds levels up to `--max-size`; counts & rows maintained; support recalculated as `count/total_rows`.
3. LLM: `llm_extract_full` chunked invocation (prompt requires strict JSON array). Missing credentials -> abort (exit code 3), no fallback generation.
4. Validation: `validate_all` ensures invariants (support math, item presence, row label integrity, count/unique row alignment).
5. Persistence: `build_run_summary` then `persist_run_to_sqlite`; new columns via ALTER TABLE if absent; indices on timestamp, validation_passed, dataset_id.
6. Cleanup: `--cleanup-old` removes generic artifact base names without hash.

## Conventions
- All artifact outputs are organized under `artifacts/` root directory with subdirectories for each stage.
- Always create output directories before writing (artifacts/apriori_outputs, artifacts/extractor_outputs, artifacts/validation_reports, artifacts/db_prepared). Existing code does this per run using `parents=True, exist_ok=True`.
- Time stamps: use `datetime.now(UTC)`; dataset hash prefix = first 12 chars of SHA256.
- Exit codes (in `pipeline.py`): 0 success (even partial), 1 all failed, 2 bad data-dir, 3 missing LLM creds.
- `llm_model` supplied via `--llm-model` or env `LLM_MODEL`; include in future enhancements to JSON summaries if needed.

## Extending Safely
- To add new metrics: extend `build_run_summary` then update `persist_run_to_sqlite` new_cols list; run automatically migrates schema.
- To support new model prefixes: update generation log filename construction and pass `--llm-model`.
- For performance: consider replacing Apriori with FP-Growth for high column counts.
- When modifying validation logic ensure invariants list stays in sync and update summary JSON + DB fields.

## Typical Commands (PowerShell)
```powershell
python pipeline.py --data data/datasets_v2/ds_0001_5x53.csv --min-support 3 --max-size 3 --llm-full --llm-model gpt_4_1
python pipeline.py --data-dir data/datasets_v2 --min-support 3 --max-size 3 --llm-full --llm-chunk-size 50 --llm-model gpt_4_1
python src/training/run_sft_full.py
python src/evaluation/eval_finetuned_model.py --model-path OliverSlivka/qwen2.5-3b-itemset-extractor
python src/utils/visualization.py --db runs.db --outdir visuals --bins 5
```

## Pitfalls / Gotchas
- Wide CSVs inflate item count; numeric values become items—filter if you introduce large numeric ranges.
- Synthetic gpt_5_mini generation logs are not real extraction metrics; avoid using them for validation comparisons.
- LLM extraction aborts entirely if any credential is missing (no fallback) — handle with pre-run env checks when scripting batch runs.
- Duplicate evidence rows in LLM responses are deduplicated; counts must match unique rows or validation fails.

## What NOT To Do
- Do not hardcode secrets; rely on env variables / `azure.env` (never commit real keys).
- Do not rename artifact directories within `artifacts/` structure; many scripts assume current organization.
- Avoid modifying Apriori output schema (itemset/count/rows/support) unless you also adjust validation & visualization.

## Quick Enhancement Ideas
- Add filtering step in `load_transactions_csv` to exclude purely numeric tokens.
- Record per-stage timing (Apriori vs LLM vs validation) into DB for longitudinal performance analysis.

## Security Rules
- **NEVER** commit `azure.env`, `openai.env`, or any file with real API keys
- **NEVER** hardcode API keys, tokens, or credentials in code
- **ALWAYS** use environment variables for secrets
- Files to NEVER modify or read sensitive content from:
  - `azure.env` (local only, gitignored)
  - `openai.env` (local only, gitignored)
  - `runs.db` (local only, gitignored)
- Template files are safe: `azure.env.template`, `openai.env.template`

## Agent System
This repo uses 9 specialized agents in `.github/agents/`:
- **orchestrator.md** - Master workflow coordinator
- **pipeline-agent.md** - Apriori + LLM extraction
- **training-agent.md** - Model fine-tuning
- **evaluation-agent.md** - Model evaluation
- **deployment-agent.md** - HuggingFace deployment
- **monitoring-agent.md** - Metrics & visualization
- **dataset-agent.md** - Dataset generation
- **maintainer-agent.md** - Documentation maintenance
- **cleanup-agent.md** - Repository hygiene

Skills are in `.github/agents/skills/`, logs in `.github/agents_log/`, memory in `.github/agents_memory/`.

---
Provide feedback if model handling, synthetic log differentiation, or validation invariants need deeper clarification.
