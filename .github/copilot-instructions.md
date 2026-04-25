# AI Coding Agent Instructions

**Version:** 6.0 (v3 Training: SFT-CoT → DPO-Real, GRPO skipped)  
**Last Updated:** 2026-03-09

Concise guidance for working productively in this frequent itemset mining + LLM fine-tuning project.

## Multi-Agent Workflow System

This project uses **interactive multi-agent orchestration** where users switch between specialized agents to execute workflow stages.

### 🧠 Memory-First Rule (MANDATORY)

**Every agent MUST read its memory file from the Obsidian vault before starting ANY task:**

| Agent | Memory File (Obsidian) |
|-------|------------------------|
| Orchestrator | `obsidian-brain/Agents/Orchestrator.md` |
| Dataset Agent | `obsidian-brain/Agents/Dataset Agent.md` |
| Pipeline Agent | `obsidian-brain/Agents/Pipeline Agent.md` |
| Training Agent | `obsidian-brain/Agents/Training Agent.md` |
| Deployment Agent | `obsidian-brain/Agents/Deployment Agent.md` |
| Evaluation Agent | `obsidian-brain/Agents/Evaluation Agent.md` |
| Monitoring Agent | `obsidian-brain/Agents/Monitoring Agent.md` |
| Cleanup Agent | `obsidian-brain/Agents/Cleanup Agent.md` |
| Maintainer Agent | `obsidian-brain/Agents/Maintainer Agent.md` |

**Activity logs** go to `obsidian-brain/Logs/` using the Run Log template.
**Experiment reports** go to `obsidian-brain/Experiments/` using the Experiment template.
**Decisions** go to `obsidian-brain/Decisions/` using the Decision template.
**Reference docs** live in `obsidian-brain/References/` (API limits, model comparison, etc.).

Use `[[backlinks]]` in Obsidian notes to connect related knowledge across agents.

### How It Works

1. **User activates agent:** `@workspace /agents switch to <agent-name>`
2. **Agent reads memory FIRST** (never repeats past mistakes)
3. **User runs command:** Slash command (e.g., `/datasets`, `/pipeline`)
4. **Agent executes:** Runs scripts, validates outputs, updates workflow state
5. **Agent tells user:** Which agent to activate next

**See:** [.github/WORKFLOW_GUIDE.md](.github/WORKFLOW_GUIDE.md) for complete step-by-step guide

### Workflow State

**Location:** `.github/agents_memory/workflow_state.json`  
**Purpose:** Coordinate sequential execution between agents

**Python module:** `.github/agents_memory/workflow_state.py`

```python
from .github.agents_memory.workflow_state import load_workflow, complete_stage

# Load state
wf = load_workflow()
print(wf.get_status_summary())

# Complete stage
complete_stage("1_datasets", {"datasets_count": 500})
```

### Agent List

**Main Workflow Agents (8 stages):**

| Agent | Activation | Slash Commands | Stage |
|-------|-----------|---------------|-------|
| Orchestrator | `@workspace /agents switch to orchestrator` | `/organize`, `/status`, `/finalize` | 1 & 8 |
| Dataset Agent | `@workspace /agents switch to dataset-agent` | `/datasets`, `/analyze` | 2 |
| Pipeline Agent | `@workspace /agents switch to pipeline-agent` | `/pipeline`, `/validate-run` | 3 |
| Training Agent | `@workspace /agents switch to training-agent` | `/export`, `/validate` | 4 & 6 |
| Deployment Agent | `@workspace /agents switch to deployment-agent` | `/push`, `/deploy` | 5 |
| Monitoring Agent | `@workspace /agents switch to monitoring-agent` | `/visualize`, `/report` | 7 |
| Evaluation Agent | `@workspace /agents switch to evaluation-agent` | `/eval`, `/compare` | Support |

**🔧 Utility Agents (available anytime, not mandatory stages):**

| Agent | Activation | Slash Commands |
|-------|-----------|---------------|
| Cleanup Agent | `@workspace /agents switch to cleanup-agent` | `/cleanup` |
| Maintainer Agent | `@workspace /agents switch to maintainer-agent` | `/maintain` |

### Workflow Flow

```
Orchestrator → Dataset Agent → Pipeline Agent → Training Agent → Deployment Agent
  /organize      /datasets       /pipeline       /export          /push
     │                                                              │
     │          ⏸️ PAUSE: User trains & evaluates on Jupyter server    │
     │                                                              │
     └───────── Training Agent ← Monitoring Agent ← Orchestrator ───┘
       /finalize     /validate       /visualize
```

**Key:** After Stage 5 (`/push`), the workflow PAUSES. The user trains and evaluates the model on their own Jupyter server. Then resume with `/validate` to continue.

**See:** [AGENTS.md](../AGENTS.md) for detailed agent documentation

## Repository Structure (Updated 2026-03-09)

```
itemsety-qwen-finetuning/
├── pipeline.py                    # Core extraction pipeline (Apriori + LLM)
├── src/
│   ├── training/                  # Fine-tuning scripts
│   ├── evaluation/                # Model evaluation + inference utilities
│   ├── data_generation/           # Dataset generation
│   └── utils/                     # Utilities
├── data/
│   ├── datasets_v2/               # CSV datasets (500)
│   ├── sft_cot_v2.json            # SFT-CoT training examples v2 (348, verbose format)
│   ├── sft_cot_v3.json            # SFT-CoT training examples v3 (concise format)
│   ├── dpo_real_v2.json           # DPO preference pairs with real LLM failures (606)
│   └── hf_dataset_v2/             # HuggingFace dataset (3 configs: sft/dpo/grpo)
├── notebooks/                     # Training + evaluation notebooks
│   ├── training_3phase_7b.ipynb   # v3 training notebook (SFT→DPO, GRPO skipped)
│   └── training_sft_dpo_template.ipynb  # Legacy 2-phase template
├── obsidian-brain/                # 🧠 Obsidian knowledge vault
│   ├── Home.md                    # Vault navigation hub
│   ├── Agents/                    # Agent memory notes (9 files)
│   ├── References/                # API limits, model comparison, etc.
│   ├── Logs/                      # Per-run activity logs
│   ├── Experiments/               # Training experiment reports
│   ├── Decisions/                 # Architecture decisions
│   └── Templates/                 # Note templates
├── .github/
│   ├── agents/                    # 9 agent definition files
│   ├── agents_memory/             # workflow_state.json + workflow_state.py only
│   └── copilot-instructions.md    # This file
├── artifacts/                     # Pipeline outputs (gitignored)
├── logs/                          # Execution logs (gitignored)
└── runs.db                        # SQLite database (gitignored)
```

## Project Essence
- Pipeline (`pipeline.py`) runs Apriori (deterministic) + LLM extraction (OpenAI via LangChain) over CSV datasets (single file or batch directory).
- Artifacts are hash-suffixed JSONs in `artifacts/` directory: `apriori_outputs/`, `extractor_outputs/`, `validation_reports/`, `db_prepared/` plus generation logs in `logs/<kind>/`.
- SQLite persistence (`runs.db`) stores run metadata; auto-migrations add new columns (e.g. `llm_model`).

## Key Files
- `pipeline.py` – Orchestrates load → Apriori → LLM → validation → persistence.
- `src/data_generation/generate_datasets_v2.py` – Semi-human dataset factory (500 datasets).
- `src/utils/visualization.py` – Comparative plots Apriori vs LLM (uses `runs.db`).
- `src/training/training_utils.py` – Shared utilities: compact system prompt (~150 tokens), CoT generator (v3 concise format), CSV loader, ground truth formatter, token budget calculator.
- `src/training/generate_cot_sft_data.py` – Phase 1: Generate SFT-CoT examples with `<think>` reasoning (v3 concise format).
- `src/training/export_real_dpo_data.py` – Phase 2: Export DPO pairs using real LLM failures as rejected (606 pairs).
- `src/training/build_hf_dataset_v2.py` – Build HuggingFace dataset with 3 configs (sft/dpo/grpo).
- `src/training/upload_dataset_to_hf.py` – Push dataset to HuggingFace Hub.
- `notebooks/training_3phase_7b.ipynb` – v3 training notebook: SFT-CoT → DPO-Real (GRPO skipped) for Qwen2.5-7B.
- `src/evaluation/eval_finetuned_model.py` – Model evaluation script (F1/P/R vs Apriori, v3 inference fixes).
- `src/evaluation/inference_utils.py` – v3 inference utilities: StoppingCriteria, two-phase generation, dynamic token budget.
- `src/evaluation/council_advisor.py` – LLM Council multi-model analysis via OpenRouter.
- `requirements.txt` – Core runtime deps (pandas, langchain, unsloth, matplotlib, sqlite via stdlib).

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
- Training notebooks are **versioned artifacts**. Keep at least **one notebook per active day** using dated filenames like `notebooks/training_3phase_YYYY-MM-DD_vN.ipynb`. Prefer one notebook per day, and only create additional same-day versions when the notebook meaningfully changes and the earlier same-day state is worth preserving.
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

## Typical Commands (macOS/Linux)
```bash
# Run pipeline (single dataset or full batch)
python pipeline.py --data data/datasets_v2/ds_0001_7x12.csv --min-support 3 --max-size 3 --llm-full --llm-model gpt-4.1-mini
python pipeline.py --data-dir data/datasets_v2 --min-support 3 --max-size 3 --llm-full --llm-chunk-size 50 --llm-model gpt-4.1-mini

# Generate SFT-CoT training data (Phase 1)
python src/training/generate_cot_sft_data.py --db runs.db --output data/sft_cot_v2.json

# Export DPO pairs with real LLM failures (Phase 2)
python src/training/export_real_dpo_data.py --db runs.db --output data/dpo_real_v2.json

# Build HuggingFace dataset (3 configs: sft/dpo/grpo)
python src/training/build_hf_dataset_v2.py --sft data/sft_cot_v2.json --dpo data/dpo_real_v2.json --output data/hf_dataset_v2

# Push to HuggingFace (each version gets its own repo — NEVER overwrite old versions)
python src/training/upload_dataset_to_hf.py --dataset-path data/hf_dataset_v3 --repo OliverSlivka/itemset-extraction-v3

# Evaluate fine-tuned model
python src/evaluation/eval_finetuned_model.py --model-path OliverSlivka/qwen2.5-7b-itemset-extractor

# Visualize pipeline results
python src/utils/visualization.py --db runs.db --outdir visuals --bins 5
```

## Pitfalls / Gotchas
- Wide CSVs inflate item count; numeric values become items—filter if you introduce large numeric ranges.
- Synthetic gpt_5_mini generation logs are not real extraction metrics; avoid using them for validation comparisons.
- LLM extraction aborts entirely if any credential is missing (no fallback) — handle with pre-run env checks when scripting batch runs.
- Duplicate evidence rows in LLM responses are deduplicated; counts must match unique rows or validation fails.

## What NOT To Do
- Do not hardcode secrets; rely on env variables / `openai.env` (never commit real keys).
- Do not rename artifact directories within `artifacts/` structure; many scripts assume current organization.
- Avoid modifying Apriori output schema (itemset/count/rows/support) unless you also adjust validation & visualization.

## Quick Enhancement Ideas
- Add filtering step in `load_transactions_csv` to exclude purely numeric tokens.
- Record per-stage timing (Apriori vs LLM vs validation) into DB for longitudinal performance analysis.

## Security Rules
- **NEVER** commit `openai.env` or any file with real API keys
- **NEVER** hardcode API keys, tokens, or credentials in code
- **ALWAYS** use environment variables for secrets
- Files to NEVER modify or read sensitive content from:
  - `openai.env` (local only, gitignored)
  - `runs.db` (local only, gitignored)
- Template files are safe: `openai.env.template`

## Agent System
This repo uses 9 specialized agents in `.github/agents/`:

**Main Workflow Agents:**
- **orchestrator.md** - Master workflow coordinator (Stages 1 & 8)
- **dataset-agent.md** - Dataset generation + fixed eval datasets (Stage 2)
- **pipeline-agent.md** - Apriori + LLM extraction (Stage 3)
- **training-agent.md** - Export training data + versioned notebooks + receive eval results (Stages 4 & 6)
- **deployment-agent.md** - Push dataset + notebook to HuggingFace (Stage 5)
- **monitoring-agent.md** - Comparison visuals: base vs fine-tuned vs Apriori (Stage 7)
- **evaluation-agent.md** - Eval datasets & scripts (ready before training)

**🔧 Utility Agents (available anytime):**
- **cleanup-agent.md** - Repository hygiene
- **maintainer-agent.md** - Documentation maintenance

Skills are in `.github/agents/skills/`.

**Knowledge Base:** All agent memories, logs, experiments, and decisions are stored in the **Obsidian vault** at `obsidian-brain/`. Agents use `[[backlinks]]` to cross-reference related knowledge.

**Critical:** Every agent reads its Obsidian memory note BEFORE any task. See `obsidian-brain/Agents/` for agent-specific notes, past mistakes, and improvement history.

---
Provide feedback if model handling, synthetic log differentiation, or validation invariants need deeper clarification.
