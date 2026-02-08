# AGENTS Guidelines for itemsety-qwen-finetuning

**Version:** 2.0 (Interactive Multi-Agent System)  
**Last Updated:** 2026-02-03

This repository uses an **interactive multi-agent orchestration system** where you switch between specialized agents in VS Code to execute the workflow.

---

## 🎯 Project Overview

**Goal:** Fine-tune Qwen models to extract frequent itemsets from CSV datasets without using Apriori algorithm.

**Approach:**
1. Generate synthetic CSV datasets (500 datasets, 5-25 rows each)
2. Run Apriori + OpenAI GPT-4 to create ground truth
3. Export validated runs as training data (ChatML format with CoT)
4. Fine-tune Qwen2.5-3B using LoRA/QLoRA
5. Evaluate on unseen datasets (target: 80% F1 vs Apriori)
6. Deploy to HuggingFace Hub

---

## 🤖 How It Works

### Multi-Agent Execution Model

**You switch between 7 specialized agents** using VS Code Copilot Chat:

```
User → Orchestrator → Dataset Agent → Pipeline Agent → Training Agent → Deployment Agent → ⏸️ USER TRAINS → Training Agent → Monitoring Agent → Orchestrator
  ↓        ↓             ↓               ↓                ↓                  ↓                                     ↓                ↓              ↓
/help   /organize    /datasets      /pipeline        /export           /push          (Jupyter server)         /validate      /visualize    /finalize
```

Each agent:
1. **Reads its Obsidian memory note FIRST** from `obsidian-brain/Agents/` (never repeats past mistakes)
2. **Reads** workflow state (`.github/agents_memory/workflow_state.json`)
3. **Executes** its stage (runs Python scripts, validates outputs)
4. **Updates** workflow state (marks stage complete, adds artifact counts)
5. **Tells you** which agent to activate next

**No automation** - you manually switch agents and run commands. This gives you full control and visibility.

**After Stage 5 (/push)**, the workflow PAUSES — the user trains & evaluates on their own Jupyter server. The `.ipynb` notebook + HF dataset are the only 2 things needed.

---

## 📖 Quick Start

See [.github/WORKFLOW_GUIDE.md](.github/WORKFLOW_GUIDE.md) for detailed step-by-step instructions.

**TL;DR:**
```bash
# Stage 1: Initialize workflow
@workspace /agents switch to orchestrator
/organize

# Stage 2: Generate training + evaluation datasets
@workspace /agents switch to dataset-agent
/datasets

# Stage 3: Run pipeline
@workspace /agents switch to pipeline-agent
/pipeline

# Stage 4: Export training data + generate versioned notebook
@workspace /agents switch to training-agent
/export

# Stage 5: Push dataset + notebook to HuggingFace
@workspace /agents switch to deployment-agent
/push

# ⏸️ PAUSE: User trains & evaluates on Jupyter server
# (download notebook + dataset, run training, run eval, record metrics)

# Stage 6: Validate eval results & write improvement notes
@workspace /agents switch to training-agent
/validate

# Stage 7: Create comparison visuals (base vs fine-tuned vs Apriori)
@workspace /agents switch to monitoring-agent
/visualize

# Stage 8: Finalize workflow
@workspace /agents switch to orchestrator
/finalize

# 🔧 UTILITY (anytime): /cleanup, /maintain
```

---

## 🤖 Agent Architecture

This project uses **7 main workflow agents** + **2 utility agents** coordinated by the **Orchestrator**:

### Main Workflow Agents

### 1. **Orchestrator Agent** [`.github/agents/orchestrator.md`](.github/agents/orchestrator.md)
**Role:** Master workflow coordinator (Stages 1 & 8)  
**Responsibilities:** Initialize workflow, track progress, finalize workflow  
**Commands:** `/organize`, `/status`, `/finalize`

### 2. **Dataset Agent** [`.github/agents/dataset-agent.md`](.github/agents/dataset-agent.md)
**Role:** Dataset generation (Stage 2)  
**Responsibilities:** Generate training CSVs + **fixed evaluation datasets** (versioned, never change between models)  
**Commands:** `/datasets`, `/analyze`

### 3. **Pipeline Agent** [`.github/agents/pipeline-agent.md`](.github/agents/pipeline-agent.md)
**Role:** Frequent itemset extraction (Stage 3)  
**Responsibilities:** Run Apriori + LLM extraction, validate outputs, persist to SQLite  
**Commands:** `/pipeline`, `/validate-run`

### 4. **Training Agent** [`.github/agents/training-agent.md`](.github/agents/training-agent.md)
**Role:** Training data preparation + result validation (Stages 4 & 6)  
**Responsibilities:** Export training data, generate **versioned .ipynb notebooks**, receive eval results, write improvement notes to memory  
**Commands:** `/export`, `/validate`

### 5. **Deployment Agent** [`.github/agents/deployment-agent.md`](.github/agents/deployment-agent.md)
**Role:** HuggingFace dataset & notebook deployment (Stage 5)  
**Responsibilities:** Push ONLY training dataset + versioned notebook + eval kit to HuggingFace Hub  
**Commands:** `/push`, `/deploy`

### 6. **Monitoring Agent** [`.github/agents/monitoring-agent.md`](.github/agents/monitoring-agent.md)
**Role:** Comparison visuals & reporting (Stage 7)  
**Responsibilities:** Create comparison visuals (base model vs fine-tuned vs Apriori), compute metrics, generate reports  
**Commands:** `/visualize`, `/report`

### 7. **Evaluation Agent** [`.github/agents/evaluation-agent.md`](.github/agents/evaluation-agent.md)
**Role:** Model performance evaluation (Support agent)  
**Responsibilities:** Eval datasets generated upfront, eval script ready before training, compute P/R/F1 metrics  
**Commands:** `/eval`, `/compare`

---

## 🔧 Utility Agents (Available Anytime)

### 8. **Maintainer Agent** [`.github/agents/maintainer-agent.md`](.github/agents/maintainer-agent.md)
**Role:** Documentation accuracy & git synchronization  
**Responsibilities:** Review outputs, update documentation, push to git  
**Commands:** `/maintain`

### 9. **Cleanup Agent** [`.github/agents/cleanup-agent.md`](.github/agents/cleanup-agent.md)
**Role:** Repository hygiene  
**Responsibilities:** Verify SQLite records, clean artifacts, archive old files  
**Commands:** `/cleanup`

---

## �📋 Development Environment Tips

### Repository Structure
```
itemsety-qwen-finetuning/
├── pipeline.py                    # Core extraction pipeline
├── src/                           # Source code modules
│   ├── training/                  # Fine-tuning scripts
│   │   ├── run_sft_full.py        # Production training
│   │   ├── run_sft_test.py        # Test training
│   │   └── export_training_data.py
│   ├── evaluation/                # Model evaluation
│   │   └── eval_finetuned_model.py
│   ├── data_generation/           # Dataset generation
│   │   └── generate_datasets_v2.py
│   └── utils/                     # Utility scripts
│       └── visualization.py
│
├── data/                          # All data files
│   ├── datasets_v2/               # CSV datasets (500)
│   ├── training_v2/               # Training examples
│   └── hf_dataset_v2/             # HuggingFace format
│
├── docs/                          # Documentation
│   ├── guides/                    # How-to guides
│   └── reports/                   # Experiment reports
│
├── scripts/                       # Operational scripts
│   ├── deployment/                # HF deployment
│   ├── colab/                     # Colab scripts
│   └── db_maintenance/            # DB utilities
│
├── agents/                        # Agent definitions
├── obsidian-brain/                # 🧠 Obsidian knowledge vault
│   ├── Agents/                    # Agent memory notes (9 files)
│   ├── References/                # API limits, model comparison, etc.
│   ├── Logs/                      # Per-run activity logs
│   ├── Experiments/               # Training experiment reports
│   └── Decisions/                 # Architecture decisions
├── archive/                       # Archived files
│
├── artifacts/                     # Pipeline outputs (gitignored)
├── logs/                          # Execution logs (gitignored)
└── runs.db                        # SQLite database
```

### Key Technologies
- **Python 3.10+** (use `.venv` for environment isolation)
- **SQLite** (runs.db) for metadata persistence
- **OpenAI** (GPT-4o, GPT-4.1-mini) for baseline LLM extraction
- **HuggingFace** (Transformers, PEFT, TRL) for fine-tuning
- **PyTorch** with bitsandbytes (4-bit quantization)

### Essential Files
- **`openai.env`**: OpenAI API credentials (NEVER commit, use `openai.env.template`)
- **`runs.db`**: SQLite database with all run metadata
- **`requirements.txt`**: Python dependencies
- **`.gitignore`**: Excludes secrets, artifacts, models

---

## 🛠️ Common Commands

### Quick Start (Full Pipeline)
```powershell
# 1. Generate datasets
python src/data_generation/generate_datasets_v2.py --count 500

# 2. Run pipeline (Apriori + GPT-4)
python pipeline.py --data-dir data/datasets_v2 --llm-full --llm-model gpt_4_1 --min-support 3

# 3. Export training data
python src/training/export_training_data.py --validation-passed --min-itemsets 5

# 4. Create HF dataset
python src/training/create_hf_dataset.py --input data/training_v2 --output data/hf_dataset_v2

# 5. Upload to Hub
python src/training/upload_dataset_to_hf.py --dataset-path data/hf_dataset_v2

# 6. Train model
python src/training/run_sft_full.py

# 7. Evaluate
python src/evaluation/eval_finetuned_model.py --model-path OliverSlivka/qwen2.5-3b-itemset-extractor

# 8. Deploy
./scripts/deployment/deploy_to_hf_space.ps1

# 9. Monitor
python src/utils/visualization.py --db runs.db --outdir visuals
```

### Database Queries
```bash
# View latest runs
sqlite3 runs.db "SELECT dataset_id, validation_passed, llm_itemsets_count FROM runs ORDER BY timestamp DESC LIMIT 10"

# Count validated runs
sqlite3 runs.db "SELECT COUNT(*) FROM runs WHERE validation_passed = 1"

# Validation pass rate
sqlite3 runs.db "SELECT AVG(CAST(validation_passed AS FLOAT)) * 100 FROM runs"
```

### Artifact Inspection
```bash
# List Apriori outputs
ls artifacts/apriori_outputs/*.json

# View validation report
cat artifacts/validation_reports/gpt_4_1_validation_report_ds_0001_*.json | jq '.'

# Check log
cat logs/extractor/gpt_4_1_extractor_generation_log_ds_0001_*.json
```

---

## 🧪 Testing Instructions

### Unit Tests
```bash
# Run all tests
pytest tests/

# Test specific agent
pytest tests/test_pipeline.py
pytest tests/test_training_agent.py

# With coverage
pytest --cov=. --cov-report=html
```

### Integration Tests
```bash
# Test full pipeline on single dataset
python pipeline.py --data tests/fixtures/test_dataset_5x8.csv --llm-full

# Test training on small dataset
python run_sft_test.py  # Uses 50 examples, ~10 min

# Test evaluation
python eval_finetuned_model.py --model-path OliverSlivka/qwen2.5-3b-itemset-test --count 3
```

### Smoke Tests
```bash
# Check environment
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check database
sqlite3 runs.db "SELECT COUNT(*) FROM runs"

# Check OpenAI credentials
python -c "from dotenv import load_dotenv; import os; load_dotenv('openai.env'); print('OK' if os.getenv('OPENAI_API_KEY') else 'MISSING')"
```

---

## 📊 Code Style & Conventions

### Naming Conventions
- **Datasets:** `ds_{ID:04d}_{rows}x{cols}.csv` (e.g., `ds_0042_12x15.csv`)
- **Artifacts:** `{model}_{stage}_{stem}_{hash}.json` (e.g., `gpt_4_1_extractor_output_ds_0042_a1b2c3d4e5f6.json`)
- **Functions:** `verb_noun()` (e.g., `load_transactions_csv()`, `apriori_frequent_itemsets()`)

### Logging Format
```python
import logging
logger = logging.getLogger(__name__)

# Use structured logging
logger.info(f"[{dataset_id}] [stage] Message with {variable}")
logger.warning(f"[{dataset_id}] Retry attempt {n}/3")
logger.error(f"[{dataset_id}] Failed: {error_msg}")
```

### Error Handling
```python
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    return fallback_value  # Graceful degradation
except Exception as e:
    logger.critical(f"Unexpected error: {e}")
    raise  # Re-raise critical errors
```

### Configuration Management
- Use **environment variables** for secrets (OpenAI keys, HF tokens)
- Use **command-line arguments** for runtime params (min_support, batch_size)
- Use **YAML/JSON configs** for workflow definitions (future enhancement)

---

## 🚨 Important Boundaries

### ✅ Always Do
- Test locally before deploying
- Validate all outputs (13 invariants)
- Use hash-based artifact naming (reproducibility)
- Log all stages (apriori, extractor, validation, db_prepared)
- Persist metadata to SQLite
- Use 4-bit quantization for training (memory efficiency)
- Commit code changes (but never secrets!)

### ⚠️ Ask First
- Skip validation (risk data corruption)
- Modify Apriori logic (breaks ground truth)
- Use paid GPU without budget approval
- Deploy model without evaluation
- Delete datasets used in training

### 🚫 Never Do
- Commit `openai.env` or any file with API keys
- Modify historical DB records (data integrity)
- Run pipeline without OpenAI credentials (will fail with exit code 3)
- Delete `runs.db` (primary data source)
- Hardcode file paths (use args/config)
- Deploy untested models to production

---

## 🔧 Troubleshooting

### Issue: OpenAI API rate limit (429 errors)
**Solution:**
- Reduce batch size: `--llm-chunk-size 25`
- Add delays between calls
- Use sample mode (omit `--llm-full`)

### Issue: GPU OOM during training
**Solution:**
- Reduce batch size: `per_device_train_batch_size=1`
- Enable gradient checkpointing (already enabled)
- Use 8-bit optimizer (already enabled)
- Try smaller model (0.5B instead of 3B)

### Issue: Low F1 score (<50%)
**Solution:**
- Check training data quality: Are labels correct?
- Increase training epochs (3 → 5)
- Try larger model (3B → 7B)
- Add more training examples (500+)

### Issue: JSON parse failures (model outputs non-JSON)
**Solution:**
- Increase `max_new_tokens` (512 → 1024)
- Strip text before first `[` and after last `]`
- Add repetition penalty: `repetition_penalty=1.2`
- Retrain with better examples (strict JSON format)

---

## 📚 Documentation

### Agent Files (Primary Reference)
Each agent has a dedicated markdown file with:
- Persona & role definition
- Project knowledge & tech stack
- Commands & usage examples
- Code style & patterns
- Boundaries & limitations
- Testing instructions
- Troubleshooting guides

**Read these first** when working on specific tasks!

### Additional Documentation
- **README.md**: Project overview, setup, quick start
- **FINETUNING_README.md**: Training workflow details
- **EVALUATION_REPORT.md**: Latest model performance
- **TO-DO-LIST.md**: Task tracking and priorities
- **COMPLETION_SUMMARY.md**: Historical milestones

---

## 🎯 Performance Targets

### Pipeline (per dataset)
- CSV loading: < 0.5s
- Apriori mining: < 5s (25 rows, 20 cols)
- LLM extraction: < 60s (full mode)
- Validation: < 1s
- **Total:** < 90s per dataset

### Training
- Test mode: 10-15 min (50 examples)
- Production mode: 40-60 min (439 examples, 3 epochs)
- Memory: < 10 GB (3B model, 4-bit quantization)

### Model Quality
- **F1 Score:** ≥ 0.80 (vs Apriori ground truth)
- **Exact Match:** ≥ 0.50
- **JSON Parse Rate:** ≥ 0.90
- **Hallucination Rate:** ≤ 0.05
- **Inference Time:** ≤ 60s per dataset

---

## 🚀 Next Steps for New Contributors

1. **Read this file** (you're here! ✅)
2. **Read agent files** relevant to your task (e.g., `agents/training-agent.md` for training work)
3. **Set up environment:** `python -m venv .venv && .venv\Scripts\Activate.ps1 && pip install -r requirements.txt`
4. **Configure secrets:** Copy `openai.env.template` to `openai.env`, fill in your OpenAI API key
5. **Run tests:** `pytest tests/` to verify setup
6. **Try a small workflow:** `python pipeline.py --data tests/fixtures/test_dataset_5x8.csv --llm-full`
7. **Explore database:** `sqlite3 runs.db` and run some queries
8. **Generate visualizations:** `python visualization.py --db runs.db --outdir visuals`

---

## 💡 When Stuck

- **Check agent files** for detailed instructions (e.g., `agents/pipeline-agent.md` for pipeline issues)
- **Inspect logs:** `logs/apriori/`, `logs/extractor/`, etc.
- **Query database:** `sqlite3 runs.db` for debugging
- **Review artifacts:** `artifacts/validation_reports/` for error details
- **Ask for help:** Create an issue with: error message, commands run, logs

---

**Last Updated:** 2026-02-01  
**Maintained By:** Oliver Slivka  
**Project:** https://github.com/oliversl1vka/itemsety-qwen-finetuning  
**License:** Apache 2.0
