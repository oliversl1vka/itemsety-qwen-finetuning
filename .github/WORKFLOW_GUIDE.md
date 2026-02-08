# Multi-Agent Workflow Guide

**Last Updated:** 2026-02-08  
**Version:** 3.0 (Interactive Multi-Agent + Jupyter Training)

This guide explains how to execute the full training workflow by switching between specialized agents in VS Code.

---

## 🎯 Workflow Overview

The full training workflow consists of **8 sequential stages**, each handled by a specialized agent.  
After Stage 4, the workflow **PAUSES** for the user to train and evaluate on their own Jupyter server.

```
1. Orchestrator       → /organize   → Initialize workflow
2. Dataset Agent      → /datasets   → Generate training + evaluation datasets (versioned)
3. Pipeline Agent     → /pipeline   → Run Apriori + LLM extraction
4. Training Agent     → /export     → Export training data + generate versioned .ipynb notebook
5. Deployment Agent   → /push       → Push training dataset + notebook to HuggingFace
   ── PAUSE ── User trains & evaluates on Jupyter server ──
6. Training Agent     → /validate   → Receive eval results, validate, write improvement notes
7. Monitoring Agent   → /visualize  → Create comparison visuals (base vs fine-tuned vs Apriori)
8. Orchestrator       → /finalize   → Finalize workflow

🔧 UTILITY AGENTS (available anytime, not mandatory):
   Cleanup Agent      → /cleanup    → Repository hygiene
   Maintainer Agent   → /maintain   → Documentation updates
```

**End Goal:** Versioned training notebook + dataset on HuggingFace → User trains → Results validated → Comparison visuals → Improvement notes saved for next iteration.

---

## 🧠 Memory-First Rule

**EVERY agent MUST read its memory file before executing ANY command.**

This ensures we never repeat past mistakes and continuously improve:

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

Memory notes live in the **Obsidian vault** at `obsidian-brain/`.
Activity logs go to `obsidian-brain/Logs/`.
Experiment reports go to `obsidian-brain/Experiments/`.
Decisions go to `obsidian-brain/Decisions/`.
Reference docs live in `obsidian-brain/References/`.

---

## 🚀 Step-by-Step Execution

### Stage 1: Initialize Workflow
```
@workspace /agents switch to orchestrator
/organize
```

**What happens:**
- Reads orchestrator memory from Obsidian vault
- Creates workflow state file
- Shows 8-stage plan + utility agents
- Tells you which agent to activate next

---

### Stage 2: Generate Datasets (Training + Evaluation)
```
@workspace /agents switch to dataset-agent
/datasets
```

**What happens:**
- Reads dataset agent memory from Obsidian vault
- Generates 500 training CSV files in `data/datasets_v2/`
- Generates evaluation datasets in `data/eval_datasets_v1/` (**versioned, FIXED across model versions**)
- Eval datasets enable fair comparison across all future model iterations
- Logs metadata, updates workflow state

**Duration:** ~5-10 minutes  
**Artifacts:** 500 training CSVs + 9 eval CSVs (versioned)

---

### Stage 3: Run Pipeline (Apriori + LLM)
```
@workspace /agents switch to pipeline-agent
/pipeline
```

**What happens:**
- Reads pipeline agent memory from Obsidian vault for API rate limit patterns, chunk sizes
- Runs Apriori + LLM extraction on all training datasets
- Validates outputs (13 invariants), persists to `runs.db`

**Duration:** ~2-4 hours  
**Artifacts:** `artifacts/`, `runs.db`

---

### Stage 4: Export Training Data + Generate Notebook
```
@workspace /agents switch to training-agent
/export
```

**What happens:**
- Reads training agent memory from Obsidian vault for past training insights and improvement notes
- Exports validated runs to training format
- Creates HuggingFace dataset
- **Generates a versioned `.ipynb` training notebook** (e.g., `notebooks/training_dpo_v3.ipynb`)
- The notebook + HF dataset are the **ONLY 2 things needed** for user training

**Duration:** ~2-5 minutes  
**Artifacts:** Training data + HF dataset + versioned `.ipynb` notebook

---

### Stage 5: Push to HuggingFace
```
@workspace /agents switch to deployment-agent
/push
```

**What happens:**
- Pushes ONLY the training dataset + versioned notebook + eval kit to HuggingFace Hub
- Updates workflow state

**Duration:** ~5-15 minutes  

---

### ⏸️ WORKFLOW PAUSES HERE

**The user now trains and evaluates on their own Jupyter server:**

1. Download the `.ipynb` notebook and HF dataset on your Jupyter server
2. Run the training notebook (SFT: ~40-60min, DPO: ~60-90min)
3. Run the evaluation cells (uses the fixed eval datasets)
4. Record the metrics: F1, Precision, Recall, JSON parse rate, hallucination rate
5. Come back to VS Code when done

---

### Stage 6: Validate Results & Write Improvement Notes
```
@workspace /agents switch to training-agent
/validate
```

**What happens:**
- Reads ALL training agent memory from Obsidian vault (past iterations, what worked, what didn't)
- Asks user for evaluation results (F1, parse rate, hallucinations, etc.)
- Analyzes results against targets and previous iterations
- **Writes detailed improvement notes to `obsidian-brain/Agents/Training Agent.md`**
- Creates experiment report in `obsidian-brain/Experiments/`
- Suggests concrete changes for next training notebook version if needed

**Duration:** ~5 minutes

---

### Stage 7: Create Comparison Visuals
```
@workspace /agents switch to monitoring-agent
/visualize
```

**What happens:**
- Creates comparison charts: Base model vs Fine-tuned model vs Apriori
- F1, Precision, Recall, JSON parse rate, hallucination rate breakdowns
- Model version progression over time (if multiple iterations)
- Saves visuals to `visuals/`

**Duration:** ~1-2 minutes

---

### Stage 8: Finalize Workflow
```
@workspace /agents switch to orchestrator
/finalize
```

**What happens:**
- Verifies all 8 stages completed
- Generates final workflow summary with metrics
- Lists improvement notes saved
- Marks workflow as completed

---

## 🔧 Utility Agents (Available Anytime)

### Cleanup Agent
```
@workspace /agents switch to cleanup-agent
/cleanup
```
Repository hygiene: verify SQLite records, clean old artifacts, check for orphaned files.

### Maintainer Agent
```
@workspace /agents switch to maintainer-agent
/maintain
```
Documentation updates: review outputs, update docs, push to git.

---

## 📋 Agent Reference

| Agent | Commands | Stage | Type |
|-------|----------|-------|------|
| Orchestrator | `/organize`, `/finalize`, `/status` | 1, 8 | Main |
| Dataset Agent | `/datasets`, `/analyze` | 2 | Main |
| Pipeline Agent | `/pipeline`, `/validate-run` | 3 | Main |
| Training Agent | `/export`, `/validate` | 4, 6 | Main |
| Deployment Agent | `/push`, `/deploy` | 5 | Main |
| Monitoring Agent | `/visualize`, `/report` | 7 | Main |
| Evaluation Agent | `/eval`, `/compare` | Support | Support |
| Cleanup Agent | `/cleanup` | — | 🔧 Utility |
| Maintainer Agent | `/maintain` | — | 🔧 Utility |

---

## 🔄 Iterative Improvement Cycle

The workflow is designed for **iterative improvement**:

1. **First iteration:** Full 8-stage workflow → baseline metrics
2. **Subsequent iterations:** Start from Stage 4 (`/export` with improvements from memory notes)
   - Training agent reads past improvement notes from memory
   - Generates new versioned notebook with adjustments (e.g., `training_dpo_v4.ipynb`)
   - Same fixed eval datasets ensure fair comparison
   - New notes appended to memory after validation
3. **Over time:** Memory accumulates insights, each iteration improves on the last

---

## ⚠️ Important Notes

### Evaluation Datasets Are Sacred
- Generated ONCE in Stage 2, versioned (e.g., `data/eval_datasets_v1/`)
- NEVER modified between model versions
- Enable fair apples-to-apples comparison across all iterations

### Versioning
- **Datasets:** `data/datasets_v2/`, `data/eval_datasets_v1/`
- **Notebooks:** `notebooks/training_{method}_v{N}.ipynb`
- **HF Datasets:** `OliverSlivka/itemset-extraction-v{N}`
- **Memory notes:** Append-only, timestamped entries in `obsidian-brain/Agents/`

### Resume from Failure
If a stage fails, fix the issue and re-run that agent. The workflow state tracks progress.

---

## 🐛 Troubleshooting

### "Workflow state not found"
**Solution:** Run `/organize` first to initialize workflow

### "Eval datasets missing"
**Solution:** Run `/datasets` — it generates both training AND eval datasets

### "Notebook version conflict"
**Solution:** Check `notebooks/notebook_versions.json` for latest version number

### "Credentials missing"
**Solution:** Ensure `openai.env` exists with valid `OPENAI_API_KEY`, and `HF_TOKEN` is set

---

**Questions?** Ask the orchestrator: `@workspace /agents switch to orchestrator` then `/help`
