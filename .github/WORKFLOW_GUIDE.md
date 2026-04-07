# Multi-Agent Workflow Guide

**Last Updated:** 2026-03-01
**Version:** 5.0 (3-Phase Training: SFT-CoT → DPO-Real → GRPO)

This guide explains how to execute the full training workflow by switching between specialized agents in VS Code.

---

## 🎯 Workflow Overview

The full training workflow consists of **8 sequential stages**, each handled by a specialized agent.
After Stage 5 the workflow **PAUSES** for the user to fine-tune and evaluate on school GPUs.

```
1. Orchestrator        → /organize    → Initialize workflow
2. Dataset Agent       → /datasets    → ⚙️  OPTIONAL — only if new training/eval datasets needed
3. Pipeline Agent      → /pipeline    → Run Apriori + LLM extraction (repeat per model/day)
4. Training Agent      → /export      → Export 3-phase training data + generate notebook
5. Deployment Agent    → /push        → Push dataset + notebook to HuggingFace repository
   ── PAUSE ── User fine-tunes on school GPUs, then runs evaluation notebook ──
6. Training Agent      → /validate    → Receive results (+ errors), validate, improve training script
7. Monitoring Agent    → /visualize   → Comparison visuals: base vs fine-tuned vs Apriori
8. Orchestrator        → /finalize    → Finalize workflow

🔧 UTILITY AGENTS (available anytime, not mandatory):
   Cleanup Agent       → /cleanup     → Repository hygiene
   Maintainer Agent    → /maintain    → Documentation + git push
```

**End Goal:** 3-phase training data (SFT-CoT + DPO-Real + GRPO) + training notebook on HuggingFace → User trains Qwen2.5-7B on school GPUs → Results validated by Training Agent + Evaluation Agent → Comparison visuals → Improvement notes saved.

---

## 🧠 Memory-First Rule

**EVERY agent MUST read its memory file before executing ANY command.**

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

Memory notes live in `obsidian-brain/`.
Activity logs → `obsidian-brain/Logs/`.
Experiment reports → `obsidian-brain/Experiments/`.
Decisions → `obsidian-brain/Decisions/`.

---

## 🚀 Step-by-Step Execution

### Stage 1: Initialize Workflow
```
@workspace /agents switch to orchestrator
/organize
```

**What happens:**
- Creates `.github/agents_memory/workflow_state.json`
- Shows the 8-stage plan + multi-model pipeline strategy
- Tells you which agent to activate next

---

### Stage 2: Generate Datasets ⚙️  OPTIONAL
```
@workspace /agents switch to dataset-agent
/datasets
```

> **When to run:** Only when you need NEW training or evaluation datasets.
> If you already have `data/datasets_v2/` (500 CSVs) and `data/eval_datasets_v1/` (9 eval CSVs), **skip this stage** and go straight to Pipeline Agent.

**What happens:**
- Generates 500 training CSV files in `data/datasets_v2/`
- Generates versioned evaluation datasets in `data/eval_datasets_v1/` (FIXED — never changed between models)
- Eval datasets enable fair apples-to-apples comparison across all fine-tuning iterations

**Duration:** ~5–10 minutes
**Artifacts:** 500 training CSVs + 9 eval CSVs (versioned)

---

### Stage 3: Run Pipeline (Apriori + LLM) — repeat per model per day
```
@workspace /agents switch to pipeline-agent
/pipeline
```

> **This is the most-used stage.** Run it multiple times across multiple days with different LLM models to accumulate a large, diverse training set.

**Multi-model accumulation strategy:**
- You have 500 datasets and plan to run ~5 different models
- Goal: ~2500 diverse training examples (500 datasets × 5 models)
- All runs go into `runs.db` (keyed by dataset_hash + llm_model — no duplicates)
- Each day, run a different model (respecting your daily API quota)

**Example multi-day schedule:**
| Day | Model | Datasets | New examples |
|-----|-------|----------|-------------|
| 1 | `gpt-4.1-mini` | 500 | ~450 validated |
| 2 | `gemini-2.0-flash` | 500 | ~450 validated |
| 3 | `claude-3-5-haiku` | 500 | ~450 validated |
| 4 | `gpt-4.1` | 500 | ~450 validated |
| 5 | `llama-3.3-70b` | 500 | ~450 validated |
| **Total** | | 2500 | **~2250 validated** |

**Per-model run command:**
```bash
python pipeline.py \
  --data-dir data/datasets_v2 \
  --min-support 3 --max-size 3 \
  --llm-full --llm-chunk-size 50 \
  --llm-model <model_name>

# Check accumulation progress
sqlite3 runs.db "SELECT llm_model, COUNT(*) total, SUM(validation_passed) valid FROM runs GROUP BY llm_model"
```

**What the agent does:**
- Reads pipeline memory for rate limit patterns and optimal chunk sizes
- Checks `runs.db` to confirm which models have already run (avoids duplicate work)
- Runs Apriori + LLM extraction on all datasets for the chosen model
- Validates all outputs (13 invariants), persists to `runs.db`

**Duration:** ~2–4 hours per model run
**Artifacts:** `artifacts/` (accumulates per model), `runs.db` (accumulates all model runs)

---

### Stage 4: Export 3-Phase Training Data + Generate Notebook
```
@workspace /agents switch to training-agent
/export
```

**What happens:**
- Reads training agent memory for past training insights
- Phase 1: Generates SFT-CoT examples with `<think>` reasoning from validated runs
- Phase 2: Exports DPO preference pairs (Apriori+CoT as chosen, real LLM failures as rejected)
- Phase 3: Creates GRPO examples with Apriori ground truth for reward functions
- Builds HuggingFace dataset with 3 configs (`data/hf_dataset_v2/`)
- Training notebook: `notebooks/training_3phase_7b.ipynb` (SFT-CoT → DPO-Real → GRPO)

**Duration:** ~2–5 minutes
**Artifacts:** `data/hf_dataset_v2/` (3 configs: sft/dpo/grpo) + `training_3phase_7b.ipynb`

---

### Stage 5: Push to HuggingFace (Repository — storage only)
```
@workspace /agents switch to deployment-agent
/push
```

> **HuggingFace destination:** A plain Hub **repository** (e.g., `OliverSlivka/itemset-finetuning-workspace`) — purely file storage, **no training runs on HuggingFace**.

**What gets pushed:**
1. 3-phase training dataset (`data/hf_dataset_v2/` — sft/dpo/grpo configs)
2. Training notebook (`notebooks/training_3phase_7b.ipynb`)
3. Evaluation assets (`src/evaluation/eval_finetuned_model.py` + eval datasets)

**Duration:** ~5–15 minutes

---

### ⏸️  WORKFLOW PAUSES HERE

**You now train and evaluate on your school GPUs:**

1. Download the training notebook + dataset from HuggingFace
2. Run the **training notebook** (3 phases: SFT-CoT ~20 min + DPO ~20 min + GRPO ~15 min = ~1–2 hours, Unsloth)
3. The notebook automatically runs evaluation at the end
4. Record all output metrics:
   - F1 Score, Precision, Recall
   - JSON Parse Rate
   - Hallucination Rate
   - Inference Time (per dataset)
   - Any error messages / training issues
5. Come back to VS Code when done

---

### Stage 6: Validate Results + Write Improvement Notes
```
@workspace /agents switch to training-agent
/validate
```

**What happens (training agent + evaluation agent working together):**

1. **Training Agent reads ALL memory** — previous iterations, what worked, what failed
2. **You provide your results** — paste `evaluation_summary.json`, metric values, or error messages
3. **Evaluation Agent invoked internally:**
   - Runs deep failure analysis on model outputs
   - Optionally invokes LLM Council (`council_advisor.py`) for multi-LLM expert opinions on results and training script improvements
4. **Training Agent analyzes results:**
   - Compares against targets (F1 ≥ 0.80, parse rate ≥ 0.90, hallucination ≤ 5%)
   - Compares against previous iterations from memory
5. **If targets met or improvement shown:** Writes success notes to memory
6. **If below target OR errors occurred:**
   - Edits training notebook with concrete fixes (hyperparameters, data quality, LoRA config)
   - Or gives specific manual suggestions if the fix requires user judgment
7. **Improvement notes + experiment report saved** to Obsidian vault
8. Workflow state updated

**Also available standalone (Evaluation Agent for deeper analysis):**
```
@workspace /agents switch to evaluation-agent
/eval      → Detailed metric analysis + failure pattern report
/council   → Run LLM Council review of results + training script improvements
/compare   → Compare two model versions side by side
```

---

### Stage 7: Create Comparison Visuals
```
@workspace /agents switch to monitoring-agent
/visualize
```

**What happens:**
- Creates comparison charts: Base Qwen (no fine-tuning) vs Fine-tuned model vs Apriori (ground truth)
- F1, Precision, Recall, JSON parse rate, hallucination rate, inference time per dataset
- Model version progression over time (if multiple training iterations in memory)
- Saves all charts to `visuals/`

**Duration:** ~1–2 minutes

---

### Stage 8: Finalize Workflow
```
@workspace /agents switch to orchestrator
/finalize
```

**What happens:**
- Verifies all 8 stages completed
- Generates final workflow summary with metrics
- Lists all improvement notes saved
- Marks workflow as completed in state

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
Documentation updates + git push: review outputs, update `AGENTS.md` / `README.md` / agent files, push to git.

---

## 📋 Agent Reference

| Agent | Commands | Stage | Type |
|-------|----------|-------|------|
| Orchestrator | `/organize`, `/finalize`, `/status` | 1, 8 | Main |
| Dataset Agent | `/datasets`, `/analyze` | 2 ⚙️ optional | Main |
| Pipeline Agent | `/pipeline`, `/validate-run`, `/status` | 3 (repeat) | Main |
| Training Agent | `/export`, `/validate` | 4, 6 | Main |
| Deployment Agent | `/push`, `/status` | 5 | Main |
| Monitoring Agent | `/visualize`, `/report` | 7 | Main |
| Evaluation Agent | `/eval`, `/council`, `/compare` | Support (called in Stage 6) | Support |
| Cleanup Agent | `/cleanup` | — | 🔧 Utility |
| Maintainer Agent | `/maintain` | — | 🔧 Utility |

---

## 🔄 Multi-Model Pipeline Strategy

Because you have limited daily API calls, the pipeline runs **incrementally** across multiple days:

```
Day 1:  pipeline-agent /pipeline  (model: gpt-4.1-mini)    → ~450 runs
Day 2:  pipeline-agent /pipeline  (model: gemini-flash)     → ~450 runs
Day 3:  pipeline-agent /pipeline  (model: claude-haiku)     → ~450 runs
Day N:  pipeline-agent /pipeline  (model: <next-model>)     → ~450 runs
                                                               ──────────
                                                               ~2250 total
                  ↓
        training-agent /export   (exports ALL accumulated runs as 3-phase training data)
```

- `runs.db` accumulates ALL model runs — no overwrites (keyed by dataset_hash + llm_model)
- Each artifact is named with the model prefix, so they never collide
- The training export script picks up everything validated in `runs.db`

**Check accumulation progress:**
```bash
sqlite3 runs.db "SELECT llm_model, COUNT(*) as total, SUM(validation_passed) as valid FROM runs GROUP BY llm_model ORDER BY total DESC"
```

---

## 🔄 Iterative Improvement Cycle

The workflow supports **iterative model improvement** across training versions:

1. **Iteration 1:** Full 8-stage workflow → baseline metrics
2. **Iteration 2+:** Start from Stage 4 (`/export` with memory-informed improvements)
   - Training agent reads all past improvement notes from memory
   - Updates training notebook hyperparameters based on previous results
   - Same fixed eval datasets ensure fair comparison across all versions
3. **Over time:** Memory accumulates insights, each iteration improves on the last

---

## ⚠️ Important Notes

### Evaluation Datasets Are Fixed
- Generated ONCE in Stage 2, versioned (`data/eval_datasets_v1/`)
- **NEVER modified** between model versions
- Enable fair apples-to-apples comparison across all training iterations

### Training Notebook
- Main notebook: `notebooks/training_3phase_7b.ipynb` (21 cells, SFT-CoT → DPO-Real → GRPO)
- Hyperparameters can be updated between iterations based on evaluation results

### HuggingFace is Storage Only
- Each version has its own HF Hub repo: `OliverSlivka/itemset-extraction-v{N}` (currently v3)
- **No training runs on HuggingFace** — all training happens on your school GPUs
- Repo holds: 3-phase dataset (sft/dpo/grpo) + training notebook + eval assets
- ⚠️ **NEVER overwrite previous version repos** — each new version creates a new repo

### Resume from Failure
If a stage fails, fix the issue and re-run that agent. Workflow state tracks progress.

---

## 🐛 Troubleshooting

### "Workflow state not found"
**Solution:** Run `/organize` first

### "Eval datasets missing"
**Solution:** Run dataset-agent `/datasets` — generates training AND eval datasets

### "Pipeline already ran for this model"
**Solution:** Check `runs.db` with the progress SQL above. If the model is complete, pick a new model.

### "Training crashed on school GPU"
**Solution:** Come back to VS Code → training-agent `/validate`, paste the error. The agent will diagnose and edit the training notebook.

### "Model output quality is low (F1 < 0.5)"
**Solution:** Run `/validate` with results → training agent + evaluation agent will analyze failures and suggest fixes. Use evaluation-agent `/council` for multi-LLM expert advice.

### "Credentials missing"
**Solution:** `openai.env` → OPENAI_API_KEY; `openrouter.env` → OPENROUTER_API_KEY; `hf.env` → HF_TOKEN

---

**Questions?** Ask the orchestrator: `@workspace /agents switch to orchestrator` then `/help`
