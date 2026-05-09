# AI-Augmented Development

This project was developed using a structured multi-agent AI workflow. Nine specialized agents — defined in `.github/agents/` — executed specific tasks under a human-designed orchestration system with persistent memory in `obsidian-brain/`.

This page explains the philosophy, the division of responsibility, and the quality control mechanisms that make this approach defensible as a thesis methodology.

---

## The Core Principle

**AI handles execution velocity; the human handles design and verification.**

The division is precise:

| Responsibility | Owner | Examples |
|---------------|-------|----------|
| System architecture | Human | Apriori as oracle, 3-phase training, two-phase inference |
| Success criteria | Human | 13 validation invariants, 7 evaluation metrics, fixed eval set |
| Workflow design | Human | 8-stage pipeline, agent roles, memory-first rule |
| Decision authority | Human | Every ADR reviewed and approved before implementation |
| Code execution | AI agents | Dataset generation, pipeline runs, training data export |
| Notebook generation | AI agents | Training notebook with verified hyperparameters |
| Deployment mechanics | AI agents | HuggingFace Hub push, artifact naming |
| Documentation | AI agents | Under human-defined structure and content specs |

The key insight: **the Apriori oracle is an immutable, mathematical ground truth**. No matter what the agents produced, correctness was always measured against a deterministic algorithm — not against AI judgment. This is the architectural key that makes the entire system self-validating through mathematics.

---

## Why This Works for a Thesis

Traditional ML research relies on human annotation or subjective benchmarks. This project eliminates both:

1. **Ground truth is algorithmic.** Apriori produces the exact set of frequent itemsets for any dataset. There is no annotation disagreement, no inter-rater reliability concern, no labeling budget.

2. **Validation is automated.** The 13 invariants in `pipeline.py:64-141` check every LLM output against the CSV data — item existence, row membership, support count, canonical format. A model cannot pass validation by "looking reasonable."

3. **Evaluation is deterministic.** The 30 fixed evaluation datasets never change. The 7 metrics (Precision, Recall, F1, Exact Match, Count Accuracy, Hallucination Rate, JSON Parse Rate) are computed by comparing model output against Apriori output, set-for-set.

Because the oracle, validation, and evaluation are all deterministic, the AI agents operate within hard guardrails. They can generate datasets, run pipelines, and export training data — but they cannot influence what "correct" means.

---

## The 8-Stage Workflow

| Stage | Agent | Action | Artifact |
|-------|-------|--------|----------|
| 1. Organize | Orchestrator | Initialize workflow state | `workflow_state.json` |
| 2. Datasets | Dataset Agent | Generate 500 synthetic CSVs | `data/datasets_v2/` |
| 3. Pipeline | Pipeline Agent | Run Apriori + LLM extraction | `runs.db`, JSON artifacts |
| 4. Export | Training Agent | Export SFT/DPO data, generate notebook | `sft_cot_v3.json`, `dpo_real_v2.json` |
| 5. Deploy | Deployment Agent | Push dataset + notebook to HuggingFace | HF repo |
| **Pause** | **Human** | **Train on school GPU server (H200)** | **LoRA adapter weights** |
| 6. Validate | Training Agent | Validate evaluation results | Improvement notes |
| 7. Visualize | Monitoring Agent | Generate comparison charts | `visuals/` |
| 8. Finalize | Orchestrator | Close workflow | Final state |

After Stage 5, the workflow pauses. The human downloads the training notebook and HuggingFace dataset to the school's TLJH Jupyter server (NVIDIA H200, 150 GB VRAM), runs training, and evaluates. The AI agents do not have access to the GPU server — training is always a human-executed step.

---

## Quality Control Mechanisms

### 1. Deterministic Oracle

Every pipeline run compares LLM output against Apriori output on the same CSV. The comparison is exact — no fuzzy matching, no thresholds. An itemset is either correct or it is not.

### 2. 13 Validation Invariants

`pipeline.py` enforces 13 invariant checks on every LLM output:

- JSON parseability and array structure
- Non-empty itemset items
- Canonical format (lowercase, sorted, trimmed)
- `Row N` evidence format with 1-based indexing
- Item existence in the actual CSV row (catches hallucinations)
- Support count verification against evidence rows

These invariants cannot be bypassed. A run with any invariant failure is flagged `validation_passed=0` in `runs.db`.

### 3. Persistent Memory

Every agent reads its memory file from `obsidian-brain/Agents/` before executing any task (the "Memory-First Rule"). This ensures:

- Decisions from previous sessions are not forgotten
- Known failure modes (e.g., `merged_4bit_forced` destroys LoRA quality) are never repeated
- Experiment results inform the next iteration

### 4. LLM Council

Major decisions (v3 training strategy, hyperparameter selection) were reviewed by a multi-model "council" — 3-4 different LLMs independently analyzing the same evidence and providing recommendations. The human reviewed all council outputs and made the final call. Council reports are archived in `docs/reports/`.

### 5. SQLite Audit Trail

`runs.db` records every pipeline run with 27 columns of metadata — timestamps, model IDs, itemset counts, validation results, error messages, and artifact paths. This is the single source of truth for the entire project.

---

## Comparison to Standard Practice

| Aspect | Traditional Development | This Project |
|--------|------------------------|-------------|
| Code generation | Manual or Copilot inline | Specialized agents with domain context |
| Decision tracking | Git commit messages | 26 ADRs + obsidian-brain vault |
| Experiment logging | Spreadsheets / W&B | Structured markdown + SQLite |
| Quality assurance | Formal unit test suite + manual review | Empirical validation via 13 pipeline invariants, deterministic Apriori oracle, evaluation scripts, and manual review |
| Reproducibility | "Run this notebook" | Fixed eval set + hash-based artifacts + versioned data |

Using AI agents for execution is analogous to using GitHub Copilot at scale — it accelerates routine coding tasks while the developer retains architectural and design authority. The difference is structure: each agent has a constrained scope, persistent memory, and defined skill modules rather than ad-hoc inline suggestions. The repository does not currently include a formal pytest-style unit/integration test suite; reproducibility is supported through deterministic validators, Apriori-based evaluation, archived artifacts, and manual review.

---

## Source Evidence

- `.github/agents/` — 9 agent definitions
- `.github/agents/skills/` — 11 modular skill definitions
- `.github/WORKFLOW_GUIDE.md` — 8-stage workflow guide
- `obsidian-brain/` — persistent memory vault
- `AGENTS.md` — multi-agent system documentation
- [ADR-026: Agent Workflow](../decisions/adr-026-agent-workflow.md)
