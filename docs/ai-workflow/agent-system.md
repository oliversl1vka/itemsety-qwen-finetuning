# Agent System

The project uses 9 specialized AI agents coordinated through a defined workflow. Each agent has a single responsibility, a persistent memory file, and access to modular skill definitions.

---

## Agent Roster

### Main Workflow Agents

| Agent | Stage | Responsibilities | Memory File |
|-------|-------|-----------------|-------------|
| Orchestrator | All | Coordinates workflow, resolves inter-agent conflicts, manages workflow state | `obsidian-brain/Agents/Orchestrator.md` |
| Dataset Agent | 2 | Generates 500 synthetic CSV datasets with controlled distributions (4-26 rows, 3-12 columns) | `obsidian-brain/Agents/Dataset Agent.md` |
| Pipeline Agent | 3 | Runs `pipeline.py` across datasets — Apriori extraction, LLM extraction, validation, SQLite persistence | `obsidian-brain/Agents/Pipeline Agent.md` |
| Training Agent | 4, 6 | Exports SFT/DPO training data, generates training notebooks, validates post-training evaluation results | `obsidian-brain/Agents/Training Agent.md` |
| Deployment Agent | 5 | Pushes datasets and notebooks to HuggingFace Hub, manages model card metadata | `obsidian-brain/Agents/Deployment Agent.md` |
| Evaluation Agent | Post-training | Runs `eval_finetuned_model.py`, collects 7 metrics, compares models | `obsidian-brain/Agents/Evaluation Agent.md` |
| Monitoring Agent | 7 | Generates comparison visualizations (base vs fine-tuned vs Apriori), creates summary reports | `obsidian-brain/Agents/Monitoring Agent.md` |

### Utility Agents

| Agent | Scope | Responsibilities | Memory File |
|-------|-------|-----------------|-------------|
| Cleanup Agent | Ongoing | Removes stale artifacts, manages `.gitignore`, enforces repository hygiene | `obsidian-brain/Agents/Cleanup Agent.md` |
| Maintainer Agent | Ongoing | Repository maintenance, documentation updates, dependency management | `obsidian-brain/Agents/Maintainer Agent.md` |

---

## The Memory-First Rule

Every agent reads its memory file from `obsidian-brain/Agents/` before executing any command. This is the single most important rule in the workflow.

**Why it matters:**

- The Orchestrator's memory contains workflow validation patterns discovered through "diamond knowledge" extraction (reviewing 11 external Unsloth notebooks to identify mandatory verification gates)
- The Training Agent's memory contains the critical finding that `merged_4bit_forced` destroys LoRA adapter quality — discovered in v2 and confirmed by an LLM Council diagnosis
- The Pipeline Agent's memory contains known edge cases in validation invariant checking

Without the memory-first rule, agents would repeat mistakes from previous sessions. With it, knowledge accumulates across the entire development timeline.

---

## Skill Modules

Agents access 11 modular skill definitions in `.github/agents/skills/`. Each skill encapsulates domain-specific knowledge for a single operation:

| Skill | Purpose | Used By |
|-------|---------|---------|
| `csv-dataset-generation` | Synthetic CSV generation with controlled distributions | Dataset Agent |
| `apriori-extraction` | Running the Apriori algorithm for ground-truth itemsets | Pipeline Agent |
| `llm-itemset-extraction` | Calling LLM APIs (GPT-4.1-mini, o4-mini, etc.) for extraction | Pipeline Agent |
| `validation-pipeline` | 13 invariant checks on LLM output | Pipeline Agent |
| `sqlite-persistence` | Writing run metadata to `runs.db` | Pipeline Agent |
| `training-data-export` | Exporting SFT-CoT and DPO training data from `runs.db` | Training Agent |
| `qwen-finetuning` | LoRA/QLoRA configuration, training notebook generation | Training Agent |
| `model-evaluation` | Running `eval_finetuned_model.py`, collecting 7 metrics | Evaluation Agent |
| `huggingface-deployment` | HuggingFace Hub push, model card, dataset card | Deployment Agent |
| `metrics-visualization` | Chart generation for model comparison | Monitoring Agent |
| `agent-logging` | Structured logging to `obsidian-brain/Logs/` | All agents |

Skills are defined as markdown files (`SKILL.md`) containing the exact commands, expected inputs/outputs, and error handling procedures. This makes each operation reproducible and auditable.

---

## Workflow State

The workflow state is tracked in `.github/agents_memory/workflow_state.json`. This JSON file records:

- Which stages have been completed
- Artifact counts produced at each stage
- Timestamps for stage transitions
- The current active stage

When the Orchestrator initializes a workflow run, it creates this state file. Each agent updates it upon completing its stage. This prevents agents from executing out of order or re-running completed stages.

---

## Persistent Memory Vault

The `obsidian-brain/` directory serves as the project's persistent memory vault:

```
obsidian-brain/
  Agents/           # 9 agent memory files
  Decisions/        # Architectural decision records (original format)
  Experiments/      # 4 timestamped experiment logs
  Logs/             # Activity logs per agent
  References/       # External reference material
  README.md         # Vault overview
```

This vault is an Obsidian-compatible markdown collection. It uses `[[wikilinks]]` for cross-referencing between notes. The content was migrated to the formal ADR format in `docs/decisions/` for public documentation, but the vault remains in the repository as the original source of record.

---

## Source Evidence

- `.github/agents/` — Agent definitions (9 markdown files)
- `.github/agents/skills/` — Skill modules (11 directories with `SKILL.md`)
- `.github/WORKFLOW_GUIDE.md` — Step-by-step execution guide
- `AGENTS.md` — System overview and quick start
- `obsidian-brain/` — Persistent memory vault (35+ markdown files)
