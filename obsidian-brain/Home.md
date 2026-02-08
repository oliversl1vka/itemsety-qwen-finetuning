# 🧠 Itemsety Project Brain

**Project:** Fine-tune Qwen models to extract frequent itemsets from CSV datasets  
**Repository:** [itemsety-qwen-finetuning](https://github.com/oliversl1vka/itemsety-qwen-finetuning)

---

## 📂 Navigation

### Agent Knowledge
- [[Agents/Orchestrator|Orchestrator]] — Workflow coordination
- [[Agents/Dataset Agent|Dataset Agent]] — Dataset generation
- [[Agents/Pipeline Agent|Pipeline Agent]] — Apriori + LLM extraction
- [[Agents/Training Agent|Training Agent]] — Fine-tuning preparation
- [[Agents/Evaluation Agent|Evaluation Agent]] — Model evaluation
- [[Agents/Deployment Agent|Deployment Agent]] — HuggingFace deployment
- [[Agents/Monitoring Agent|Monitoring Agent]] — Visuals & reporting
- [[Agents/Cleanup Agent|Cleanup Agent]] — Repo hygiene
- [[Agents/Maintainer Agent|Maintainer Agent]] — Documentation maintenance

### Key References
- [[References/API Limits|API Token Limits]]
- [[References/Model Comparison|Model Comparison]]
- [[References/Pipeline Bug 2026-02-08|Pipeline Bug (2026-02-08)]]

### Run Logs
- Browse `Logs/` for per-run activity logs

### Decisions
- Browse `Decisions/` for architecture & strategy decisions

---

## 🔗 How This Vault Works

This is the **persistent knowledge base** for the multi-agent workflow system.

**Agents read from here** before every task (Memory-First Rule):
- Each agent has a note in `Agents/` with curated knowledge
- Agents append new discoveries after completing tasks
- `[[backlinks]]` connect related knowledge across agents

**Humans browse here** in Obsidian for:
- Project understanding via the graph view
- Decision archaeology ("why did we choose X?")
- Experiment history and model comparison
- Quick reference cards (API limits, commands, etc.)

**What stays in the repo** (NOT in this vault):
- `workflow_state.json` — runtime coordination (changes every run)
- Agent definition files (`.github/agents/*.md`) — Copilot reads these for behavior
- `copilot-instructions.md` — Copilot needs this in workspace

---

## 📋 Vault Conventions

### Note Naming
- Agent memories: `Agents/{Agent Name}.md`
- Run logs: `Logs/{YYYY-MM-DD}_{agent}_{action}.md`
- Decisions: `Decisions/{YYYY-MM-DD} {Title}.md`
- References: `References/{Topic}.md`

### Tags
- `#agent/orchestrator`, `#agent/pipeline`, etc.
- `#experiment`, `#decision`, `#bug`, `#insight`
- `#model/gpt-4o`, `#model/gpt-4.1-mini`, `#model/qwen-3b`

### Memory Entry Format
```markdown
## [YYYY-MM-DD] Brief Title

**Context:** Why this is worth remembering
**Insight:** The actual knowledge/pattern/solution
**Application:** When/how to use this in future
**Tags:** #relevant #tags
```
