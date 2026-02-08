---
name: agent-logging
description: Record agent activities to obsidian-brain/Logs/ and insights to obsidian-brain/Agents/. Use after completing any significant task.
---

# Agent Logging & Memory (Obsidian Brain)

Record agent activities and persist useful insights using the **Obsidian vault** at `obsidian-brain/`.

## Overview

All knowledge management uses the Obsidian vault with `[[backlinks]]` for cross-referencing:

| Purpose | Location | Template |
|---------|----------|----------|
| Agent memory & insights | `obsidian-brain/Agents/{Agent Name}.md` | Memory Entry |
| Activity logs | `obsidian-brain/Logs/{YYYY-MM-DD}_{agent}_{action}.md` | Run Log |
| Training experiments | `obsidian-brain/Experiments/{experiment_name}.md` | Experiment |
| Architecture decisions | `obsidian-brain/Decisions/{decision_name}.md` | Decision |
| Reference docs | `obsidian-brain/References/` | — |

## Vault Structure

```
obsidian-brain/
├── Home.md                        # Navigation hub
├── Agents/                        # Agent memory notes (9 files)
│   ├── Orchestrator.md
│   ├── Pipeline Agent.md
│   ├── Dataset Agent.md
│   ├── Training Agent.md
│   ├── Evaluation Agent.md
│   ├── Deployment Agent.md
│   ├── Monitoring Agent.md
│   ├── Cleanup Agent.md
│   └── Maintainer Agent.md
├── References/                    # Shared knowledge
│   ├── API Limits.md
│   ├── Model Comparison.md
│   └── Pipeline Bug 2026-02-08.md
├── Templates/                     # Note templates
│   ├── Run Log.md
│   ├── Decision.md
│   ├── Memory Entry.md
│   └── Experiment.md
├── Logs/                          # Per-run activity logs
├── Experiments/                   # Training experiment reports
└── Decisions/                     # Architecture decisions
```

## Activity Logs

### Log Naming Convention
```
obsidian-brain/Logs/{YYYY-MM-DD}_{agent}_{action}.md
```
Example: `obsidian-brain/Logs/2026-02-08_pipeline_batch_run.md`

### Log Template (use Run Log template)
```markdown
# {Action Title}

**Agent:** {agent-name}
**Date:** {YYYY-MM-DD HH:MM:SS}
**Duration:** {X min Y sec}
**Status:** ✅ Success | ⚠️ Partial | ❌ Failed

## Summary
Brief description of what was done.

## Actions
- Action 1: Description
- Action 2: Description

## Outputs
- `path/to/output1`
- `path/to/output2`

## Metrics
- Items processed: N
- Success rate: X%
- Duration: Y seconds

## Notes
Any issues, warnings, or observations.

## Links
- Related agent: [[Agents/Pipeline Agent]]
- Related reference: [[References/API Limits]]
```

**Key:** Always add `[[backlinks]]` to connect logs to related agents, references, experiments, and decisions.

## Agent Memory

### Location
Each agent has a dedicated memory note at `obsidian-brain/Agents/{Agent Name}.md`.

### Memory Entry Format
Append new entries to the agent's memory note:

```markdown
### [YYYY-MM-DD] Brief Title

**Context:** Why this is worth remembering
**Insight:** The actual knowledge/pattern/solution
**Application:** When/how to use this in future

Related: [[References/Relevant Reference]] | [[Logs/YYYY-MM-DD_agent_action]]
```

### Tags
Use Obsidian tags for discoverability:
- `#bug` — Bug discoveries
- `#insight` — Learnings & patterns
- `#model/gpt-4o` `#model/gpt-4.1-mini` — Model-specific notes
- `#experiment` — Training experiment references
- `#decision` — Architecture decisions

## When to Log

### Activity Logs (Always → `obsidian-brain/Logs/`)
- ✅ Completed batch processing
- ✅ Training run finished
- ✅ Deployment completed
- ✅ Evaluation finished
- ✅ Cleanup operation done
- ✅ Any failed operation

### Memory (Selectively → `obsidian-brain/Agents/`)
- ✅ Discovered non-obvious solution
- ✅ Found environment-specific workaround
- ✅ Learned user preference
- ✅ Identified performance optimization
- ❌ Routine task completion (use logs)
- ❌ Temporary issues
- ❌ Already documented information

### Experiments (→ `obsidian-brain/Experiments/`)
- ✅ Training run with metrics
- ✅ Hyperparameter comparisons
- ✅ Model evaluation results

### Decisions (→ `obsidian-brain/Decisions/`)
- ✅ Significant architecture changes
- ✅ Tool/library choices
- ✅ Breaking workflow changes

## Cross-Referencing with Backlinks

Use `[[backlinks]]` to connect knowledge across notes:

```markdown
# In a log entry:
This run discovered the [[References/Pipeline Bug 2026-02-08]] which affected all prior data.

# In agent memory:
See [[Logs/2026-02-08_pipeline_batch_run]] for details. Updated [[References/Model Comparison]].

# In an experiment:
Training used data from [[Logs/2026-02-03_training_export]]. Results tracked by [[Agents/Training Agent]].
```

## Querying the Vault

### In Obsidian App
- Use **graph view** to see relationships between notes
- Use **search** to find across all notes
- Click any `[[backlink]]` to navigate

### From Terminal
```bash
# Find recent logs
ls -lt obsidian-brain/Logs/ | head -10

# Search for specific topic
grep -r "rate limit" obsidian-brain/

# Count logs by status
grep -r "✅ Success" obsidian-brain/Logs/ | wc -l
grep -r "❌ Failed" obsidian-brain/Logs/ | wc -l
```

## Retention Policy

- **Activity Logs:** Keep indefinitely (Obsidian handles large vaults well)
- **Agent Memory:** Keep indefinitely, review quarterly
- **Experiments:** Keep indefinitely (critical for model improvement history)
- **Decisions:** Keep indefinitely (architectural record)
