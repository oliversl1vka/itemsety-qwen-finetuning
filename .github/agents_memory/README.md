# Agent Memory Store

This directory contains:
1. **Persistent memory files** - Curated knowledge for each agent
2. **Workflow state** - Runtime coordination between agents

## Workflow State Management

### `workflow_state.json` (Auto-generated, Runtime Only)
Current workflow execution state:

```json
{
  "workflow_id": "wf_20260203_143022",
  "status": "running",
  "current_stage": 2,
  "stages": {
    "1_datasets": "completed",
    "2_pipeline": "running",
    "3_export": "pending",
    "4_push": "pending",
    "5_validate": "pending",
    "6_finalize": "pending"
  },
  "artifacts": {
    "datasets_count": 500,
    "pipeline_runs": 999,
    "training_examples": 0,
    "hf_space_url": null
  },
  "config": {
    "datasets": 500,
    "min_support": 3,
    "llm_model": "gpt-4.1-mini"
  },
  "started_at": "2026-02-03T14:30:22Z",
  "updated_at": "2026-02-03T16:45:10Z"
}
```

**Usage by agents:**
- Read state before executing
- Update state after completion
- Check dependencies between stages

### `workflow_state.py` (Python Module)
Helper functions for state management:

```python
from .github.agents_memory.workflow_state import load_workflow, complete_stage

# Load current state
wf = load_workflow()
print(wf.get_status_summary())

# Mark stage complete
complete_stage("1_datasets", {"datasets_count": 500})
```

## Agent Memory Files

Unlike logs (which record every action), memory files contain **curated knowledge**:
- Patterns that worked well
- Common pitfalls to avoid
- Environment-specific quirks
- Optimization discoveries
- User preferences learned over time

## Structure

```
agents_memory/
├── README.md
├── orchestrator_memory.md
├── dataset_agent_memory.md
├── pipeline_agent_memory.md
├── training_agent_memory.md
├── evaluation_agent_memory.md
├── deployment_agent_memory.md
├── monitoring_agent_memory.md
├── maintainer_agent_memory.md
└── cleanup_agent_memory.md
```

## Usage Guidelines

### When to Add Memory
- ✅ Discovered non-obvious solution to recurring problem
- ✅ Found environment-specific configuration that works
- ✅ Learned user preference through interaction
- ✅ Identified pattern that speeds up workflow
- ✅ Found workaround for external service quirk

### When NOT to Add Memory
- ❌ Routine task completion (use logs instead)
- ❌ Temporary issues that won't recur
- ❌ Information already in documentation
- ❌ Sensitive data (credentials, keys)

## Memory Entry Format

```markdown
### [YYYY-MM-DD] Brief Title

Context: Why this is worth remembering
Insight: The actual knowledge/pattern/solution
Application: When/how to use this in future
```

## Maintenance

- Review memories quarterly
- Remove outdated entries
- Consolidate related memories
- Keep file under 500 lines (archive old entries)
