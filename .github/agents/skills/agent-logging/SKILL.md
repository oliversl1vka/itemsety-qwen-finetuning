---
name: agent-logging
description: Record agent activities to .github/agents_log/ and insights to .github/agents_memory/. Use after completing any significant task.
---

# Agent Logging & Memory

Record agent activities and persist useful insights for future reference.

## Overview

Two logging mechanisms:
- **Activity Logs** (`.github/agents_log/`): Record every significant action
- **Persistent Memory** (`.github/agents_memory/`): Store curated insights

## Activity Logs

### Location
```
.github/agents_log/
├── orchestrator/
├── dataset/
├── pipeline/
├── training/
├── evaluation/
├── deployment/
├── monitoring/
├── maintainer/
└── cleanup/
```

### Log Naming Convention
```
YYYY-MM-DD_HHMMSS_{action}.md
```
Example: `2026-02-01_143022_batch_generation.md`

### Log Template
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
```

### Example Log Entry
```markdown
# Batch Pipeline Run

**Agent:** pipeline-agent
**Date:** 2026-02-01 14:30:22
**Duration:** 45 min 12 sec
**Status:** ⚠️ Partial

## Summary
Processed 500 datasets through Apriori + LLM extraction pipeline.

## Actions
- Loaded 500 CSV files from data/datasets_v2/
- Ran Apriori on all datasets
- Called Azure OpenAI for LLM extraction
- Validated outputs against 13 invariants
- Persisted results to runs.db

## Outputs
- `artifacts/apriori_outputs/` (500 files)
- `artifacts/extractor_outputs/` (487 files)
- `artifacts/validation_reports/` (500 files)

## Metrics
- Datasets processed: 500
- Validation passed: 439 (87.8%)
- Validation failed: 61 (12.2%)
- Avg Apriori time: 0.3s
- Avg LLM time: 42s
- Total duration: 45 min

## Notes
- 13 datasets failed due to Azure API rate limiting (429)
- Retry with smaller chunk size recommended
```

## Persistent Memory

### Location
```
.github/agents_memory/
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

### Memory Entry Format
```markdown
### [YYYY-MM-DD] Brief Title

**Context:** Why this is worth remembering
**Insight:** The actual knowledge/pattern/solution
**Application:** When/how to use this in future
```

### Example Memory Entry
```markdown
### [2026-02-01] Azure API Rate Limit Mitigation

**Context:** Pipeline failed on batch of 500 datasets with 429 errors
**Insight:** Chunk size 25 + 2 second delay between calls prevents rate limiting. Azure has 60 RPM limit for GPT-4.
**Application:** Always use `--llm-chunk-size 25` for batches > 50 datasets. Add retry logic with exponential backoff.
```

## When to Log

### Activity Logs (Always)
- ✅ Completed batch processing
- ✅ Training run finished
- ✅ Deployment completed
- ✅ Evaluation finished
- ✅ Cleanup operation done
- ✅ Any failed operation

### Memory (Selectively)
- ✅ Discovered non-obvious solution
- ✅ Found environment-specific workaround
- ✅ Learned user preference
- ✅ Identified performance optimization
- ❌ Routine task completion (use logs)
- ❌ Temporary issues
- ❌ Already documented information

## Python Utilities

### Write Log
```python
from datetime import datetime
from pathlib import Path

def write_activity_log(agent: str, action: str, content: dict) -> str:
    """Write activity log entry."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_dir = Path(f".github/agents_log/{agent}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / f"{timestamp}_{action}.md"
    
    log_content = f"""# {content['title']}

**Agent:** {agent}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Duration:** {content.get('duration', 'N/A')}
**Status:** {content.get('status', '✅ Success')}

## Summary
{content.get('summary', '')}

## Actions
{chr(10).join(f"- {a}" for a in content.get('actions', []))}

## Outputs
{chr(10).join(f"- `{o}`" for o in content.get('outputs', []))}

## Notes
{content.get('notes', 'None')}
"""
    
    log_path.write_text(log_content)
    return str(log_path)
```

### Append Memory
```python
def append_memory(agent: str, title: str, context: str, insight: str, application: str) -> bool:
    """Append insight to agent memory file."""
    memory_path = Path(f".github/agents_memory/{agent}_memory.md")
    
    entry = f"""
### [{datetime.now().strftime("%Y-%m-%d")}] {title}

**Context:** {context}
**Insight:** {insight}
**Application:** {application}
"""
    
    with open(memory_path, "a") as f:
        f.write(entry)
    
    return True
```

## Querying Logs

### Find Recent Logs
```bash
# Last 10 logs for pipeline agent
ls -lt .github/agents_log/pipeline/ | head -10

# Search for specific action
grep -r "rate limit" .github/agents_log/
```

### Count by Status
```bash
# Count successes
grep -r "✅ Success" .github/agents_log/ | wc -l

# Count failures
grep -r "❌ Failed" .github/agents_log/ | wc -l
```

## Retention Policy

- **Activity Logs:** Keep 90 days, then archive
- **Memory:** Keep indefinitely, review quarterly
- **Archive Location:** `.github/agents_log/archive/YYYY-MM/`
