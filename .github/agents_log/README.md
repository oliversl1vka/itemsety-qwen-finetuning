# Agent Activity Logs

This directory contains activity logs from all agents in the orchestration system.

## Structure

```
agents_log/
├── orchestrator/    # Workflow coordination logs
├── dataset/         # Dataset generation logs
├── pipeline/        # Apriori + LLM extraction logs
├── training/        # Model fine-tuning logs
├── evaluation/      # Performance evaluation logs
├── deployment/      # HF Hub deployment logs
├── monitoring/      # Observability & reporting logs
├── maintainer/      # Agent file maintenance logs
└── cleanup/         # Repository cleanup logs
```

## Log Format

Each log file follows the naming convention:
```
YYYY-MM-DD_HHMMSS_{action}.md
```

Example: `2026-02-01_143022_full_audit.md`

## Log Template

```markdown
# {Action Title}

**Agent:** {agent-name}
**Date:** {YYYY-MM-DD HH:MM:SS}
**Duration:** {X min Y sec}
**Status:** ✅ Success | ⚠️ Partial | ❌ Failed

## Summary
Brief description of what was done.

## Actions
- Action 1
- Action 2

## Outputs
- File/artifact created or modified

## Notes
Any additional context or issues encountered.
```

## Retention Policy

- Keep logs for 90 days
- Archive older logs to `agents_log/archive/`
- Delete archived logs after 1 year

## Usage

Agents automatically write logs here after completing tasks.
Query logs using:

```bash
# Find recent logs
ls -lt agents_log/*/

# Search for specific action
grep -r "dataset generation" agents_log/

# Count actions by agent
for d in agents_log/*/; do echo "$d: $(ls -1 $d 2>/dev/null | wc -l)"; done
```
