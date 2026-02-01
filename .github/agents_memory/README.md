# Agent Memory Store

This directory contains persistent memory files for each agent. Agents use these files to record insights, lessons learned, and useful knowledge that may help in future tasks.

## Purpose

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
