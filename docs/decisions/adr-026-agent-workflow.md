# ADR-026: Multi-Agent AI Workflow

**Status:** Accepted  
**Date:** 2026-02

## Context

This project involves 8 distinct stages: repository organization, dataset generation, pipeline execution, training data export, model deployment, evaluation, monitoring, and finalization. How should the development workflow be structured?

## Options Considered

| Approach | Velocity | Traceability | Consistency |
|----------|----------|-------------|-------------|
| Single-developer manual scripts | Low | Git history only | Variable |
| Pair programming (human + AI) | Medium | Session-based | Good |
| **Multi-agent AI workflow** | High | Persistent memory vault | High |

## Decision

**9 specialized AI agents** with persistent memory, orchestrated through a defined workflow.

## Rationale

The project decomposes naturally into 8 independent stages, each requiring different domain knowledge:

| Agent | Stage | Domain |
|-------|-------|--------|
| Orchestrator | All | Workflow coordination |
| Dataset Agent | 2 | CSV generation, data quality |
| Pipeline Agent | 3 | Apriori, LLM APIs, validation |
| Training Agent | 4 | SFT/DPO data export, notebook generation |
| Deployment Agent | 5 | HuggingFace Hub operations |
| Evaluation Agent | Post-training | Metrics, inference optimization |
| Monitoring Agent | 7 | Visualization, reporting |
| Cleanup Agent | Ongoing | Artifact management |
| Maintainer Agent | Ongoing | Repository maintenance |

Each agent has a constrained scope and reads its persistent memory from `obsidian-brain/Agents/<name>.md` before any task (the "Memory-First Rule"). This ensures continuity across development sessions and prevents conflicting actions.

**The human role:** Design the system architecture, define success criteria (Apriori oracle, 13 validation invariants, 7 evaluation metrics), review all outputs, and make final decisions. AI agents execute within these guardrails.

## Consequences

- Every major architectural decision is logged in `obsidian-brain/Decisions/` as a Decision Record
- Experiment results are timestamped in `obsidian-brain/Experiments/`
- Workflow state is tracked in `.github/agents_memory/workflow_state.json`
- The audit trail enables post-hoc review of why each decision was made and when

## Source Evidence

- `.github/agents/` -- 9 agent definitions + 12 skill modules
- `obsidian-brain/` -- persistent memory vault (35 markdown files)
- `AGENTS.md` -- workflow documentation
- `.github/WORKFLOW_GUIDE.md` -- 8-stage workflow guide
