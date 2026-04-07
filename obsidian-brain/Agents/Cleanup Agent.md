# Cleanup Agent Memory

Persistent knowledge store for repository organization insights.

**Agent file:** `.github/agents/cleanup-agent.md`  
**Tags:** #agent/cleanup

---

## [2026-02-08] Major repository cleanup

**Context:** Comprehensive cleanup reduced repo from ~13GB to 235MB.  
**Insight:** Purged 2,452 invalid artifact files, 1,863 log files, 6 old training data directories, 8.4GB archive of unsuitable datasets, 4GB temp files. All pre-2026-02-08 pipeline data was invalid due to pipeline.py bug.  
**Application:** After any pipeline bug fix, always invalidate and clean ALL artifacts from before the fix date.  
**Tags:** #insight #cleanup

See also: [[References/Pipeline Bug 2026-02-08]]

---

## [2026-03-17] 🔬 Diamond Knowledge — Knowledge Extraction Cleanup

**Source:** Diamond phase extracted knowledge from 11 Unsloth × Qwen notebooks into `knowledge_extraction/unsloth_notebooks/`

### Files Created During Diamond Phase

The `knowledge_extraction/unsloth_notebooks/` directory contains:
- `notes/DIAMOND_KNOWLEDGE.md` (309 lines) — Full synthesized knowledge
- `notes/AGENT_MEMORY_PACKET.md` (229 lines) — Agent-focused recommendations
- `notes/review_notes/` — Per-notebook review notes (11 files)
- `notes/batch_*_notes.md` — Batch summaries
- `notes/diamond_context.md` — Assembled context for synthesis
- `triage_results/`, `scoring/`, `reviews/` — Intermediate OpenAI triage outputs

**Status:** All actionable knowledge has been extracted and integrated into agent memories and training notebook (v3.2). The `knowledge_extraction/` directory can be archived or kept as reference.

**Tags:** #diamond-knowledge #cleanup #knowledge-extraction
