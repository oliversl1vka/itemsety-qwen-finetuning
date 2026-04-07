# Maintainer Agent Memory

Persistent knowledge store for agent file maintenance insights.

**Agent file:** `.github/agents/maintainer-agent.md`  
**Tags:** #agent/maintainer

---

## [2026-02-03] First workflow documentation audit

**Context:** First multi-agent workflow execution (wf_20260203_113413).  
**Insight:**
- Documentation (README.md, AGENTS.md, WORKFLOW_GUIDE.md) was accurate, no drift detected
- All 9 agent files up to date with current scripts and workflows
- Security: .env properly gitignored, HF_TOKEN not in git history ✅
- Git workflow smooth (stage → commit → push)

**Application:** Documentation-as-code approach prevents drift. Interactive multi-agent system working as designed.  
**Tags:** #insight #documentation

---

## [2026-03-17] Diamond knowledge integration — documentation updates

**Context:** Integrated findings from diamond knowledge extraction (11 Unsloth × Qwen notebooks reviewed) into project files.

**Files modified:**
- `obsidian-brain/Agents/Training Agent.md` — Validated defaults, optimizer upgrade, label verification, GSPO future config
- `obsidian-brain/Agents/Evaluation Agent.md` — Inference temperature calibration, A/B comparison pattern
- `obsidian-brain/Agents/Deployment Agent.md` — GGUF multi-quant export, 16-bit merge path
- `obsidian-brain/Agents/Dataset Agent.md` — Prompt length filtering, chat template, packing consideration
- `obsidian-brain/Agents/Orchestrator.md` — Workflow verification gates between phases
- `obsidian-brain/Agents/Monitoring Agent.md` — VRAM tracking, reward distribution monitoring
- `obsidian-brain/Agents/Pipeline Agent.md` — FastLanguageModel.for_inference() confirmed
- `obsidian-brain/Agents/Cleanup Agent.md` — Knowledge extraction cleanup guidance
- `obsidian-brain/Agents/Maintainer Agent.md` — This entry
- `notebooks/training_3phase_2026-03-09_v3.ipynb` — v3.2 changes (paged_adamw_8bit, label masking verification, format gate)
- `notebooks/notebook_versions.json` — v3.2 entry added
- `obsidian-brain/Decisions/Diamond Knowledge Integration 2026-03-17.md` — Decision log
- `obsidian-brain/References/Unsloth Notebook Patterns.md` — Reference note

**Key principle:** Only non-redundant changes that advance the fine-tuning goal. Many diamond findings were already implemented in v3 (train_on_responses_only, gradient checkpointing, adapter-only save, memory cleanup). Changes applied: `paged_adamw_8bit`, label masking verification, SFT format gate.

**Tags:** #documentation #diamond-knowledge #v3.2
