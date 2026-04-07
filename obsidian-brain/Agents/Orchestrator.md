# Orchestrator Memory

Persistent knowledge store for workflow coordination insights.

**Agent file:** `.github/agents/orchestrator.md`  
**Tags:** #agent/orchestrator

---

<!-- Append new memories below, newest last -->

## [2026-03-17] 🔬 Diamond Knowledge — Workflow Validation Patterns

**Source:** Review of 11 Unsloth × Qwen notebooks (diamond phase extraction)

### Mandatory Verification Gates Between Phases

Diamond review confirmed: **every** multi-phase notebook includes intermediate verification between training phases. Our v3 notebook had a placeholder eval gate (Cell 9: print-only). v3.2 now adds:

1. **Label masking verification** (new cell after SFT trainer creation): Decode tokenized labels, confirm `-100` on prompt tokens and real token IDs on assistant response tokens. Catches silent masking failures.

2. **SFT format verification gate** (enhanced Cell 9): Generate on 2 validation samples after SFT. Check for `<think>`, valid JSON array, `col:value` format. If gate fails → increase SFT epochs before DPO.

### Memory Cleanup Mandate

All multi-phase notebooks follow: `del model, trainer; gc.collect(); torch.cuda.empty_cache()` between phases. Already implemented in our v3 Cell 9 ✅.

### Adapter-Only Save Between Phases

Diamond review: 100% of reviewed notebooks save LoRA adapters only (not merged). Our v3 notebook correctly does this ✅. The v3.1 fix (continuing same adapter for DPO) is the correct pattern — confirmed by multiple notebooks.

### v3.2 Changes Applied

| Change | Cell | Source |
|--------|------|--------|
| `paged_adamw_8bit` optimizer | Cell 8 (SFT), Cell 12 (DPO) | Coder notebook: better memory paging |
| Label masking verification | New cell after Cell 8 | Thinking + Coder notebooks: silent failure prevention |
| SFT format verification gate | Enhanced Cell 9 | DeepSeek-R1 + Pre-finetuning pattern: generate-and-check |

**Tags:** #diamond-knowledge #workflow-validation #phase-gates

See also: [[Training Agent]] [[References/Unsloth Notebook Patterns]] [[Decisions/Diamond Knowledge Integration 2026-03-17]]
