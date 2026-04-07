# Decision: Diamond Knowledge Integration

**Date:** 2026-03-17  
**Status:** Accepted  
**Tags:** #decision #diamond-knowledge #v3.2

---

## Context

Reviewed 11 Unsloth × Qwen notebooks (from `unslothai/notebooks` repository) through a structured diamond phase extraction process. Out of 81 Qwen-family notebooks, 22 candidates scored ≥80 in OpenAI triage, and 11 were selected for premium manual review across 2 batches.

**Notebooks reviewed:**
- Batch 1 (8): Qwen2.5 SFT (original_template + nb), Qwen2.5 GRPO, Qwen2.5 Instruct-QAT, Qwen3-4B-GRPO, QwQ-32B-GRPO, Qwen2.5-Coder SFT, Qwen2-VL GRPO
- Batch 2 (3): Qwen3-Thinking-Conversational, Qwen3-Instruct SFT, DeepSeek-R1 GRPO

**Full knowledge:** `knowledge_extraction/unsloth_notebooks/notes/DIAMOND_KNOWLEDGE.md` (309 lines)

## Decision

Apply only **non-redundant changes** that directly advance our Qwen2.5-7B fine-tuning for frequent itemset extraction.

### Changes Applied (v3.2)

| Change | Where | Why | Source |
|--------|-------|-----|--------|
| `paged_adamw_8bit` optimizer | Cell 8 (SFT), Cell 12 (DPO) | Pages optimizer states to CPU when GPU RAM is tight — strictly better than `adamw_8bit` | Coder SFT notebook |
| Label masking verification | New Cell 8b | Catches silent `train_on_responses_only` failures by decoding labels | Thinking + Coder notebooks |
| SFT format verification gate | Enhanced Cell 9 | Generates on val samples between phases — catches format regression before DPO | DeepSeek-R1 + Pre-finetuning pattern |

### Changes NOT Applied (with reasoning)

| Candidate | Why Not |
|-----------|---------|
| `packing=True` | v3 Council explicitly set False; can corrupt `train_on_responses_only` masking |
| `random_state=3407` | Our seed 42 is fine; no evidence of meaningful difference |
| `lr=2e-4` for SFT | Council overrode to 1e-4 based on our specific situation (repetition loops) |
| Vision/MoE patterns | Not applicable — text-only 7B model |
| QAT training | Alternative quantization path — not needed for current workflow |
| `standardize_data_formats()` | Our HF dataset already uses correct format |
| Prompt length filtering | Our v3 concise CoT is already uniform (~500-700 tokens) |
| A/B comparison loop | Good idea but separate eval enhancement, not training change |
| GSPO config | GRPO is skipped in v3; documented in Training Agent memory for future |

## Consequences

- Training notebook becomes v3.2 with 3 targeted improvements
- All 9 agent memories updated with diamond knowledge findings
- Knowledge extraction artifacts preserved in `knowledge_extraction/` for reference
- `notebook_versions.json` updated with v3.2 entry

## See Also

- [[Training Agent]]
- [[References/Unsloth Notebook Patterns]]
- `knowledge_extraction/unsloth_notebooks/notes/DIAMOND_KNOWLEDGE.md`
