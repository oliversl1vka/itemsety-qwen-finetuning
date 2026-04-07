# Unsloth Notebook Patterns

**Date:** 2026-03-17  
**Source:** Diamond knowledge extraction from 11 Unsloth × Qwen notebooks  
**Tags:** #reference #unsloth #diamond-knowledge

---

## Quick Reference

### Universal Patterns (all notebooks)
- `FastLanguageModel.from_pretrained()` with `max_seq_length`, `load_in_4bit=True`, `dtype=None`
- `get_peft_model()` with target_modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- `use_gradient_checkpointing="unsloth"` for 30% VRAM savings
- `train_on_responses_only()` with label verification for instruction tuning
- `FastLanguageModel.for_inference(model)` before `.generate()` for 2× speedup
- Memory cleanup between phases: `del model, trainer; gc.collect(); torch.cuda.empty_cache()`
- Adapter-only save: `model.save_pretrained()` — NEVER `merge_and_unload()` on 4-bit

### SFT Defaults
| Parameter | Value |
|-----------|-------|
| learning_rate | 2e-4 |
| optim | paged_adamw_8bit |
| batch_size | 2 |
| gradient_accumulation | 4 |
| weight_decay | 0.001-0.01 |
| lr_scheduler | linear or cosine |
| packing | False (True for short sequences, test first) |

### LoRA Defaults
| Parameter | SFT | GRPO |
|-----------|-----|------|
| r (rank) | 16 | 32 |
| lora_alpha | =r | =2r |
| lora_dropout | 0 | 0 |

### GRPO/GSPO Config
| Parameter | Value |
|-----------|-------|
| learning_rate | 5e-6 |
| max_grad_norm | 0.1 |
| weight_decay | 0.1 |
| num_generations | 2-4 |
| GSPO flags | `importance_sampling_level="sequence"`, `loss_type="dr_grpo"` |

### Our v3.2 Overrides (Council + Diamond)
| Parameter | Unsloth Default | Our Value | Reason |
|-----------|----------------|-----------|--------|
| learning_rate (SFT) | 2e-4 | 1e-4 | Council: prevent repetition loops |
| lora_r | 16 | 32 | Council: more capacity for structured extraction |
| lora_alpha | 16 | 64 (ratio 2.0) | Council: original ratio 0.25 was too low |
| lora_dropout | 0 | 0.05 | v3: regularization against repetition |
| optim | adamw_8bit | paged_adamw_8bit | Diamond: better memory paging |
| seed | 3407 | 42 | Our choice |
| packing | varies | False | Council: protect masking boundaries |

## Full Knowledge Document

See `knowledge_extraction/unsloth_notebooks/notes/DIAMOND_KNOWLEDGE.md` for complete extraction (309 lines).

## See Also

- [[Training Agent]]
- [[Decisions/Diamond Knowledge Integration 2026-03-17]]
