# 2026-03-09 LLM Council v3 Plan

**Date:** 2026-03-09
**Type:** Council Session (Planning)
**Agent:** Training Agent
**Tags:** #council #v3-plan #repetition-loop

---

## Objective

Decide v3 training strategy based on v2 adapter evaluation results (repetition loops on 87% of datasets, 78.6% F1 on small datasets only).

## Council Configuration

| Model | Status | Response Time | Response Length |
|-------|--------|---------------|-----------------|
| google/gemini-3-flash-preview | ✅ | 12.6s | 4,908 chars |
| deepseek/deepseek-v3.2 | ✅ | 74.9s | 10,262 chars |
| x-ai/grok-4.1-fast | ✅ | 24.3s | 8,655 chars |
| anthropic/claude-sonnet-4.6 | ❌ | Timeout | Connection drop via OpenRouter |

**Query:** 9,704 chars covering full training history (v1→v2), both eval datasets (raw_capture + reppenalty), 9 specific questions.

## Key Findings

### Unanimous Consensus

1. **Root Cause:** CoT format too repetitive (60%) + over-training 5 epochs (25%) + insufficient diversity (15%)
2. **Fix:** Concise CoT (no evidence in `<think>`, only counts) + reduce epochs to 3 + expand to 1000 examples
3. **DPO:** YES, add after SFT (1 epoch, beta=0.1, real v2 failures as rejected)
4. **GRPO:** SKIP entirely
5. **LoRA:** r=32 (down from 64), alpha=64, dropout=0.1
6. **Save:** NEVER merge_4bit_forced. Adapter-only loading.
7. **Inference:** temp=0.05, top_k=40, StoppingCriteria at `</think>`, constrained decoding

### Disagreements (minor)

- LR range: 5e-5 (Gemini) to 1.5e-4 (DeepSeek) → median 1e-4
- Data count: 748 (DeepSeek) to 1200 (Grok) → target 1000
- LoRA alpha: 32 (Grok) vs 64 (DeepSeek/Gemini) → go with 64

## v3 Plan Summary

```
Phase 0: Data Prep (1-2 days) → 1000 concise CoT + 500 DPO pairs
Phase 1: SFT (4-6h) → r=32, alpha=64, lr=1e-4, 3 epochs, 2048 seq
Phase 2: DPO (2-3h) → beta=0.1, lr=5e-5, 1 epoch
Phase 3: Eval (1h) → target F1≥0.80, parse≥0.90
```

## Immediate Actions

1. Test v2 adapter with inference fixes (before retraining)
2. Generate concise CoT training data
3. Execute v3 training plan

## Files

- Council report: `docs/reports/council_v3_plan_2026-03-09.json`
- Raw responses: `docs/reports/council_v3_partial.json`
- Council script: `scripts/council_v3_sequential.py`

## Related

- [[Agents/Training Agent]]
- [[Experiments/2026-03-08 Adapter Eval v2]]
- [[Experiments/2026-03-07 Qwen2.5-7B v2]]
