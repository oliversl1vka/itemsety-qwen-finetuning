# Model Comparison

**Last Updated:** 2026-02-08  
**Tags:** #reference #experiment

---

## Post-Bugfix Results (2026-02-08+)

Only results from after the [[References/Pipeline Bug 2026-02-08|pipeline bug fix]] are valid.

| Model | Validation Pass Rate | API Tier | Daily Limit | Status |
|-------|---------------------|----------|-------------|--------|
| `gpt-4o-mini` | ❌ **0%** | Tier 2 (2.5M/day) | Cheap but useless | Hallucinated items in rows |
| `gpt-4o` | ✅ **~80%** | Tier 1 (250k/day) | ~125 datasets/day | Works, but expensive tier |
| `gpt-4.1-mini` | ⏳ **Not tested** | Tier 2 (2.5M/day) | ~1250 datasets/day | **Test this first** |
| `gpt-4.1` | ⏳ **Not tested** | Tier 1 (250k/day) | ~125 datasets/day | Expensive tier |

## Pre-Bugfix Results (INVALID — do not use)

All results before 2026-02-08 used buggy pipeline that passed singletons instead of LLM output. "100% pass rates" were meaningless.

## Fine-Tuned Model Results

| Model | Version | F1 | Precision | Recall | JSON Parse | Notes |
|-------|---------|-----|-----------|--------|------------|-------|
| *(none yet — first valid training pending)* | | | | | | |

> Add rows here after each `/validate` cycle

---

**Related:** [[References/API Limits]], [[Agents/Pipeline Agent]], [[Agents/Training Agent]]
