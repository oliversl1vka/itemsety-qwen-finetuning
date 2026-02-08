# Pipeline Bug — 2026-02-08

**Severity:** 🔴 CRITICAL  
**Status:** ✅ Fixed  
**Tags:** #bug #critical #pipeline

---

## What Happened

A bug in `pipeline.py` caused **singletons** (Apriori size-1 itemsets) to be passed into the LLM extractor response field instead of the **actual LLM output**.

## Impact

- ALL validation "passes" before 2026-02-08 were **meaningless** — the system was validating Apriori output against itself
- The LLM was **never truly evaluated** in any pre-bugfix run
- ALL training data exported from pre-bugfix runs is **INVALID**
- Claims of "100% pass rate" for `gpt-4.1-mini` were false

## What Was Invalidated

- All pipeline runs in `runs.db` before 2026-02-08
- All training data in `data/training_v1/` and `data/training_v2/` (deleted in cleanup)
- All artifacts in `artifacts/` from pre-bugfix runs (deleted in cleanup)
- All HuggingFace datasets uploaded before this date

## Resolution

- Bug was fixed in `pipeline.py` on 2026-02-08
- runs.db was kept (760 rows from 2026-02-08 post-fix: 668 gpt-4o, 92 gpt-4.1-mini)
- Of those 760 rows: 47 passed validation, 713 failed (realistic numbers)
- All pre-bugfix artifacts were deleted in the repo cleanup

## Lessons Learned

1. **Always validate that LLM output is actually LLM output** — never trust the pipeline blindly
2. **Sanity-check pass rates** — 100% on a hard task is suspicious
3. **Timestamp all data** so invalid data can be identified and purged

---

**Related:** [[Agents/Pipeline Agent]], [[Agents/Training Agent]], [[Agents/Cleanup Agent]]
