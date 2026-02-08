# Training Agent Memory

Persistent knowledge store for model fine-tuning insights.

**Agent file:** `.github/agents/training-agent.md`  
**Tags:** #agent/training

---

## [2026-02-08] ⚠️ All pre-2026-02-08 pipeline data is INVALID

**Context:** A bug in `pipeline.py` passed singletons (Apriori size-1 itemsets) into the LLM extractor response field instead of the actual LLM output. All validation "passes" before this date were meaningless.  
**Insight:** Training data exported from runs before 2026-02-08 MUST NOT be used. Only post-bugfix pipeline runs produce real training signal.  
**Application:** When running `/export`, verify all source runs are from 2026-02-08 or later.  
**Tags:** #bug #critical

See also: [[References/Pipeline Bug 2026-02-08]]

---

## [2026-02-08] LLM Model Selection for Training Data Generation

**Context:** After fixing the pipeline bug, user tested models on real LLM extraction.

**Validated results (post-bugfix only):**

| Model | Result | API Tier | Daily Limit |
|-------|--------|----------|-------------|
| `gpt-4o-mini` | ❌ **0% validation pass** | Tier 2 (2.5M/day) | Cheap but useless — hallucinated items in rows |
| `gpt-4o` | ✅ **~80% validation pass** | Tier 1 (250k/day) | Works but only ~125 datasets/day |
| `gpt-4.1-mini` | ⏳ **Not yet tested post-bugfix** | Tier 2 (2.5M/day) | ~1250 datasets/day — **test this first** |

**Insight:** Prefer `gpt-4.1-mini` (pending validation). It's in the high-limit tier (2.5M tokens/day) and should handle 500+ datasets in a single day. If it fails, fall back to `gpt-4o` (will take ~4-6 days due to 250k limit).

**Pitfalls:**
- ❌ Never use `gpt-4o-mini` — 0% validation pass, wastes tokens
- ❌ Don't confuse `gpt-4o-mini` (bad) with `gpt-4.1-mini` (expected good)
- ⚠️ Don't trust any "100% pass rate" claims from before 2026-02-08 (pipeline bug)

**Tags:** #experiment #model/gpt-4o #model/gpt-4o-mini #model/gpt-4.1-mini

See also: [[References/Model Comparison]], [[References/API Limits]]

---

## [2026-02-08] OpenAI API Token Limits

See [[References/API Limits]] for the full breakdown.

**Planning guidance:** At ~2000 tokens/dataset:
- Tier 1: ~125 datasets/day (multi-day batch)
- Tier 2: ~1250 datasets/day (single-day batch)

**Application:** Always prefer Tier 2 models for batch pipeline runs. Use `--llm-model gpt-4.1-mini`.  
**Tags:** #insight #api
