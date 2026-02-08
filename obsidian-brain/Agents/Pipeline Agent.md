# Pipeline Agent Memory

Persistent knowledge store for Apriori + LLM extraction insights.

**Agent file:** `.github/agents/pipeline-agent.md`  
**Tags:** #agent/pipeline

---

## [2026-02-08] ⚠️ INVALIDATED: 2026-02-03 batch run data

**Context:** A bug in `pipeline.py` caused singletons (Apriori size-1 itemsets) to be passed into the LLM extractor response field instead of the actual LLM output. This means all validation "passes" were meaningless — the LLM was never truly evaluated.  
**Insight:** ALL data generated before 2026-02-08 is INVALID and must not be used for training.  
**Application:** Only trust pipeline runs from 2026-02-08 onward. See [[References/Pipeline Bug 2026-02-08]] for full details.  
**Tags:** #bug #critical

---

## [2026-02-08] Validated Model Test Results (post-bugfix)

**Context:** After fixing the pipeline.py singleton bug, user tested models on real LLM extraction.

| Model | Result | Details |
|-------|--------|---------|
| `gpt-4o-mini` | ❌ **FAILED** (0% pass) | Found 4/31 itemsets, all invalid. Hallucinated item presence in rows. |
| `gpt-4o` | ✅ **PASSED** (~80% pass) | Good quality, but in the expensive API tier (250k tokens/day limit). |
| `gpt-4.1-mini` | ⏳ **NOT YET TESTED** post-bugfix | Expected to work well. In cheap API tier (2.5M tokens/day). **Test this first.** |

**Insight:** Prefer `gpt-4.1-mini` (pending validation). It's in the high-limit tier and should handle 500+ datasets in a single day.  
**Application:** Test `gpt-4.1-mini` on 5-10 datasets first. If it passes, use for full batch.  
**Tags:** #experiment #model/gpt-4o #model/gpt-4o-mini #model/gpt-4.1-mini

See also: [[References/Model Comparison]], [[References/API Limits]]

---

## [2026-02-08] OpenAI API Token Limits

**Context:** API tier awareness is critical for planning batch runs.

See [[References/API Limits]] for the full breakdown.

**Key insight:** `gpt-4.1-mini` is in Tier 2 (2.5M/day). At ~2000 tokens/dataset, this supports **~1250 datasets/day** — more than enough for our 500-711 dataset batch.

**Pitfall:** `gpt-4o` is in Tier 1 (250k/day) — only ~125 datasets/day. Do NOT plan batch runs with Tier 1 models.  
**Tags:** #insight #api
