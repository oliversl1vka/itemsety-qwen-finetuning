# Dataset Agent Memory

Persistent knowledge store for dataset generation insights.

**Agent file:** `.github/agents/dataset-agent.md`  
**Tags:** #agent/dataset

---

## [2026-02-03] Optimized size distribution working well

**Context:** 500 datasets generated with optimized distribution.  
**Insight:**
- 47% small (5-10 rows), 35% medium (11-18 rows), 18% large (19-25 rows)
- Average token usage ~272 per dataset (well within LLM context limits)
- 10 valid source datasets loaded from `real_datasets/` (15 skipped for insufficient categorical columns)
- Variation strategy distribution: 27% row_subsample, 23% combined, 19% col_subsample, 16% shuffle, 15% noise

**Application:** All datasets fit within typical LLM context windows (32k-128k tokens). Current distribution is well-balanced.  
**Tags:** #insight #dataset-generation
