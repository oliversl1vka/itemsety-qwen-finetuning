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

## [2026-02-25] ❌ gpt-4o-mini Re-tested — Still FAILS (1/9 pass, ~11%)

**Context:** User requested re-testing `gpt-4o-mini` for a 500-dataset batch run. Tested on 9 datasets (ds_0001 through ds_0009).

**Results:**
- Passed: **1/9 (11%)** — and the only "pass" was because the model returned **0 itemsets** (vacuous pass)
- All real extraction attempts failed validation
- Primary error: `item_missing_in_row` — model hallucinating that items exist in rows when they don't
- LLM typically finds only 3-6 itemsets vs Apriori's 31-43 (massive under-extraction)
- `llm_valid_ratio: 0.0` on most datasets (zero valid itemsets when it does extract)

**Error pattern:** `gpt-4o-mini` consistently hallucinates item presence. E.g., claims `freetime:3` is in Row 6, but the actual data says otherwise. This is the same behavior seen on 2026-02-08.

**Conclusion:** `gpt-4o-mini` is **NOT viable** for this pipeline. Do NOT run a full 500-dataset batch with it — it would produce ~0 usable training examples and waste API quota.

**Viable models for extraction (ordered by quality):**

| Model | Pass Rate | Viable? | API Tier |
|-------|-----------|---------|----------|
| `gpt-4o` | ~80% | ✅ Yes (expensive) | Tier 1 (250k/day) |
| `gpt-4.1-mini` | 40.2% | ✅ Yes (primary) | Tier 2 (2.5M/day) |
| `gpt-4.1-nano` | 16.2% | ⚠️ Marginal | Tier 2 (2.5M/day) |
| `gpt-4o-mini` | ~11% (vacuous) | ❌ No | Tier 1 |
| `gpt-5-mini` | N/A (empty output) | ❌ No | Reasoning model |
| `gpt-5-nano` | N/A (hangs) | ❌ No | Reasoning model |

**Tags:** #experiment #model/gpt-4o-mini #negative-result

See also: [[References/Model Comparison]]

---

## [2026-02-17] ✅ gpt-4.1-mini Full Batch Run — COMPLETED

**Context:** Ran full 500-dataset batch with `gpt-4.1-mini` (post-bugfix, Tier 2 API).

**Results:**
- Total runs: 500
- Validated (passed): **201 (40.2%)**
- Failed: 299 (59.8%)
- Date range: 2026-02-17 → 2026-02-18
- Command: `python pipeline.py --data-dir data/datasets_v2 --llm-full --llm-model gpt-4.1-mini --min-support 3`

**Insight:** 40.2% pass rate is good for this dataset distribution. The 59.8% failures are mostly due to strict invariant checks (row count mismatches, support math). These validated runs ARE usable for RLHF training.
**Tags:** #experiment #model/gpt-4.1-mini #production

---

## [2026-02-22] ✅ gpt-4.1-nano Full Batch Run — COMPLETED

**Context:** Ran full 500-dataset batch with `gpt-4.1-nano` (Tier 2 API, smaller/cheaper model).

**Results:**
- Total runs: 500
- Validated (passed): **81 (16.2%)**
- Failed: 419 (83.8%)
- Date range: 2026-02-22
- Command: `python pipeline.py --data-dir data/datasets_v2 --llm-full --llm-model gpt-4.1-nano --min-support 3`

**Insight:** 16.2% pass rate is lower than gpt-4.1-mini (40.2%). The nano model struggles with strict itemset extraction. These 81 validated runs still contribute to RLHF training but at lower quality.
**Recommendation:** Prefer `gpt-4.1-mini` as the primary model for future batches. `gpt-4.1-nano` adds marginal diversity.
**Tags:** #experiment #model/gpt-4.1-nano #production

---

## [2026-02-22] Production DB Summary (both models complete)

| Model | Total | Valid | Pass Rate | API Tier |
|-------|-------|-------|-----------|----------|
| `gpt-4.1-mini` | 500 | 201 | **40.2%** | Tier 2 (2.5M/day) |
| `gpt-4.1-nano` | 500 | 81 | **16.2%** | Tier 2 (2.5M/day) |
| **TOTAL** | **1000** | **282** | **28.2%** | — |

**282 validated runs → SUPERSEDED by corrected numbers (see 2026-03-01 entry)**
**Tags:** #summary #production

---

## [2026-02-08] OpenAI API Token Limits

**Context:** API tier awareness is critical for planning batch runs.

See [[References/API Limits]] for the full breakdown.

**Key insight:** `gpt-4.1-mini` is in Tier 2 (2.5M/day). At ~2000 tokens/dataset, this supports **~1250 datasets/day** — more than enough for our 500-711 dataset batch.

**Pitfall:** `gpt-4o` is in Tier 1 (250k/day) — only ~125 datasets/day. Do NOT plan batch runs with Tier 1 models.  
**Tags:** #insight #api

---

## [2026-02-25] 🚫 gpt-5-mini & gpt-5-nano — UNUSABLE for extraction

**Context:** User attempted to run pipeline with `gpt-5-mini`. Previous attempts timed out every time. Debugged on 2026-02-25.

**Root cause — TWO compounding issues:**

1. **`temperature=0.0` rejected:** `gpt-5-mini` returns HTTP 400: *"Unsupported value: 'temperature' does not support 0.0 with this model."* — it's a reasoning model like o1/o3/o4.
2. **`pipeline.py` reasoning model detection is incomplete:** The `is_reasoning_model` check only matches `o1-`, `o3-`, `o4-` prefixes, so `gpt-5-*` models are NOT detected as reasoning models → `temperature=0.0` is sent → 400 error or infinite hang.
3. **Even without temperature, gpt-5-mini is USELESS:** All 4096 completion tokens are consumed by internal chain-of-thought reasoning. The visible `content` field is **empty string** (`""`). The model "thinks" for ~55 seconds and produces zero usable output. Tested with `reasoning_effort=low`, `medium`, and default — same result every time.
4. **gpt-5-nano hangs indefinitely:** Even with explicit `timeout=120s` on the raw OpenAI client, gpt-5-nano never returns (connection hangs at SSL read level). The `request_timeout` parameter on LangChain `ChatOpenAI` defaults to `None` (infinite), making this an infinite block.

**Test results (gpt-5-mini with raw OpenAI client, timeout=600s):**

| Config | Time | Tokens | Visible Content |
|--------|------|--------|-----------------|
| `reasoning_effort=low, max_tokens=4096` | 55.8s | 8392 | **empty** |
| `reasoning_effort=medium, max_tokens=4096` | 51.4s | 8392 | **empty** |
| `no special params, max_tokens=4096` | 57.8s | 8392 | **empty** |

**Conclusion:** GPT-5 family models are reasoning models that use internal CoT. They are fundamentally incompatible with structured JSON extraction tasks that need visible output. Do NOT attempt to use `gpt-5-mini`, `gpt-5-nano`, or any `gpt-5-*` variant for this pipeline.

**Action items (for future):**
- Update `is_reasoning_model` in `pipeline.py` to also detect `gpt-5` prefix
- Always set `request_timeout` and `max_retries` on `ChatOpenAI` to prevent infinite hangs
- Stick to GPT-4.x family for extraction (`gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`)

**Tags:** #bug #critical #model/gpt-5-mini #model/gpt-5-nano #reasoning-models

See also: [[References/Model Comparison]], [[References/API Limits]]

---

## [2026-02-25] ✅ gpt-4o First 50 Datasets — COMPLETED

**Context:** Ran pipeline on first 50 datasets using `gpt-4o` (Tier 1 API, 250k tokens/day).

**Results:**
- Total runs: 53 (3 duplicates from crash/restart: ds_0001, ds_0003, ds_0004)
- Validated (passed): **15 (28.3%)**
- Failed: 38 (71.7%)
- Avg duration: 9.96s per dataset
- Avg LLM itemsets: 10.5 (very conservative — Apriori avg is 175)
- Command: `python pipeline.py --data-dir data/batch50_gpt4o --min-support 3 --max-size 3 --llm-full --llm-model gpt-4o`

**Pipeline fix applied during this run:** Rewrote `llm_extract_full()` to use raw `OpenAI` client instead of LangChain `ChatOpenAI` because `request_timeout=180` on LangChain was silently ignored (requests hung forever). The raw client's `timeout=180.0` at the httpx transport level works correctly.

**Observation:** gpt-4o is more conservative than gpt-4.1-mini (10.5 vs 12.8 avg itemsets) but has a lower pass rate (28% vs 40%). The high Apriori count (avg 175 for these 50 datasets) suggests these particular datasets are harder.

**DB cleanup:** Duplicate rows from crash/restart were cleaned. Rows 1001-1006 (earlier test artifacts) were also deleted.

**Tags:** #experiment #model/gpt-4o #production

---

## [2026-02-25] Updated DB Summary (all models)

| Model | Total | Valid | Pass Rate | API Tier |
|-------|-------|-------|-----------|----------|
| `gpt-4.1-mini` | 500 | 201 | **40.2%** | Tier 2 (2.5M/day) |
| `gpt-4.1-nano` | 500 | 81 | **16.2%** | Tier 2 (2.5M/day) |
| `gpt-4o` | 53 | 15 | **28.3%** | Tier 1 (250k/day) |
| **TOTAL** | **1053** | **297** | **28.2%** | — |

**Tags:** #summary #production

---

## [2026-02-25] 🔧 Artifact Regeneration — validation_reports & db_prepared

**Context:** Discovered that `artifacts/validation_reports/` and `artifacts/db_prepared/` had NO files for gpt-4.1-mini and gpt-4.1-nano (only gpt-4o). The DB had paths stored but files never existed on disk.

**Cause:** Unknown — either the original pipeline version didn't write them, or they were cleaned up before this session. `apriori_outputs` and `extractor_outputs` were intact.

**Fix:** Created `scripts/regenerate_artifacts.py` to re-run `validate_all()` locally on existing apriori + extractor outputs. Zero API calls needed.

**Results:**
- gpt-4.1-mini: 496 validation_reports + 496 db_prepared regenerated (0 errors, 4 skipped — missing extractor outputs)
- gpt-4.1-nano: 497 validation_reports + 497 db_prepared regenerated (0 errors, 3 skipped)
- Cross-checked 5 samples against DB: **all matched perfectly** (error counts identical)

**Scripts deleted after use** (not needed again since artifacts are now complete):
- `scripts/regenerate_artifacts.py` — Re-ran validation locally from existing apriori+extractor outputs
- `scripts/check_gpt4o_results.py` — One-time analysis of gpt-4o result authenticity

**Tags:** #maintenance #artifacts #regeneration

---

## [2026-02-28] 🐛 Empty LLM Validation Bug — FIXED + Retroactive Correction

**Context:** Discovered that when LLM produced 0 itemsets but Apriori found >0, validation incorrectly PASSED (0 errors on 0 itemsets). The check `apr_err == 0 and llm_err == 0` silently passed because the LLM validation loop never ran.

**Root cause:** `validate_source()` iterates per-itemset. Empty input → no errors → passes. No cross-source guard existed.

**Fix (pipeline.py line 150):** Added guard in `validate_all()` — if `len(apriori_sets) > 0 and len(llm_sets) == 0`, inject `llm_empty_output` error into `llm_report['errors']`. Edge case preserved: if both sources find 0, LLM returning 0 is valid.

**Retroactive fix:** `scripts/db_maintenance/fix_empty_llm_validation.py` corrected 71 runs across all 4 data stores (DB, validation_reports, db_prepared, validation logs). Retroactive records tagged with `retroactive_fix: true` for auditability.

**Impact — corrected pass rates:**

| Model | Before Fix | After Fix |
|-------|-----------|-----------|
| `gpt-4.1-mini` (500) | 201 (40.2%) | **135 (27.0%)** |
| `gpt-4.1-nano` (500) | 81 (16.2%) | **79 (15.8%)** |
| `gpt-4o` (100) | 22 (22.0%) | **22 (22.0%)** |
| `gpt-4o-mini` (10) | 1 (10.0%) | **0 (0.0%)** |
| **TOTAL (1110)** | **297 (26.8%)** | **236 (21.3%)** |

**⚠️ IMPORTANT:** Previous pass-rate figures in this file (40.2%, 16.2%, 28.2%) are now SUPERSEDED by these corrected numbers. The 236 validated runs are now trustworthy.

**Action needed:** ✅ DONE — 3-phase training data (SFT-CoT/DPO-Real/GRPO) re-exported with corrected pass rates. See 2026-03-01 entry.

**Tags:** #bug #validation #retroactive-fix

See also: [[Decisions/Empty LLM Validation Bug 2026-02-28]]

---

## [2026-03-01] Corrected DB Summary + o4-mini Results

**Context:** o4-mini batch was added. Combined with retroactive bug fix, the corrected totals:

| Model | Total | Valid | Pass Rate |
|-------|-------|-------|-----------|
| `gpt-4.1-mini` | 500 | **135** | **27.0%** |
| `gpt-4.1-nano` | 500 | **79** | **15.8%** |
| `gpt-4o` | 100 | **22** | **22.0%** |
| `o4-mini` | ~500 | **~214** | **~42.8%** |
| **TOTAL** | **~1600** | **~450** | **~28%** |

**Insight:** o4-mini has the best pass rate (42.8%). These runs are now being used in 3-phase training pipeline (SFT-CoT → DPO-Real → GRPO).

**Training data generated from these runs:**
- SFT-CoT: 348 examples (from 348 datasets with valid Apriori data)
- DPO-Real: 606 pairs (real LLM failures as rejected, Apriori+CoT as chosen)
- GRPO: 314 examples (with Apriori ground truth for reward functions)

**Tags:** #summary #production #3-phase

---

## [2026-03-17] 🔬 Diamond Knowledge — Inference Speedup Pattern

**Source:** Review of 11 Unsloth × Qwen notebooks (diamond phase extraction)

### `FastLanguageModel.for_inference()` — 2× Speedup

All 11 reviewed notebooks call `FastLanguageModel.for_inference(model)` before generation. This enables Unsloth's native inference optimizations (fused kernels, memory-efficient attention). Already used in our training notebook Cell 19 and eval scripts ✅.

**Important:** If building any new pipeline inference scripts (e.g., using the fine-tuned model for extraction), always call `for_inference()` after loading the model + adapter.

### Inference Config from Diamond Review

Standard Unsloth inference uses `temperature=1.5, min_p=0.1` (creative/diverse). This is NOT appropriate for our structured extraction task. Our v3 config (`temperature=0.3, top_k=50, top_p=0.90`) is correct for precision tasks and was validated by the LLM Council.

**Tags:** #diamond-knowledge #inference #pipeline-patterns

See also: [[Evaluation Agent]] [[References/Unsloth Notebook Patterns]]
