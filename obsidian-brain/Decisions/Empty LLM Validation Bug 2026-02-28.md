# Empty LLM Validation Bug Fix

**Date:** 2026-02-28  
**Context:** Discovered that LLM producing 0 itemsets was incorrectly marked as validation PASSED when Apriori found >0 itemsets. Zero errors on zero itemsets silently passed the `apr_err == 0 and llm_err == 0` check.  
**Tags:** #decision #bug #validation

---

## Options Considered
1. **Add check in `validate_source` (per-source level):**
   - Pros: Error caught at the lowest level
   - Cons: `validate_source` doesn't know what the *other* source found — it only sees its own itemsets. Adding cross-source awareness would break encapsulation.
2. **Add check in `validate_all` (cross-source level) ✅ CHOSEN:**
   - Pros: Has visibility into both Apriori and LLM results. Natural place for cross-source invariants. Error is injected into `llm_report['errors']` so it flows through the existing error-counting pipeline.
   - Cons: None significant.
3. **Add check at the caller level (in `process_single`):**
   - Pros: Simple
   - Cons: The validation report JSON on disk wouldn't contain the error — only the DB would reflect the failure. Breaks the principle that artifact files are the source of truth.

## Decision
We chose **Option 2 — check in `validate_all`** because it's the only place that sees both sources and ensures the error is recorded in all downstream artifacts (validation report JSON, db_prepared JSON, validation log, and DB).

### Code Change (`pipeline.py`, line 150)
```python
# In validate_all(), after calling validate_source for both:
if len(apriori_sets) > 0 and len(llm_sets) == 0:
    llm_report['errors'].append({
        'type': 'llm_empty_output',
        'source': 'llm',
        'apriori_count': len(apriori_sets),
        'detail': 'LLM extracted 0 itemsets while Apriori found itemsets',
    })
```

### Edge Case Preserved
If **both** Apriori and LLM find 0 itemsets (dataset has no patterns above `min_support`), LLM returning 0 is **correct** — the guard only triggers when Apriori found something.

## Retroactive Fix
Created `scripts/db_maintenance/fix_empty_llm_validation.py` (idempotent, supports `--apply` / dry-run).

**71 affected runs** across 4 models were corrected:
- **`runs.db`**: `validation_passed` 1 → 0
- **`artifacts/validation_reports/*.json`**: Injected `llm_empty_output` error into `llm.errors[]`
- **`artifacts/db_prepared/*.json`**: `validation_passed` true → false, `llm_errors` 0 → 1
- **`logs/validation/*.json`**: `llm_errors` 0 → 1, added `llm_error_sample`

Retroactively fixed records are tagged with `"retroactive_fix": true` and `"fixed_at"` timestamp for auditability. Forward pipeline runs produce the same error structure **without** these audit tags.

### Impact on Pass Rates

| Model | Before Fix | After Fix |
|-------|-----------|-----------|
| `gpt-4.1-mini` | 201 (40.2%) | **135 (27.0%)** |
| `gpt-4.1-nano` | 81 (16.2%) | **79 (15.8%)** |
| `gpt-4o` | 15→22 (22.0%) | **22 (22.0%)** |
| `gpt-4o-mini` | 1 (10%) | **0 (0.0%)** |
| **TOTAL** | **297 (28.2%)** | **236 (21.3%)** |

**Note:** gpt-4o-mini's only "pass" was a vacuous 0-itemset pass — now correctly 0%.

## Consequences
- Pass rates are now **lower but honest**. Training data from these 236 validated runs is trustworthy.
- The previous gpt-4o-mini "1/9 pass" was entirely vacuous — it's now confirmed 0% viable.
- RLHF training data export must be re-run to exclude these 71 now-failed runs.
- All future pipeline runs will correctly reject empty LLM output.

## Output Format Uniformity
- **Forward runs**: Error has keys `type`, `source`, `apriori_count`, `detail`
- **Retroactive fixes**: Same keys **plus** `retroactive_fix: true`, `fixed_at: "<ISO timestamp>"`
- The extra audit keys on retroactive records are intentional and harmless — any code checking `error['type'] == 'llm_empty_output'` works uniformly.

## Review Date
No re-evaluation needed — this is a permanent invariant.

---

**Related:** [[Agents/Pipeline Agent]], [[References/Pipeline Bug 2026-02-08]]
