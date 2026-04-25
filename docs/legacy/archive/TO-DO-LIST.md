# Project TODO List: Frequent Itemsets Pipeline

**Last Updated:** November 22, 2025

Task list with completion status. Codes:
- ✅ **DONE** - Completed and verified
- 🚧 **IN PROGRESS** - Currently being worked on
- ⏸️ **DEFERRED** - Postponed for later
- ❌ **CANCELLED** - No longer needed
- 📋 **TODO** - Not started

---
## 1. Logging Layout Cleanup (Spec 2.1) - ✅ COMPLETED

### 1.1 Create dedicated logging folders ✅
- **Status:** DONE
- Path: `logs/apriori/`, `logs/extractor/`, `logs/validation/`, `logs/db_prepared/`
- Auto-created at pipeline start
- **Completion:** November 22, 2025

### 1.2 Redirect generation-log artifacts ✅
- **Status:** DONE
- All `*_generation_log_*.json` files now stored in respective `logs/` subfolders
- Original output folders contain only run artifacts
- Model prefix added to all log filenames
- **Completion:** November 22, 2025

### 1.3 Logging path resolver utility ✅
- **Status:** DONE
- Implemented `resolve_log_path(kind, stem, hash_prefix, suffix, model_prefix)`
- Standardizes naming & location for all log types
- **Completion:** November 22, 2025

### 1.4 Minimal log content (NEW) ✅
- **Status:** DONE
- Reduced log files from ~250 bytes to ~120 bytes (~52% reduction)
- Removed redundant fields (dataset, hash in filename; config in DB)
- Kept only unique debugging info: timing, counts, errors
- Created `reduce_logs.ps1` for batch reduction
- **Completion:** November 22, 2025

### 1.5 Retrospective log generation (NEW) ✅
- **Status:** DONE
- Created `generate_extractor_logs.py` for missing extractor logs
- Created `generate_all_logs_retrospective.py` for complete log sets
- Successfully generated 400 logs for gpt_5_mini historical runs
- **Completion:** November 22, 2025

---
## 2. Pipeline Range Selection & Defaults (Spec 2.2) - 📋 TODO

### 2.1 Auto-discovery of datasets 📋
- **Status:** TODO
- Default invocation should scan `datasets/` for `ds_*.csv`
- Process all in numeric ID order
- **Priority:** P1 (Current workaround: `--data-dir datasets`)

### 2.2 Range filtering arguments 📋
- **Status:** TODO
- Add CLI flags: `--dataset-start`, `--dataset-end`
- Accept integer (15) or string (ds_0015), inclusive range
- Validation: if start > end → exit code 4
- **Priority:** P1

### 2.3 Backward compatibility ✅
- **Status:** DONE (already preserved)
- Existing `--data` and `--data-dir` work correctly
- **Completion:** Already functional

### 2.4 Selection logging 📋
- **Status:** TODO
- Print: `Selected datasets: [0001, 0002, ...] (N total)` before processing
- **Priority:** P2

### 2.5 Pure helper for filtering 📋
- **Status:** TODO
- Implement `filter_datasets(filenames, start=None, end=None)`
- Unit tests for boundary cases
- **Priority:** P1

---
## 3. Security Hygiene (Spec 2.3) - ⚠️ PARTIAL

### 3.1 Provide `.env.example` ⏸️
- **Status:** DEFERRED
- Current: `azure.env` serves as template
- **Priority:** P2 (low risk with current .gitignore)

### 3.2 Ensure `.env` ignored ✅
- **Status:** DONE
- `.gitignore` includes `azure.env` and credential files
- **Completion:** Already configured

### 3.3 Requirements pruning ⏸️
- **Status:** DEFERRED
- Current `requirements.txt` is minimal (pandas, langchain, matplotlib)
- **Priority:** P3

### 3.4 README credential instructions ✅
- **Status:** DONE
- Security Notice section added with best practices
- Clear setup instructions provided
- **Completion:** November 22, 2025

---
## 4. Visualization Enhancements (Spec 2.4) - ⏸️ DEFERRED

### 4.1 Heatmap of rows vs cols ⏸️
- **Status:** DEFERRED
- `visualization.py` already creates comparison plots
- **Priority:** P3

### 4.2 CDF or boxplot ⏸️
- **Status:** DEFERRED
- Current visualizations sufficient for analysis
- **Priority:** P3

---
## 5. Documentation Updates (Spec 2.5) - ✅ COMPLETED

### 5.1 README dataset range section ✅
- **Status:** DONE
- Comprehensive command-line arguments documented
- Usage examples with all major flags
- **Completion:** November 22, 2025

### 5.2 Logging separation explanation ✅
- **Status:** DONE
- Generation Logs section with structure and examples
- Retrospective generation documented
- **Completion:** November 22, 2025

### 5.3 Testing guidance inclusion ⏸️
- **Status:** DEFERRED
- No formal test suite yet
- **Priority:** P2

---
## 6. Testing & Quality (Spec 3 & 4) - 📋 TODO

### 6.1 Unit tests for dataset filtering 📋
- **Status:** TODO
- Need `tests/test_filter_datasets.py`
- **Priority:** P1 (when range filtering implemented)

### 6.2 Unit tests for logging resolver ⏸️
- **Status:** DEFERRED
- Current implementation working reliably
- **Priority:** P2

### 6.3 Manual run checklist ✅
- **Status:** DONE (documented in README)
- Setup, running, verification steps all documented
- **Completion:** November 22, 2025

---
## 7. Model Tracking & Database (NEW) - ✅ COMPLETED

### 7.1 Model prefix in filenames ✅
- **Status:** DONE
- All artifacts prefixed with model identifier
- Format: `{model}_{type}_{stem}_{hash}.json`
- **Completion:** November 22, 2025

### 7.2 llm_model column in database ✅
- **Status:** DONE
- Auto-migration adds column if missing
- Stored with every run
- **Completion:** November 22, 2025

### 7.3 Database cleanup utility ✅
- **Status:** DONE
- Created `cleanup_database.py` with backup functionality
- Successfully cleaned to 200 records (100 gpt_4_1, 100 gpt_5_mini)
- **Completion:** November 22, 2025

### 7.4 Extractor logging ✅
- **Status:** DONE
- Added `logs/extractor/` with timing and counts
- Symmetry with apriori logging achieved
- **Completion:** November 22, 2025

---
## 8. Current Status Summary (November 22, 2025)

### ✅ COMPLETED Features:
1. **Logging Infrastructure** (100%)
   - Dedicated `logs/` subdirectories for all stages
   - Minimal log content (~52% size reduction)
   - Model prefix in all log filenames
   - Retrospective generation scripts

2. **Model Tracking** (100%)
   - `llm_model` column in database
   - Model prefix in all artifact filenames
   - Auto-migration support

3. **Database Management** (100%)
   - Clean 200-record database
   - Cleanup utility with backup
   - Query examples and tools

4. **Documentation** (100%)
   - Comprehensive README with all CLI options
   - Generation log structure documented
   - Maintenance scripts documented
   - Security best practices

### 📋 PENDING Features:
1. **Dataset Range Filtering** (0%)
   - Auto-discovery on no args
   - `--dataset-start`, `--dataset-end` flags
   - Helper functions

2. **Testing** (0%)
   - Unit tests for filtering logic
   - Unit tests for logging resolver
   - CI/CD pipeline

3. **Optional Enhancements** (deferred)
   - `.env.example` template
   - Additional visualizations
   - Requirements audit

### 🎯 Priority Recommendations:
1. **High Priority:**
   - Range filtering implementation (improves UX)
   - Unit tests for core functions

2. **Medium Priority:**
   - Test documentation
   - CI/CD setup

3. **Low Priority:**
   - Additional visualizations
   - `.env.example` (current setup adequate)

---
## 7. Non-Goals / Explicitly Out of Scope
- Modify `dataset_generation.py` logic (forbidden by spec).
- Introduce Apriori-only execution mode.
- Change existing artifact folder naming conventions.

---
## 8. Task Dependency Ordering (Suggested Execution Sequence)
1. Create logging folders & path helper (1.1, 1.3).
2. Implement redirection logic (1.2).
3. Add dataset auto-discovery + filtering helpers (2.1, 2.5).
4. Wire CLI parameters & validation (2.2, 2.3, 2.4).
5. Create `.env.example` & update `.gitignore` (3.1, 3.2).
6. Requirements audit (3.3).
7. README updates (5.1, 5.2, 3.4).
8. Add unit tests (6.1, 6.2).
9. Optional visualization enhancements (4.1, 4.2).
10. Manual run & checklist confirmation (6.3).

---
## 9. Acceptance Summary Matrix
| Spec Ref | TODO Item | Must Have | Deliverable |
|----------|-----------|-----------|------------|
| 2.1 | Logging folders & redirection | Yes | New `logs/...` dirs + moved JSON logs |
| 2.2 | Range selection CLI | Yes | Updated `pipeline.py` flags & behavior |
| 2.3 | Env template & ignore | Yes | `.env.example`, updated `.gitignore` |
| 2.4 | Visualization extras | Optional | New charts or TODO comments |
| 2.5 | README updates | Yes | Expanded README sections |
| 4 | Tests for filtering/logging | Recommended | Test files + passing run |

---
## 10. Risk & Mitigation Notes
- Risk: Mis-parsing dataset IDs if filename pattern changes. Mitigation: robust regex + unit tests.
- Risk: Accidentally committing secrets. Mitigation: Template + ignore verified.
- Risk: Removing used requirement. Mitigation: static search before deletion.
- Risk: Large log folder growth. Mitigation: future rotation policy (add later TODO).

---
## 11. Future Backlog (Post-Spec Ideas)
- Store derived metrics directly into `runs.db` for multi-run comparative dashboards.
- Add command `--dry-run` to list selected datasets without processing.
- Implement concurrency (parallel Apriori + LLM) with locking/hashing validation.
- Introduce config-driven min_support / max_size sweeps.

---
## 12. Tracking Placeholder
Add status markers when tasks begin/complete; integrate with issue tracker if desired. (Convert each subsection into GitHub Issues.)

---
_End of TO-DO-LIST.md_
