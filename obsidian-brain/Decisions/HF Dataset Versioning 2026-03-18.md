# HF Dataset Versioning (2026-03-18)

**Status:** ✅ Implemented  
**Decision Type:** Infrastructure  
**Tags:** #decision #versioning #huggingface #critical

---

## Context

Since 2026-03-01, every dataset rebuild used `scripts/push_versioned_datasets.py` (or `upload_dataset_to_hf.py`) to push to a single HF repo: `OliverSlivka/itemset-extraction-v2`. Each push **overwrote** the previous data, destroying training history.

This meant:
- v2 training data was lost when v3 data was pushed
- No way to reproduce v2 training runs from HF
- No way to compare v2 vs v3 dataset quality side-by-side

## Decision

**Each training data version gets its own FROZEN HuggingFace dataset repo.**

| Version | HF Repo | SFT | Format | Notebook | Status |
|---------|---------|-----|--------|----------|--------|
| v2 | `OliverSlivka/itemset-extraction-v2` | 314/34 | Verbose `Row N` | `training_3phase_2026-03-07_v2.ipynb` | 🔒 Frozen |
| v3 | `OliverSlivka/itemset-extraction-v3` | 245/27 | Concise spaced `R1, R10` | `training_3phase_2026-03-09_v3.ipynb` | 🔒 Frozen |
| v4+ | `OliverSlivka/itemset-extraction-v{N}` | TBD | TBD | TBD | Future |

DPO data (546/60) is shared across versions (same 606 real LLM failure pairs).

## Rules

### ⚠️ CRITICAL: Never Overwrite Previous Version Repos

For ALL future versions:
1. **Create NEW repo:** `OliverSlivka/itemset-extraction-v{N}`
2. Push data + notebook to new repo
3. Update notebook CONFIG `hf_dataset` to point to new repo
4. **Previous repos remain FROZEN** — never push to them again
5. Use `scripts/push_versioned_datasets.py` as template for push logic

### Local Data Layout

```
data/
├── sft_cot_v2.json          # v2 source (348 examples, verbose Row N)
├── sft_cot_v3.json          # v3 source (272 examples, concise spaced R-refs)
├── dpo_real_v2.json          # Shared across versions (606 pairs)
├── hf_dataset_v2/            # Local v2 dataset (for rebuilds)
│   ├── sft/                  # 314 train / 34 val
│   ├── dpo/                  # 546 train / 60 val
│   └── grpo/                 # 314 train / 34 val
└── hf_dataset_v3/            # Local v3 dataset
    ├── sft/                  # 245 train / 27 val
    ├── dpo/                  # 546 train / 60 val
    └── grpo/                 # 245 train / 27 val
```

### Deployment Agent Protocol

When deployment-agent runs `/push`:
1. Read `workflow_state.json` to determine current version (e.g., `"dataset_version": "v3"`)
2. Push to `OliverSlivka/itemset-extraction-v{version}` (NOT hardcoded v2)
3. Upload matching versioned notebook
4. Update workflow state with correct URLs

## Alternatives Considered

1. **Git tags on single repo** — Rejected: HF dataset repos don't support branching well for `load_dataset()`
2. **Version suffixes in config names** — Rejected: Complicates `load_dataset()` calls
3. **Keep overwriting** — Rejected: Destroys reproducibility

## Links

- [[Agents/Training Agent]] — v3.11 entry documents this change
- [[Agents/Deployment Agent]] — Must use versioned repos
- [[References/HF Dataset Repos]] — Quick reference for repo URLs
