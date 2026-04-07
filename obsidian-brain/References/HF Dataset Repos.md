# HF Dataset Repos

Quick reference for all HuggingFace dataset repositories.

**Tags:** #reference #huggingface #datasets

---

## Active Repos

| Version | Repo | SFT (train/val) | DPO | GRPO | Format | Notebook |
|---------|------|-----------------|-----|------|--------|----------|
| **v3** (current) | [itemset-extraction-v3](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v3) | 245/27 | 546/60 | 245/27 | Concise spaced `R1, R10` | v3.11 |
| v2 (frozen) | [itemset-extraction-v2](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2) | 314/34 | 546/60 | 314/34 | Verbose `Row N` | v2 |

## Model Repo

| Repo | Purpose |
|------|---------|
| [qwen2.5-7b-itemset-extractor](https://huggingface.co/OliverSlivka/qwen2.5-7b-itemset-extractor) | Production model (merged weights) |

## Versioning Rule

⚠️ **NEVER overwrite a previous version repo.** Each new version → new repo.

Pattern: `OliverSlivka/itemset-extraction-v{N}`

See [[Decisions/HF Dataset Versioning 2026-03-18]] for full rationale.

## Local Source Files

| File | Version | Examples | Format |
|------|---------|----------|--------|
| `data/sft_cot_v2.json` | v2 | 348 | Verbose `Row N` |
| `data/sft_cot_v3.json` | v3 | 272 | Concise spaced R-refs |
| `data/dpo_real_v2.json` | Shared | 606 | Real LLM failures |

## Useful Commands

```bash
# Load v3 from Hub
python -c "from datasets import load_dataset; ds = load_dataset('OliverSlivka/itemset-extraction-v3', 'sft'); print(ds)"

# Load v2 from Hub
python -c "from datasets import load_dataset; ds = load_dataset('OliverSlivka/itemset-extraction-v2', 'sft'); print(ds)"

# Push new version (template)
python scripts/push_versioned_datasets.py
```
