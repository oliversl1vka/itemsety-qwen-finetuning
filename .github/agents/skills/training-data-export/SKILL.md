---
name: training-data-export
description: Export validated pipeline runs as 3-phase training data (SFT-CoT, DPO-Real, GRPO). Creates ChatML format with concise Chain-of-Thought. Use before fine-tuning to prepare HuggingFace dataset.
---

# Training Data Export (3-Phase Pipeline)

Convert validated pipeline runs into a multi-config HuggingFace dataset for 3-phase fine-tuning.

## Overview

The export pipeline produces three training data types:
1. **SFT-CoT** — Supervised examples with `<think>` reasoning (ground truth from Apriori)
2. **DPO-Real** — Preference pairs using real LLM failures as rejected responses
3. **GRPO** — Reward-based prompts (format ready, training phase currently skipped)

## Current Scripts

| Script | Phase | Output |
|--------|-------|--------|
| `src/training/generate_cot_sft_data.py` | SFT-CoT | `data/sft_cot_v3.json` |
| `src/training/export_real_dpo_data.py` | DPO-Real | `data/dpo_real_v2.json` |
| `src/training/build_hf_dataset_v2.py` | All → HF | `data/hf_dataset_v3/` |
| `src/training/upload_dataset_to_hf.py` | Upload | HuggingFace Hub |

## Quick Start

```bash
# Phase 1: Generate SFT-CoT (v3 concise format with R-shorthand)
python src/training/generate_cot_sft_data.py --db runs.db --output data/sft_cot_v3.json

# Phase 2: Export DPO pairs (real LLM failures as rejected)
python src/training/export_real_dpo_data.py --db runs.db --output data/dpo_real_v2.json

# Phase 3: Build HuggingFace dataset (3 configs: sft/dpo/grpo)
python src/training/build_hf_dataset_v2.py \
  --sft data/sft_cot_v3.json \
  --dpo data/dpo_real_v2.json \
  --output data/hf_dataset_v3

# Phase 4: Upload (versioned — NEVER overwrite old repos)
python src/training/upload_dataset_to_hf.py \
  --dataset-path data/hf_dataset_v3 \
  --repo OliverSlivka/itemset-extraction-v3
```

## SFT-CoT Format (v3 — Concise)

### ChatML Structure
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You extract frequent itemsets from CSV data. Return JSON array."
    },
    {
      "role": "user",
      "content": "Extract frequent itemsets (min_support=3, max_size=3):\n\nCol1,Col2,Col3\nA,B,C\nA,B,D\n..."
    },
    {
      "role": "assistant",
      "content": "<think>\n8R×3C. min_sup=3, max=3.\n\n1-sets: {A}×5 R1-R5, {B}×4 R1-R4...\n2-sets: {A,B}×4 R1-R4...\n</think>\n\n[{\"itemset\":[\"A\",\"B\"],\"count\":4,\"rows\":[\"R1\",\"R2\",\"R3\",\"R4\"],\"support\":0.500}]"
    }
  ]
}
```

### v3 Concise CoT Features
- **R-shorthand**: "R1-R4" instead of "Row 1, Row 2, Row 3, Row 4" (saves tokens)
- **Compact header**: "8R×3C. min_sup=3, max=3." (dimensions + params)
- **No verbose prose** — structured enumeration only
- **Token budget**: ~150 tokens for system prompt (compact)

### CoT Generator (in training_utils.py)
```python
from src.training.training_utils import generate_cot_reasoning_v3

cot = generate_cot_reasoning_v3(
    csv_text=csv_text,
    apriori_itemsets=itemsets,
    min_support=3,
    max_size=3,
)
```

## DPO Preference Pair Format

### Structure
```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Extract frequent itemsets..."}
  ],
  "chosen": [
    {"role": "assistant", "content": "<think>...</think>\n\n[{correct JSON}]"}
  ],
  "rejected": [
    {"role": "assistant", "content": "[{actual wrong output from LLM}]"}
  ]
}
```

### DPO Data Sources
- **Chosen** = Apriori ground truth with generated CoT
- **Rejected** = Real LLM extraction failures (from `extractor_outputs/`)
- Current: 606 preference pairs from real GPT-4.1-mini failures

## GRPO Prompt Format (Ready but Skipped)

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Extract frequent itemsets..."}
  ]
}
```

⚠️ **GRPO phase is currently skipped** — Produced F1=0 in v1 experiments (Council decision).

## HuggingFace Dataset Structure

```
data/hf_dataset_v3/
├── dataset_dict.json
├── sft/              # SFT-CoT config
│   ├── data-00000-of-00001.arrow
│   └── state.json
├── dpo/              # DPO preference config
│   ├── data-00000-of-00001.arrow
│   └── state.json
└── grpo/             # GRPO prompts config
    ├── data-00000-of-00001.arrow
    └── state.json
```

### Inspect Dataset
```python
from datasets import load_from_disk

ds = load_from_disk("data/hf_dataset_v3")
print(ds)                           # Shows all configs
print(f"SFT: {len(ds['sft'])} examples")
print(f"DPO: {len(ds['dpo'])} pairs")
print(f"GRPO: {len(ds['grpo'])} prompts")

# Verify SFT example
print(ds["sft"][0]["messages"][2]["content"][:200])
```

## Quality Criteria

### SFT Inclusion
- ✅ `validation_passed = 1` in runs.db
- ✅ Apriori output has valid itemsets
- ✅ CoT is well-formed (v3 concise format)
- ✅ JSON output matches Apriori ground truth

### DPO Inclusion
- ✅ Real LLM extraction exists (not synthetic)
- ✅ LLM output differs from ground truth (actual failure)
- ✅ Both chosen and rejected are valid JSON
- ✅ Chosen (Apriori) strictly better than rejected (LLM)

## Shared Utilities

Key functions in `src/training/training_utils.py`:
- `get_system_prompt()` — Compact system prompt (~150 tokens)
- `generate_cot_reasoning_v3()` — Concise CoT with R-shorthand
- `load_csv_text()` — Load CSV as raw text for prompt
- `format_ground_truth()` — Format Apriori output as target JSON
- `estimate_token_budget()` — Calculate token budget per example

## Versioning Rules

| Version | SFT Data | DPO Data | CoT Style | Notes |
|---------|----------|----------|-----------|-------|
| v2 | sft_cot_v2.json (348) | dpo_real_v2.json (606) | Verbose | Original |
| **v3** | **sft_cot_v3.json** | dpo_real_v2.json (606) | **Concise** | **Current** |

**NEVER overwrite old HF repos** — each version gets its own repo:
- `OliverSlivka/itemset-extraction-v2`
- `OliverSlivka/itemset-extraction-v3`

## Troubleshooting

### No examples exported
- Check validation pass rate: `sqlite3 runs.db "SELECT AVG(validation_passed) FROM runs"`
- Verify artifacts exist: `ls artifacts/apriori_outputs/ | wc -l`
- Ensure pipeline was run with `--llm-full`

### Low DPO pair count
- Need more pipeline runs with real LLM failures
- DPO requires both Apriori AND LLM outputs per dataset
- Check: `sqlite3 runs.db "SELECT COUNT(*) FROM runs WHERE validation_passed=1 AND llm_itemsets_count>0"`

### CoT quality issues
- Verify `generate_cot_reasoning_v3()` produces R-shorthand
- Check token counts (system prompt should be ~150 tokens)
- Inspect `data/sft_cot_v3.report.json` for generation statistics

### Dataset config mismatch
- Ensure `build_hf_dataset_v2.py` gets correct input files
- Verify all three configs appear in `dataset_dict.json`
- Test loading each config separately
