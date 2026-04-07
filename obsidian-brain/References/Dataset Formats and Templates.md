# Dataset Formats and Templates

**Date:** 2026-03-18  
**Source:** HuggingFace Skills `hugging-face-datasets`, `training_methods.md` + project data  
**Tags:** #reference #dataset #format #chatml #dpo #sft

---

## Our Dataset Configs

Our HuggingFace dataset (`itemset-extraction-v3`) has 3 configs:

| Config | Format | Count | Purpose |
|--------|--------|-------|---------|
| `sft` | ChatML messages | 348 (v2) / varies (v3) | Supervised fine-tuning with CoT |
| `dpo` | Preference pairs | 606 | DPO alignment training |
| `grpo` | Prompts only | varies | GRPO reward-based training (currently unused) |

---

## SFT Format (ChatML with CoT)

### Structure
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a frequent‑itemset extraction engine. ..."
    },
    {
      "role": "user",
      "content": "Dataset (5 rows × 3 cols, min_support=3, max_size=3):\ncol_A,col_B,col_C\nApple,Banana,Cherry\n..."
    },
    {
      "role": "assistant",
      "content": "<think>\nR1-R4 have {Apple, Banana} → count=4, support=4/5=0.800\n...</think>\n[{\"itemset\":[\"Apple\",\"Banana\"],\"count\":4,\"rows\":[\"R1\",\"R2\",\"R3\",\"R4\"],\"support\":0.800}]"
    }
  ]
}
```

### Key Design Decisions (v3 Concise Format)
- **System prompt:** ~150 tokens, compact, defines exact output schema
- **CoT in `<think>` tags:** Concise R-shorthand format (e.g., "R1-R4" not "Row 1, Row 2, Row 3, Row 4")
- **Output:** Strict JSON array after `</think>` tag
- **No preamble/postamble:** Assistant message starts with `<think>` and ends with `]`

### SFT Generation Script
```bash
python src/training/generate_cot_sft_data.py --db runs.db --output data/sft_cot_v3.json
```

---

## DPO Format (Preference Pairs)

### Structure
```json
{
  "prompt": [
    {"role": "system", "content": "You are a frequent‑itemset extraction engine. ..."},
    {"role": "user", "content": "Dataset (10 rows × 5 cols, ...):\n..."}
  ],
  "chosen": [
    {"role": "assistant", "content": "<think>...</think>\n[{\"itemset\":...}]"}
  ],
  "rejected": [
    {"role": "assistant", "content": "[{\"itemset\":[\"Wrong\",\"Items\"],\"count\":99,...}]"}
  ]
}
```

### Key Design Decisions
- **Chosen:** Apriori ground truth with generated CoT reasoning
- **Rejected:** Real LLM failures (GPT extraction errors, not synthetic)
- **606 pairs** from runs where LLM output ≠ Apriori output
- Real failures include: wrong counts, hallucinated items, missing itemsets, format errors

### DPO Generation Script
```bash
python src/training/export_real_dpo_data.py --db runs.db --output data/dpo_real_v2.json
```

---

## GRPO Format (Prompts Only)

### Structure
```json
{
  "prompt": [
    {"role": "system", "content": "You are a frequent‑itemset extraction engine. ..."},
    {"role": "user", "content": "Dataset (8 rows × 4 cols, ...):\n..."}
  ]
}
```

### Status
⚠️ GRPO is currently **skipped** — the format exists for future use.

---

## Dataset Inspection

### Verify Format Compatibility
Before training, verify your dataset matches the expected format:

```python
from datasets import load_from_disk

ds = load_from_disk("data/hf_dataset_v3")

# Check SFT config
sft = ds["sft"]
print(f"SFT examples: {len(sft)}")
print(f"Columns: {sft.column_names}")
# Expected: ['messages']

# Check first example structure
example = sft[0]
assert "messages" in example
assert len(example["messages"]) == 3  # system, user, assistant
assert example["messages"][0]["role"] == "system"
assert example["messages"][2]["role"] == "assistant"
assert "<think>" in example["messages"][2]["content"]

# Check DPO config
dpo = ds["dpo"]
print(f"DPO pairs: {len(dpo)}")
# Expected columns: ['prompt', 'chosen', 'rejected']
```

### TRL Compatibility Check

TRL trainers expect specific column names:

| Trainer | Required Columns | Optional |
|---------|-----------------|----------|
| SFTTrainer | `messages` OR `text` | - |
| DPOTrainer | `prompt`, `chosen`, `rejected` | `system` |
| GRPOTrainer | `prompt` | - |

### Dataset Inspector Script (from HF Skills)
```python
def check_sft_compatibility(columns):
    """Check if dataset works for SFT training."""
    return {
        "ready": "messages" in columns or "text" in columns,
        "format": "messages" if "messages" in columns else "text",
    }

def check_dpo_compatibility(columns):
    """Check if dataset works for DPO training."""
    has_all = all(c in columns for c in ["prompt", "chosen", "rejected"])
    return {
        "ready": has_all,
        "missing": [c for c in ["prompt", "chosen", "rejected"] if c not in columns],
    }
```

---

## Token Budget Calculation

For our itemset extraction task, sequence lengths vary by dataset size:

```python
def estimate_tokens(rows, cols, avg_item_length=8):
    """Estimate total tokens for a dataset prompt + response."""
    # System prompt: ~150 tokens (fixed)
    system_tokens = 150
    
    # User message: CSV header + data rows
    csv_tokens = cols * avg_item_length  # header
    csv_tokens += rows * cols * avg_item_length  # data
    
    # Assistant response: CoT + JSON
    # CoT grows with itemset count; rough estimate
    cot_tokens = rows * cols * 2  # reasoning
    json_tokens = rows * cols * 5  # JSON output
    
    total = system_tokens + csv_tokens + cot_tokens + json_tokens
    return total
```

### Recommended max_seq_length by dataset size
| Dataset | Rows × Cols | Estimated Tokens | Recommended |
|---------|------------|------------------|-------------|
| Small | 5×3 | ~300 | 512 |
| Medium | 10×8 | ~1,200 | 2048 |
| Large | 25×12 | ~4,000 | 4096 |

---

## Building HuggingFace Dataset

### From SFT + DPO JSON files
```bash
python src/training/build_hf_dataset_v2.py \
  --sft data/sft_cot_v3.json \
  --dpo data/dpo_real_v2.json \
  --output data/hf_dataset_v3
```

### Output Structure
```
data/hf_dataset_v3/
├── dataset_dict.json
├── sft/
│   ├── data-00000-of-00001.arrow
│   └── state.json
├── dpo/
│   ├── data-00000-of-00001.arrow
│   └── state.json
└── grpo/
    ├── data-00000-of-00001.arrow
    └── state.json
```

### Upload to Hub
```bash
python src/training/upload_dataset_to_hf.py \
  --dataset-path data/hf_dataset_v3 \
  --repo OliverSlivka/itemset-extraction-v3
```

⚠️ **Never overwrite** — each version gets its own repo. See [[HuggingFace Hub Operations]].

---

## Chat Template Reference

### Qwen2.5 ChatML Format
```
<|im_start|>system
You are a frequent‑itemset extraction engine.<|im_end|>
<|im_start|>user
Dataset (5 rows × 3 cols, min_support=3, max_size=3):
col_A,col_B,col_C
...<|im_end|>
<|im_start|>assistant
<think>
...reasoning...
</think>
[{"itemset": [...], ...}]<|im_end|>
```

### Apply Template in Code
```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False  # False for training, True for inference
)
```

---

## Quality Criteria

### Include in SFT if:
- ✅ `validation_passed = 1` in runs.db
- ✅ `apriori_itemsets_count >= 5` (enough signal)
- ✅ All 13 validation invariants pass
- ✅ Valid JSON output from Apriori

### Include in DPO if:
- ✅ Run has both Apriori output AND LLM output
- ✅ LLM output differs from Apriori (provides contrast)
- ✅ Apriori validation passed (chosen is correct)

### Exclude if:
- ❌ Empty LLM response (no meaningful negative)
- ❌ Apriori validation failed (chosen would be wrong)
- ❌ Dataset has < 5 rows (too trivial)

---

## See Also

- [[Training Methods Guide]] — How each format is used in training
- [[Training Agent]] — Data generation history and decisions
- [[HF Dataset Repos]] — Versioned dataset registry
- [[HuggingFace Hub Operations]] — Upload procedures
