# SFT Training (Phase 1)

Supervised Fine-Tuning teaches the model the output format and reasoning process. The model learns to produce structured `<think>` reasoning followed by a JSON array of itemsets, using the Apriori algorithm's output as ground truth.

## Training Data Format

Each example follows the ChatML message format:

```json
{
  "messages": [
    {"role": "system", "content": "[compact system prompt]"},
    {"role": "user", "content": "Find all frequent itemsets with min_support=3...\n[CSV data]"},
    {"role": "assistant", "content": "<think>\n[CoT reasoning]\n</think>\n[{\"itemset\": [...], ...}]"}
  ]
}
```

The `train_on_responses_only` helper from Unsloth ensures that only the assistant turn (reasoning + JSON) contributes to the training loss. System and user messages are masked.

## System Prompt

The training system prompt is deliberately compact (~150 tokens), optimized for a 7B model. It is also the prompt used when evaluating the fine-tuned model. It is not the baseline prompt used by the earlier commercial GPT pipeline runs; those use the legacy API baseline prompt in `extractor_system_prompt.md`.

```
You are a frequent itemset extractor. Given CSV transaction data and a
minimum support count, identify all itemsets whose items co-occur in at
least that many rows.

Rules:
1. Scan single items, pairs, and triples (up to size 3)
2. Count = number of distinct rows containing ALL items in the itemset
3. Only report itemsets with count >= min_support
4. Canonicalize items: lowercase, trimmed, sorted alphabetically
5. Row references: "Row N" format, 1-based indexing

Think step by step inside <think> tags, then output ONLY a JSON array:
[{"itemset": ["item1", "item2"], "count": N, "rows": ["Row 1", "Row 3"]}]
```

Source: `src/training/training_utils.py:22-34`

## CoT Format v3 (Column-Grouped)

The chain-of-thought reasoning uses a column-grouped scanning strategy introduced in v3, replacing the row-by-row format from v2. This change reduced token usage by approximately 40% and eliminated repetition loops during inference.

**Structure:**

```
Dataset: 7 rows, 4 cols. min_support=3.

SINGLES SCAN
age:young=4(R1,R2,R5,R7)✓  age:old=3(R3,R4,R6)✓
income:high=3(R1,R3,R5)✓  income:low=4(R2,R4,R6,R7)✓
...

FREQUENT SINGLES: [8] items: age:young, age:old, income:high, ...

PAIRS SCAN
(age:young,income:low)=3(R2,R5,R7)✓
(age:old,income:high)=2✗
...

FREQUENT PAIRS: [3] pairs

TRIPLES SCAN
(age:young,income:low,married:yes)=3(R2,R5,R7)✓
...

RESULT SUMMARY: 8 singles + 3 pairs + 1 triple = 12 itemsets
```

**Key design features:**

- **Column-grouped scanning** groups all occurrences of each item together (e.g., all `age:young` rows before `age:old`), preventing repetition loops that occurred with row-by-row scanning. See [ADR-012](../decisions/adr-012-column-grouped-cot.md).
- **RESULT SUMMARY** termination signal provides a clear stopping point (`N singles + M pairs + K triples = TOTAL`), preventing the model from continuing to generate reasoning indefinitely.
- **Compact notation**: `item=count(R1,R2,R3)✓` or `item=count✗` packs maximum information per token.

Source: `src/training/training_utils.py:60-191`

## Hyperparameters

All values from `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 2 (v3 configuration):

| Parameter | Value | v2 Value | Change Rationale |
|-----------|-------|----------|------------------|
| Base model | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | same | -- |
| Quantization | 4-bit NF4, bfloat16 compute | same | -- |
| LoRA rank (r) | 32 | 64 | Reduce overfitting on 272 examples |
| LoRA alpha | 64 | 16 | Maintain alpha/r=2.0 ratio |
| LoRA dropout | 0.05 | 0 | Added regularization |
| Target modules | q, k, v, o, gate, up, down | same | All attention + MLP projections |
| Learning rate | 1e-4 | 2e-4 | Prevent overfitting |
| Batch size | 2 (per device) | same | -- |
| Gradient accumulation | 4 | same | Effective batch = 8 |
| Epochs | 3 | 2 | More passes on small dataset |
| Warmup ratio | 0.10 | 0.05 | Stabilize early training |
| Weight decay | 0.01 | 0 | L2 regularization |
| Max sequence length | 4096 | 4096 | Initially reduced to 2048 in v3, restored in v3.7 to avoid truncating CoT+JSON targets |
| Optimizer | AdamW | same | TRL default |
| Scheduler | Linear warmup + cosine | same | TRL default |
| Gradient checkpointing | "unsloth" | same | Memory optimization |
| Random seed | 42 | same | Reproducible |

See [ADR-013](../decisions/adr-013-sft-hyperparams.md) for the rationale behind each change.

## Framework

- **Unsloth** `FastLanguageModel` for model loading (2x speed, 70% less VRAM via custom CUDA kernels)
- **TRL** `SFTTrainer` for the training loop
- **`train_on_responses_only`** masks loss on system/user turns
- **BitsAndBytes** 4-bit NF4 quantization

## Save Strategy

The SFT checkpoint is saved as a LoRA adapter only (~65 MB), not merged into the base model. This is necessary because merging LoRA into a 4-bit NF4 base (`merged_4bit_forced`) produces corrupted weights. See [ADR-020](../decisions/adr-020-adapter-only-push.md).

The SFT adapter becomes the starting point for Phase 2 (DPO).
