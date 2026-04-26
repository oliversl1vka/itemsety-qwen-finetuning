# Prompt Templates

All prompts used throughout the project, shown verbatim.

## Training System Prompt

Used in SFT and DPO training data. Source: `src/training/training_utils.py:22-34`

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

**Token count:** ~150 tokens. Deliberately compact for a 7B model.

## Training User Message

Source: `src/training/training_utils.py:37-50`, `generate_cot_sft_data.py`

```
Find all frequent itemsets with minimum support count = {min_support}
in this dataset:

Row 1: item_a, item_b, item_c
Row 2: item_a, item_d
Row 3: item_b, item_c, item_d
...
```

CSV data is formatted as `Row N: item1, item2, ...` with 1-based indexing, deduplicated items per row, non-empty cells only.

## Pipeline LLM Extraction Prompt

Used when running `pipeline.py --llm-full`. Source: `pipeline.py:494-500`

```
{system_prompt}
Min support count: {min_support}
Transactions (rows {start+1}-{end}), each key is the absolute row label --
use these EXACT "Row N" labels in your evidence_rows:
{chunk_dict}
Return ONLY JSON array: [{"itemset":[...], "count":n, "evidence_rows":["Row N",...]}]
with count >= {min_support}.
```

This prompt uses `extractor_system_prompt.md` (the longer, 317-line system prompt) rather than the compact training prompt.

## Evaluation User Message

Used in `eval_finetuned_model.py`. Source: `eval_finetuned_model.py:324-329`

```
Find all frequent itemsets with minimum support count = {min_support}
in this dataset:

{csv_text}
```

## CoT Format v3 (Column-Grouped)

Example of the assistant's response format during training:

```
<think>
Dataset: 7 rows, 4 cols. min_support=3.

SINGLES SCAN
age:young=4(R1,R2,R5,R7)✓  age:old=3(R3,R4,R6)✓
income:high=3(R1,R3,R5)✓  income:low=4(R2,R4,R6,R7)✓
married:yes=5(R1,R2,R3,R5,R7)✓  married:no=2✗

FREQUENT SINGLES: [5] items: age:old, age:young, income:high, income:low, married:yes

PAIRS SCAN
(age:young,income:low)=3(R2,R5,R7)✓
(age:young,married:yes)=4(R1,R2,R5,R7)✓
(age:old,income:high)=2✗
(income:low,married:yes)=3(R2,R5,R7)✓

FREQUENT PAIRS: [3] pairs

TRIPLES SCAN
(age:young,income:low,married:yes)=3(R2,R5,R7)✓

RESULT SUMMARY: 5 singles + 3 pairs + 1 triple = 9 itemsets
</think>
[
  {"itemset": ["age:young"], "count": 4, "rows": ["Row 1", "Row 2", "Row 5", "Row 7"]},
  {"itemset": ["income:low"], "count": 4, "rows": ["Row 2", "Row 4", "Row 6", "Row 7"]},
  ...
]
```

Source: `src/training/training_utils.py:60-191`
