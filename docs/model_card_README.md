---
license: apache-2.0
base_model: unsloth/qwen2.5-7b-instruct-bnb-4bit
tags:
- qwen2
- text-generation-inference
- unsloth
- lora
- dpo
- frequent-itemsets
- itemset-mining
- structured-prediction
- fintetuning
language:
- en
metrics:
- f1
- precision
- recall
- exact-match
---

# Qwen2.5-7B Frequent Itemset Extractor

A fine-tuned Qwen2.5-7B-Instruct model that extracts frequent itemsets from
transactional CSV data. Trained with an Apriori-oracle pipeline using
Supervised Fine-Tuning (SFT) with Chain-of-Thought reasoning followed by
Direct Preference Optimization (DPO) on real LLM failure pairs.

## What It Does

Given a CSV of transactions and a minimum support threshold, the model outputs
all frequent itemsets (singles, pairs, triples) as a structured JSON array with
support counts and row evidence:

```json
[{"itemset": ["bread", "milk"], "count": 4, "rows": ["Row 1", "Row 3", "Row 5", "Row 7"]}]
```

The model reasons through the data inside `<think>` tags before producing the
JSON output — the Chain-of-Thought is inspectable and auditable.

## Quick Start

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "OliverSlivka/qwen2.5-7b-itemset-extractor"
)

# Format your CSV as: "Row N: item1, item2, ..."
csv_text = "\n".join([
    "Row 1: bread, milk, eggs",
    "Row 2: bread, butter",
    "Row 3: milk, eggs, bread",
])

messages = [
    {"role": "system", "content": "You are a frequent itemset extractor..."},
    {"role": "user", "content": f"Find all frequent itemsets with minimum support count = 2 in this dataset:\n\n{csv_text}"},
]

inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=2048, temperature=0.3, top_k=50, top_p=0.90)
print(tokenizer.decode(outputs[0]))
```

## Training

| Phase | Method | Examples | Epochs | LR | Details |
|-------|--------|----------|--------|----|---------| 
| 1 | SFT + CoT | 258 | 3 | 1e-4 | Column-grouped concise CoT format (~700 avg tokens) |
| 2 | DPO (real failures) | 606 pairs | 1 | 5e-5 | 4-model failures as rejected, Apriori as chosen |

- **Base model:** Qwen/Qwen2.5-7B-Instruct
- **Adapter:** LoRA (r=32, alpha=64, dropout=0.05)
- **Quantization:** 4-bit NF4 (bitsandbytes), SDPA attention
- **Sequence length:** 2048
- **Optimizer:** paged_adamw_8bit

## Evaluation (v3, 30 held-out datasets)

| Model Variant | Avg P | Avg R | Avg F1 | Hallucination |
|---------------|-------|-------|--------|---------------|
| Base Qwen2.5-7B | — | — | — | — |
| + SFT (Phase 1) | 13.4% | 19.2% | 12.6% | 0.0% |
| + SFT+DPO (Final) | 11.4% | 15.7% | 11.8% | 0.0% |

The model achieves zero hallucination (never invents items not in the input CSV)
and strong JSON parse rates. Performance is strongest on small datasets (≤8 rows,
≤4 cols) where F1 reaches 70–90%. The primary failure mode is JSON formatting
errors on larger datasets rather than incorrect itemsets.

Complete evaluation results and per-dataset breakdowns are in the
[GitHub repository](https://github.com/OliverSlivka/itemsety-qwen-finetuning).

## Intended Use

This is a **research model** developed for a bachelor's thesis at FIS VŠE Praha
(2026). It explores whether LLMs can be fine-tuned for structured combinatorial
reasoning tasks using a deterministic oracle (Apriori) as ground truth.

**Suitable for:** Research on LLM reasoning, structured prediction, itemset mining
**Not suitable for:** Production data mining (use Apriori/FP-Growth directly)

## Repository

All code, datasets, training notebooks, and evaluation scripts:
[https://github.com/OliverSlivka/itemsety-qwen-finetuning](https://github.com/OliverSlivka/itemsety-qwen-finetuning)

## Citation

```bibtex
@software{slivka2026itemset,
  author = {Oliver Slivka},
  title = {Fine-tuning Qwen2.5-7B for Frequent Itemset Extraction},
  year = {2026},
  institution = {FIS VŠE Praha},
  url = {https://github.com/OliverSlivka/itemsety-qwen-finetuning}
}
```
