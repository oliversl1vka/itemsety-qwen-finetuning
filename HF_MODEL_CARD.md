---
license: apache-2.0
base_model: unsloth/qwen2.5-7b-instruct-bnb-4bit
tags:
- qwen2.5
- lora
- peft
- itemset-mining
- frequent-pattern-mining
- structured-output
- text-generation-inference
- transformers
- unsloth
language:
- en
pipeline_tag: text-generation
---

# Qwen2.5-7B Itemset Extractor

Fine-tuned Qwen2.5-7B-Instruct for frequent itemset extraction from tabular data.

This is a **LoRA adapter** trained via QLoRA on synthetic CSV datasets (4–26 rows, 3–20 columns) spanning 5 real-world domains (student performance, insurance, mobile reviews, superhero abilities).

## What This Model Does

Given tabular data and a minimum support threshold, the model extracts frequent itemsets (singles, pairs, triples) and outputs them as structured JSON with evidence rows:

```
Input:  Row 1: age:15, medu:4, fedu:2, ...
        Row 2: age:15, medu:2, fedu:1, ...
        ...
        min_support=3

Output: [{"itemset": ["age:15"], "count": 4, "rows": ["Row 1","Row 2","Row 5","Row 7"]}, ...]
```

The model uses Chain-of-Thought reasoning inside `<think>` tags before producing the final JSON output.

## Training

- **Base model:** Qwen2.5-7B-Instruct (4-bit NF4 quantization via QLoRA)
- **Method:** Supervised Fine-Tuning (SFT) with 272 Chain-of-Thought examples (v3 concise format)
- **LoRA config:** r=32, α=64 (α/r=2.0), dropout=0.05, adapter-only save
- **Hardware:** NVIDIA H200 NVL (144 GB VRAM)
- **Training time:** ~4 minutes (SFT phase)
- **Dataset:** [OliverSlivka/itemset-extraction-v3](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v3)

This is the **SFT checkpoint** — Iteration 4's primary result. A separate DPO ablation study (trained on top of this checkpoint) confirmed that DPO is contraproductive for rigid-format extraction tasks (SFT F1=0.13 vs DPO F1=0.12 on the identical primary_v3 evaluation profile).

## Performance

| Metric | Value | Context |
|--------|-------|---------|
| Average F1 | 0.13 | 30 holdout evaluation datasets, primary_v3 profile; uploaded HF adapter verified at 13.07% |
| Best individual F1 | 0.81 | SFT primary_v3, eval_002 8×4 dataset |
| Think rate | 100% | All 120 evaluation runs (4 profiles × 30 datasets) |
| Hallucination rate | 0% | Model produces zero hallucinated evidence rows |
| JSON parse rate | 27% | Primary limitation — see token budget analysis below |

**Limitations:** The model struggles with larger datasets due to the autoregressive token budget constraint — each item in the JSON output consumes tokens. In a separate DPO raw_capture ablation with a larger 8192-token budget, average F1 reached 0.18 and the best individual F1 reached 0.94, but 70% of runs hit the token limit, so that result is not the SFT primary headline. JSON format compliance (27%) is the primary bottleneck; decoding improvements (grammar-constrained decoding) are recommended for future work.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "unsloth/qwen2.5-7b-instruct-bnb-4bit"
adapter_name = "OliverSlivka/qwen2.5-7b-itemset-extractor"

# Load base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, adapter_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Format input
csv_text = """Row 1: age:15, medu:4, fedu:2
Row 2: age:15, medu:2, fedu:1
Row 3: age:18, medu:4, fedu:4"""
prompt = f"Find all frequent itemsets with min_support=2:\n{csv_text}"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.3)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

For the ChatML format used during training (system/user/assistant messages with `<think>` blocks), use the tokenizer's chat template.

## Citation

Slivka, O. (2026). *Fine-tuning malého jazykového modelu pre úlohu hľadania častých itemsetov* [Bachelor's thesis]. Prague University of Economics and Business, Faculty of Informatics and Statistics.

## Links

- **GitHub:** [oliversl1vka/itemsety-qwen-finetuning](https://github.com/oliversl1vka/itemsety-qwen-finetuning)
- **Dataset:** [OliverSlivka/itemset-extraction-v3](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v3)
- **Thesis advisor:** doc. Ing. Tomáš Kliegr, Ph.D.
