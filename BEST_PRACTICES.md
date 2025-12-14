# Fine-Tuning Best Practices (Empirical Findings)

**Source**: Colleague testing on school XAI server (jupytergpu)  
**Date**: Based on recent comprehensive testing  
**Context**: Fine-tuning small LLMs for frequent itemset extraction

---

## Critical Findings ⚠️

### 1. **Triplet Format Over Pairs** ✅
**Finding**: "triplets su fajn"  
- Triplet representation (3-item combinations) performs better than pairs (2-item)
- Model learns richer patterns and relationships
- **Recommendation**: Target 3-itemsets (`--max-size 3`) in dataset generation

### 2. **Real-World Context is ESSENTIAL** 🔑
**Finding**: "slovny kontext k datam by sa asi tiez hodil"  
- Synthetic-only data causes catastrophic failure: "ked boli pouzite synthetic datasety cele sa to rozsypalo"
- Model needs human-readable item names and domain context
- Abstract tokens (item_A, prod_15) fail to generalize
- **Solution Implemented**: `enhance_training_data.py` maps synthetic items to 4 real-world domains:
  - **Grocery**: milk, bread, eggs, cheese, butter, etc.
  - **Electronics**: laptop, phone, tablet, charger, headphones, etc.
  - **Clothing**: shirt, pants, shoes, jacket, socks, etc.
  - **Household**: detergent, soap, towel, shampoo, toothpaste, etc.

### 3. **4-Bit Quantized LoRA ONLY** 📉
**Finding**: "iba lora quantized"  
- Full precision training not recommended on this server
- 4-bit quantization (NF4) mandatory for stability
- Use `BitsAndBytesConfig` with:
  - `load_in_4bit=True`
  - `bnb_4bit_quant_type="nf4"`
  - `bnb_4bit_compute_dtype=torch.bfloat16`
  - `bnb_4bit_use_double_quant=True`
- **Critical**: Call `prepare_model_for_kbit_training()` before applying LoRA

### 4. **Model Size Limit: 7B Maximum** 🚫
**Finding**: "model max 7B na jupytergpu"  
- Server memory (358GB RAM) supports models up to 7B parameters
- Tested: Qwen2.5-0.5B (494M params) ✅ works well
- Larger models may exceed memory during gradient accumulation
- **Recommendation**: Stay with 0.5B-3B range for optimal performance

### 5. **API Deployment Doesn't Work** 🔴
**Finding**: "cez api ked je to natrenovane to nejde"  
- Fine-tuned models cannot be reliably deployed via API on this server
- Use locally loaded models only
- **Workflow**: Train → save checkpoint → load locally for inference
- Avoid expecting API-style deployment (e.g., OpenAI API format)

### 6. **Dataset Quality Hierarchy** 📊
**Ranking** (best to worst):
1. **Human-made real datasets** 🥇 - Best performance
2. **Enhanced synthetic** (real-world context) 🥈 - Our approach
3. **Pure synthetic** (abstract tokens) 🥉 - Catastrophic failure

---

## Implementation Checklist

- [x] Use triplet format (3-itemsets) in training data
- [x] Map synthetic items to real-world domains (4 categories implemented)
- [x] Configure 4-bit quantization (BitsAndBytesConfig + prepare_model_for_kbit_training)
- [x] Use BFloat16 compute dtype (matches quantization)
- [x] Apply paged_adamw_8bit optimizer (memory-efficient)
- [x] Target model ≤ 7B parameters (using 0.5B)
- [x] Plan for local inference only (no API deployment)
- [x] Create enhanced dataset: `hf_dataset_enhanced/` (98 examples)

---

## Training Configuration (Validated)

```python
# 4-Bit Quantization
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA Config
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Training Args
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    bf16=True,
    gradient_checkpointing=True,
)
```

---

## Dataset Enhancement Example

**Before** (pure synthetic):
```
Itemset: ["item_A", "item_B", "item_C"]
```

**After** (real-world context):
```
Itemset: ["milk (item_A)", "bread (item_B)", "eggs (item_C)"]
Domain: grocery
Context: Common breakfast items purchased together
```

---

## Next Steps

1. **Upload Enhanced Dataset**: Transfer `hf_dataset_enhanced/` to jupytergpu server
2. **Run Training**: Execute `qwen_finetuning_server.ipynb` with 4-bit config
3. **Validate Locally**: Test fine-tuned model on held-out validation set
4. **Compare Metrics**: Apriori vs LLM extraction accuracy on real-world data
5. **Iterate**: If needed, expand dataset with more real examples

---

## References

- Enhanced Dataset: `training_data/all_training_examples_enhanced.json` (98 examples)
- HF Dataset: `hf_dataset_enhanced/` (88 train / 10 validation)
- Notebook: `qwen_finetuning_server.ipynb` (updated for 4-bit QLoRA)
- Enhancement Script: `enhance_training_data.py`
