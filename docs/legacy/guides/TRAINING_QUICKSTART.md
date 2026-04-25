# 🚀 Fine-Tuning Quick Start Guide

**Goal:** Train a small LLM to extract frequent itemsets from CSV data without running Apriori

---

## Prerequisites

✅ Python 3.10+ with virtual environment  
✅ 300 validated runs in `runs.db` (already completed)  
✅ HuggingFace account with API token  
✅ GPU access (Google Colab free tier or HF Jobs)  

---

## Step-by-Step Process

### 1. Install Dependencies (5 minutes)

```powershell
# Activate your virtual environment
.venv\Scripts\Activate.ps1

# Install HuggingFace dependencies
pip install datasets transformers trl peft accelerate bitsandbytes

# Verify installation
python -c "from datasets import Dataset; from transformers import AutoTokenizer; print('✅ Ready')"
```

### 2. Export Training Data (10 minutes)

```powershell
# Export 100 validated examples from runs.db
python export_training_data.py --model gpt_4_1

# Expected output:
#   ✅ Export Complete!
#   Examples: 100
#   Skipped: 0
#   Output: training_data\
```

### 3. Create HuggingFace Dataset (5 minutes)

```powershell
# Convert to HF format with train/val split
python create_hf_dataset.py

# Expected output:
#   ✅ HuggingFace Dataset Created!
#   Train: 90 examples
#   Validation: 10 examples
```

### 4. Inspect Dataset (2 minutes)

```powershell
# Validate format and check statistics
python inspect_training_data.py

# Expected checks:
#   ✅ All examples have 3 messages
#   ✅ Assistant responses are valid JSON
#   ✅ No empty messages
```

### 5. Submit Training Job with Claude

**Option A: Use HuggingFace Skills (Recommended)**

1. Install HF Skills in Claude Code:
   ```
   /plugin install hf-llm-trainer@huggingface-skills
   ```

2. Prompt Claude:
   ```
   Fine-tune Qwen2.5-0.5B-Instruct on my itemset extraction dataset.
   
   Dataset: hf_dataset/
   Task: Supervised fine-tuning (SFT)
   Hardware: t4-medium (should take ~30 min)
   
   Training config:
   - LoRA rank: 16
   - Learning rate: 2e-4
   - Epochs: 3
   - Batch size: 4
   - Max sequence length: 2048
   
   Push final model to Hub: <your-username>/itemset-extractor-qwen
   ```

**Option B: Local Training (Google Colab)**

Upload `hf_dataset/` to Google Drive, then use training script from `FINETUNING_PLAN_COMPREHENSIVE.md`.

---

## Expected Results

After training (~30-60 minutes):

✅ Fine-tuned model on HuggingFace Hub  
✅ Training metrics (loss curves, eval results)  
✅ Ready for inference testing  

### Next Steps

1. Test model on validation set
2. Compare F1 score against Apriori baseline
3. Quantize to GGUF for local deployment
4. Replace pipeline.py with trained model

---

## Troubleshooting

**Issue:** Export script fails  
**Solution:** Check runs.db has 100+ validated runs with `llm_model='gpt_4_1'`

**Issue:** Dataset creation fails  
**Solution:** Verify `extractor_system_prompt.md` exists in root directory

**Issue:** Claude can't access dataset  
**Solution:** Push dataset to HuggingFace Hub first:
```python
from datasets import load_from_disk
dataset = load_from_disk("hf_dataset")
dataset.push_to_hub("your-username/itemset-training-data")
```

---

## Cost Estimate

| Method | Hardware | Time | Cost |
|--------|----------|------|------|
| **HF Jobs (t4-medium)** | NVIDIA T4 | 30-60 min | $0.60 |
| **HF Jobs (a10g-small)** | NVIDIA A10G | 20-40 min | $1.10 |
| **Google Colab Free** | T4 (limited) | 1-2 hours | $0 |

**Recommended:** HF Jobs t4-medium for stability and reproducibility.
