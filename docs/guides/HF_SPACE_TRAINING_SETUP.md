# HuggingFace Space Training Setup Guide

**Space URL:** https://huggingface.co/spaces/OliverSlivka/testrun2  
**Deployment Date:** 2026-02-03  
**Status:** ✅ Deployed (Commit: 9a0325bb)

---

## 🎯 What Was Deployed

### Files Uploaded to Space
1. **app.py** - DPO-enabled Gradio interface
2. **README.md** - Space documentation with comparison table
3. **requirements.txt** - Python dependencies (PyTorch, TRL, PEFT, bitsandbytes)
4. **src/training/run_dpo_training.py** - DPO training script
5. **src/training/run_sft_full.py** - SFT production training
6. **src/training/run_sft_test.py** - SFT test training

### Training Methods Available
- ⭐ **DPO** (Direct Preference Optimization) - Recommended
- **SFT** (Supervised Fine-Tuning) - Baseline

---

## 🚀 Setup Instructions

### Step 1: Access the Space
Go to: https://huggingface.co/spaces/OliverSlivka/testrun2

### Step 2: Wait for Build
The Space needs to install dependencies (~2-3 minutes). You'll see:
- "Building..." status
- Then "Running" when ready

### Step 3: Configure Space Settings

**Important:** Set HF token as Space secret

1. Click **Settings** (top right)
2. Go to **Variables and secrets**
3. Click **New secret**
4. Add:
   - **Name:** `HF_TOKEN`
   - **Value:** `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` (your token)
5. Click **Save**

**Why needed?** Training scripts need to:
- Download RLHF dataset from Hub
- Push trained models back to Hub

### Step 4: Configure GPU

**Option A: Zero GPU (Free, 2h limit)**
1. Go to Settings
2. Under **Hardware**, select **Zero GPU (A10G, 16GB)**
3. Save

**Option B: Persistent GPU (Paid, no limit)**
1. Go to Settings
2. Under **Hardware**, select **L4 (24GB)** or **A10G (16GB)**
3. Confirm billing

**Recommendation for DPO Full:**
- Use persistent GPU (90 min > 2h Zero GPU limit)
- Or split into 2 runs (2 epochs + 1 epoch)

### Step 5: Download RLHF Dataset

The Space needs the dataset locally. Two options:

**Option A: Pre-download in Space (recommended)**

Add this to app.py startup:
```python
import os
from datasets import load_dataset

# Pre-download dataset at startup
if not os.path.exists("data/hf_rlhf_dataset_v1"):
    print("📥 Downloading RLHF dataset...")
    dataset = load_dataset("OliverSlivka/itemset-extraction-rlhf-v1")
    dataset.save_to_disk("data/hf_rlhf_dataset_v1")
    print("✅ Dataset cached locally")
```

**Option B: Download on first run**

The training script will auto-download from Hub using:
```python
--dataset_path OliverSlivka/itemset-extraction-rlhf-v1
```

---

## 🎮 Running Training

### Quick Test (Recommended First)

1. Select **Training Method:** DPO
2. Select **Training Mode:** test
3. Click **Submit**
4. Wait ~15-20 minutes

**What it does:**
- Loads Qwen2.5-3B with 4-bit quantization
- Trains on 100 preference pairs
- 1 epoch
- Saves to `dpo_test_checkpoints/`
- **Does NOT** push to Hub (test only)

### Production Training

1. Select **Training Method:** DPO
2. Select **Training Mode:** full
3. Click **Submit**
4. Wait ~60-90 minutes

**What it does:**
- Loads Qwen2.5-3B with 4-bit quantization
- Trains on 4,399 preference pairs
- 3 epochs
- Saves to `dpo_checkpoints/`
- Pushes to Hub: `OliverSlivka/qwen2.5-3b-itemset-dpo`

### Monitoring Progress

Watch the **Training Log** output for:
```
Loading model: Qwen/Qwen2.5-3B-Instruct
Model loaded: 3.2GB / 16GB GPU
Loading dataset from data/hf_rlhf_dataset_v1
Dataset loaded: 4399 train, 489 val
Training started (3 epochs, batch_size=1, grad_accum=8)

Epoch 1/3:
  Step 50/550 | Loss: 0.234 | Pref Acc: 0.82
  Step 100/550 | Loss: 0.189 | Pref Acc: 0.87
  ...
  
Validation: Loss: 0.145 | Pref Acc: 0.91

Saving checkpoint: dpo_checkpoints/checkpoint-550
Pushing to Hub: OliverSlivka/qwen2.5-3b-itemset-dpo

✅ Training finished successfully!
```

**Key metrics to watch:**
- **Loss:** Should decrease (0.3 → 0.1)
- **Pref Acc** (Preference Accuracy): Should increase (>0.85 is good)
- **GPU Memory:** Should stay under 16GB

---

## 🔧 Troubleshooting

### Issue: Space fails to build
**Symptom:** Stuck on "Building..."
**Solution:**
1. Check logs (click "Logs" tab)
2. Common causes:
   - Missing dependency version
   - Incompatible package versions
3. Fix: Update requirements.txt, redeploy

### Issue: Out of memory (OOM)
**Symptom:** `RuntimeError: CUDA out of memory`
**Solution:**
1. Reduce batch size in run_dpo_training.py:
   ```python
   --per_device_train_batch_size 1  # Already minimal
   --gradient_accumulation_steps 4   # Reduce from 8
   ```
2. Or use smaller model: Qwen2.5-0.5B instead of 3B

### Issue: Zero GPU timeout (2h limit exceeded)
**Symptom:** Training interrupted at ~2 hours
**Solution:**
1. Use persistent GPU (paid)
2. Or reduce epochs: 2 instead of 3
3. Or use test mode multiple times

### Issue: HF_TOKEN not working
**Symptom:** `401 Unauthorized` or `Invalid token`
**Solution:**
1. Verify token in Space secrets (Settings > Variables)
2. Check token has write permissions
3. Regenerate token on HuggingFace: https://huggingface.co/settings/tokens

### Issue: Dataset not found
**Symptom:** `FileNotFoundError: data/hf_rlhf_dataset_v1`
**Solution:**
1. Add dataset pre-download to app.py (see Step 5 above)
2. Or modify script to load from Hub directly:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("OliverSlivka/itemset-extraction-rlhf-v1")
   ```

---

## 📊 Expected Results

### After Test Run
- **Duration:** 15-20 minutes
- **Output:** `dpo_test_checkpoints/` folder
- **Model:** NOT pushed to Hub (local only)
- **Validation:** Verify training loop works

### After Production Run
- **Duration:** 60-90 minutes
- **Output:** `dpo_checkpoints/` folder
- **Model:** Pushed to `OliverSlivka/qwen2.5-3b-itemset-dpo`
- **Metrics:** F1=0.82, Hallucinations=3%, JSON Parse=98%

---

## 🎯 Next Steps After Training

### 1. Verify Model on Hub
```bash
# Check model exists
huggingface-cli repo info OliverSlivka/qwen2.5-3b-itemset-dpo

# Download and test locally
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained("OliverSlivka/qwen2.5-3b-itemset-dpo")
```

### 2. Run Evaluation
```bash
python src/evaluation/eval_finetuned_model.py \
  --model-path OliverSlivka/qwen2.5-3b-itemset-dpo \
  --eval-dir data/eval_datasets \
  --count 50
```

### 3. Compare with SFT Baseline
Run both models on same eval set and compare:
- F1 score
- Hallucination rate
- JSON parse rate
- Inference time

### 4. Deploy to Production
Once validated, update inference endpoints to use DPO model.

---

## 📚 Resources

- **Space URL:** https://huggingface.co/spaces/OliverSlivka/testrun2
- **RLHF Dataset:** https://huggingface.co/datasets/OliverSlivka/itemset-extraction-rlhf-v1
- **SFT Dataset:** https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2
- **DPO Paper:** https://arxiv.org/abs/2305.18290
- **Project GitHub:** https://github.com/oliversl1vka/itemsety-qwen-finetuning

---

**Last Updated:** 2026-02-03 15:45  
**Deployed By:** Oliver Slivka  
**Status:** ✅ Ready for training
