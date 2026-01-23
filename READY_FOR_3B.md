# ✅ UPDATED - Ready for 3B Model Training

## 🎯 Hlavná zmena

**Všetky scripty teraz používajú Qwen2.5-3B namiesto 0.5B!**

Máš úplnú pravdu - 0.5B je príliš malé:
- ❌ V1 (0.5B): 6.7% success rate
- ✅ V2 (3B): cieľ 80-90% success rate

## 📦 Čo bolo updatované

### 1. run_sft_test.py
```python
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # ✅ Changed from 0.5B
OUTPUT_DIR = "OliverSlivka/qwen2.5-3b-itemset-test"
per_device_train_batch_size=2  # Adjusted for 3B model
gradient_accumulation_steps=8  # Effective batch = 16
push_to_hub=True  # Now pushes to Hub for validation
```

### 2. run_sft_full.py
- Už používal 3B ✅ (no changes needed)

### 3. app_v2.py (Gradio UI)
- **Test mode**: Qwen2.5-3B, 50 examples, ~10-15 min
- **Full mode**: Qwen2.5-3B, 439 examples, ~40-60 min
- Both modes now use same model, just different dataset sizes

### 4. README_SPACE.md
- Updated all references from 0.5B → 3B
- Clarified both modes use 3B model
- Updated timing estimates

### 5. deploy_to_hf_space.ps1
- Now copies `run_sft_full.py` too
- Updated commit message

## 💾 Memory Estimates

### Qwen2.5-3B with 4-bit quantization:
- **Model**: ~2 GB (4-bit quantized)
- **LoRA adapters**: ~0.5 GB
- **Training activations**: ~4-6 GB (batch size 2)
- **Total**: ~7-9 GB

### Zero GPU A10G has 16 GB → **fits comfortably!** ✅

## 🚀 Deployment Commands

### Option 1: Automated (Recommended)
```powershell
cd C:\Users\slivk\Desktop\itemsety_real_training\itemsety
.\deploy_to_hf_space.ps1
```

### Option 2: Manual
```powershell
cd C:\Users\slivk\Desktop\itemsety_real_training
git clone https://huggingface.co/spaces/OliverSlivka/testrun2 hf_space
cd hf_space

# Copy updated files
Copy-Item ..\itemsety\app_v2.py .\app.py -Force
Copy-Item ..\itemsety\run_sft_test.py . -Force
Copy-Item ..\itemsety\run_sft_full.py . -Force
Copy-Item ..\itemsety\requirements.txt . -Force
Copy-Item ..\itemsety\README_SPACE.md .\README.md -Force

# Commit and push
git add .
git commit -m "feat: Add Qwen2.5-3B training (test + full modes)"
git push
```

## 📊 Expected Results

| Mode | Model | Examples | Duration | Output Repo |
|------|-------|----------|----------|-------------|
| Test | 3B | 50 | ~10-15 min | `qwen2.5-3b-itemset-test` |
| Full | 3B | 439 (3 epochs) | ~40-60 min | `qwen2.5-3b-itemset-extractor` |

## 🎯 Training Flow

1. **Deploy** → Space rebuilds (~2-3 min)
2. **Test mode** → Verify 3B works on Zero GPU
3. **Full mode** → Train production model
4. **Evaluate** → Test on 9 real datasets

## ⚠️ Potential Issues

### If 3B doesn't fit in Zero GPU:
**Symptoms**: OOM (Out of Memory) errors

**Solutions**:
1. Reduce batch size to 1: `per_device_train_batch_size=1`
2. Use gradient checkpointing (already enabled ✅)
3. Try paid GPU ($0.60/hour for A10G)

**Probability**: Very low! 4-bit quantization + batch=2 should work.

### If training is too slow:
**Symptoms**: >2 hour timeout

**Solutions**:
1. Reduce max_steps in test mode
2. Use paid GPU for full mode
3. Train 2 epochs instead of 3

## 📝 Key Takeaways

1. ✅ **3B model** namiesto 0.5B (huge improvement expected)
2. ✅ **4-bit quantization** (fits in Zero GPU)
3. ✅ **LoRA** (efficient fine-tuning)
4. ✅ **Test + Full modes** (verify before production)
5. ✅ **Push to Hub** (both modes now push models)

## 🚀 Ready to Deploy!

Všetko je **updatované na 3B model**. Zero GPU by to mal zvládnuť bez problémov!

**Stačí spustiť deployment script a potom sledovať training!** 🎉

---

**Last Updated**: 2026-01-17 (3B model update)  
**Status**: 🟢 READY FOR 3B TRAINING
