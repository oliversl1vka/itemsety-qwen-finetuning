# Server Training - Quick Steps

## ✅ Dataset Fixed
- **Problem**: Arrow files were corrupted (Not an Arrow file error)
- **Solution**: Regenerated with `create_hf_dataset.py` - now **88 train / 10 validation**
- **Status**: Local `hf_dataset_enhanced/` contains valid Arrow files

## 🚀 Training Options

### Option A: Train Locally from Disk (RECOMMENDED)
**Why**: Bypasses Hub caching issues, uses valid local files

1. **Upload to Server**:
   ```bash
   # Via SCP (from local machine)
   scp -r hf_dataset_enhanced/ user@server:/path/to/notebook/
   
   # Or via Git (if you push to GitHub)
   git add hf_dataset_enhanced/
   git commit -m "Add regenerated dataset"
   git push
   # Then on server: git pull
   ```

2. **Open Jupyter Notebook**: `qwen_finetuning_server.ipynb`

3. **Run Cells Sequentially**:
   - Cell 1-2: System check + package install
   - Cell 3: Verify `./hf_dataset_enhanced` exists ✅
   - Cell 4: **Load with `load_from_disk`** (already configured!)
   - Cell 5-6: Load 4-bit quantized model + LoRA
   - Cell 7-8: Training config (BFloat16, paged_adamw_8bit)
   - Cell 9+: Start training

4. **Expected Output**:
   ```
   Train examples: 88
   Validation examples: 10
   ✅ Model loaded successfully with 4-bit quantization!
   Total parameters: 494.0M
   Trainable parameters: 8.65M (1.75%)
   ```

### Option B: Fix Hub and Load from There
**Why**: If you want to use `load_dataset()` instead

1. **Force Hub Refresh**:
   ```bash
   # Delete and recreate Hub repo
   huggingface-cli delete-repo OliverSlivka/itemsety-real-training --type dataset
   python -c "from huggingface_hub import HfApi; HfApi().create_repo('OliverSlivka/itemsety-real-training', repo_type='dataset', private=True)"
   python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='hf_dataset_enhanced', repo_id='OliverSlivka/itemsety-real-training', repo_type='dataset')"
   ```

2. **Test Loading**:
   ```python
   from datasets import load_dataset
   ds = load_dataset('OliverSlivka/itemsety-real-training', download_mode='force_redownload')
   print(len(ds['train']))  # Should show 88
   ```

3. **Update Notebook**:
   - Change Cell 4 from `load_from_disk("./hf_dataset_enhanced")` to `load_dataset('OliverSlivka/itemsety-real-training')`

## 🔍 Troubleshooting

### TensorFlow Conflicts
If you see Keras 3 errors:
```python
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
```

### Out of Memory
Reduce batch size in training args:
```python
per_device_train_batch_size=2  # Instead of 4
gradient_accumulation_steps=8  # Instead of 4
```

### LoRA Not Applied
Check output of `model.print_trainable_parameters()`:
- Should show ~8.65M trainable (1.75%)
- If 0% or 100%, LoRA config failed

## 📊 Expected Training Time
- **88 examples** × **3 epochs** = 264 total steps
- **4-bit quantization** + **358GB RAM server** → ~10-20 minutes
- Eval every 50 steps (2 checkpoints)

## ✅ Success Criteria
1. Training loss decreases (starts ~2.5, target <1.0)
2. Validation loss follows train loss
3. No OOM errors
4. Final model saved to `./qwen_itemset_model_enhanced/`

## 🎯 Next Steps After Training
1. **Test model locally**: Load adapter and test on validation examples
2. **Export for inference**: Merge LoRA weights if needed
3. **Evaluate**: Compare to GPT-4 baseline on precision/recall

---
**Recommendation**: Use **Option A** (local disk loading) - Hub is caching old files and `load_from_disk` works perfectly now.
