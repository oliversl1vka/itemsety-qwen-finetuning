# Server Training - Quick Steps

## ✅ Dataset Fixed & Ready
- **Problem**: Arrow files were corrupted (Not an Arrow file error)
- **Solution**: Regenerated with `create_hf_dataset.py` - now **88 train / 10 validation**
- **Status**: ✅ Committed to GitHub (779c85a + 0cc31bd)
- **GitHub Repo**: https://github.com/oliversl1vka/itemsety-qwen-finetuning

## 🚀 Training on XAI Server

### Step 1: Clone Repository on Server
```bash
# SSH to server
ssh your_username@jupytergpu

# Clone repo (use HTTPS or SSH)
git clone https://github.com/oliversl1vka/itemsety-qwen-finetuning.git
cd itemsety-qwen-finetuning

# Verify dataset is present
ls -lh hf_dataset_enhanced/
python test_dataset_loading.py
```


2. **Open Jupyter Notebook**: `qwen_finetuning_server.ipynb`

3. **Run Cells Sequentially**:
   - **Cell 1**: System resources check
   - **Cell 2**: Install packages (`pip install datasets transformers trl peft accelerate bitsandbytes`)
   - **Cell 3**: Verify `./hf_dataset_enhanced` exists ✅
   - **Cell 4**: Load dataset with `load_from_disk('./hf_dataset_enhanced')` 
     - Expected: **88 train / 10 validation** ✅
   - **Cell 5**: Load 4-bit quantized Qwen2.5-0.5B model
   - **Cell 6**: Configure LoRA (r=16, alpha=32)
   - **Cell 7-8**: Training config (BFloat16, paged_adamw_8bit, 3 epochs)
   - **Cell 9+**: Start training with SFTTrainer

4. **Expected Output**:
   ```
   Train examples: 88
   Validation examples: 10
   Columns: ['messages', 'metadata']
   ✅ Model loaded successfully with 4-bit quantization!
   Total parameters: 494.0M
   Trainable parameters: 8.65M (1.75%)
   ```

### Step 2: Monitor Training
- Training should take **~10-20 minutes** (3 epochs × 88 examples with 4-bit)
- Watch for:
  - Loss decreasing (start ~2.5 → target <1.0)
  - No OOM errors
  - Eval checkpoints at steps 50, 100

### Step 3: After Training
```python
# Test the trained model
from peft import PeftModel, AutoModelForCausalLM
from transformers import AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(model, "./qwen_itemset_model_enhanced")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Test inference
test_prompt = "Extract frequent itemsets from: milk, bread, eggs, milk..."
inputs = tokenizer(test_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

## 🚫 HuggingFace Hub Issue (FYI)
**Problem**: `load_dataset('OliverSlivka/itemsety-real-training')` still shows 1 row despite fresh upload  
**Cause**: Hub caching/processing delays with Arrow files  
**Solution**: Use `load_from_disk('./hf_dataset_enhanced')` instead (already in notebook) ✅

To fix Hub later (optional):
```bash
# Force Hub to reprocess
huggingface-cli delete-repo OliverSlivka/itemsety-real-training --type dataset
huggingface-cli create-repo OliverSlivka/itemsety-real-training --type dataset --private
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='hf_dataset_enhanced', repo_id='OliverSlivka/itemsety-real-training', repo_type='dataset')"

# Test with force redownload
python -c "from datasets import load_dataset; ds = load_dataset('OliverSlivka/itemsety-real-training', download_mode='force_redownload'); print(len(ds['train']))"
```

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
