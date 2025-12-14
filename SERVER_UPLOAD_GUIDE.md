# Server Upload Guide (XAI Jupytergpu)

Guide for uploading and running fine-tuning on the school XAI server.

---

## Server Information
- **Host**: jupytergpu (school XAI server)
- **Memory**: 358GB RAM
- **Access**: Jupyter Notebook interface
- **Constraint**: Models ≤ 7B parameters

---

## Step 1: Upload Repository to Server

### Option A: Git Clone (Recommended)
If server has Git access:
```bash
# In Jupyter terminal
cd ~
git clone https://github.com/oliversl1vka/itemsety-qwen-finetuning.git
cd itemsety-qwen-finetuning
```

### Option B: Manual Upload
If Git is unavailable:
1. **Package locally**:
   ```powershell
   # In your local repository
   tar -czf itemsety-upload.tar.gz qwen_finetuning_server.ipynb hf_dataset_enhanced/ enhance_training_data.py BEST_PRACTICES.md FINETUNING_README.md
   ```

2. **Upload via Jupyter**:
   - Open Jupyter interface
   - Navigate to your workspace
   - Use "Upload" button to transfer `itemsety-upload.tar.gz`
   - Extract in terminal: `tar -xzf itemsety-upload.tar.gz`

---

## Step 2: Verify Environment

Open `qwen_finetuning_server.ipynb` and run **Cell 1 (System Check)**:

```python
# Verify Python version (should be 3.8+)
import sys
print(f"Python: {sys.version}")

# Check GPU availability
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Check RAM
import psutil
ram = psutil.virtual_memory()
print(f"RAM: {ram.available / 1e9:.2f} / {ram.total / 1e9:.2f} GB available")
```

**Expected Output**:
- Python ≥ 3.8
- CUDA Available: True
- GPU Memory: > 10GB
- RAM: ~358GB total

---

## Step 3: Install Dependencies

Run **Cell 2 (Install Dependencies)**:

```bash
%%bash
pip install -q transformers datasets trl peft accelerate torch bitsandbytes
```

**⚠️ Critical**: Ensure `bitsandbytes` installs successfully (required for 4-bit quantization).

If installation fails:
```bash
# Try with specific version
pip install bitsandbytes==0.41.1
```

---

## Step 4: Verify Dataset

Check that enhanced dataset exists:

```python
from pathlib import Path

dataset_path = Path("./hf_dataset_enhanced")
assert dataset_path.exists(), "❌ Enhanced dataset not found!"
assert (dataset_path / "train").exists(), "❌ Train split missing!"
assert (dataset_path / "validation").exists(), "❌ Validation split missing!"

print("✅ Enhanced dataset verified!")
print(f"  Train: {dataset_path / 'train'}")
print(f"  Validation: {dataset_path / 'validation'}")
```

---

## Step 5: Run Training

Execute cells **sequentially** in `qwen_finetuning_server.ipynb`:

### Cell Execution Order:
1. ✅ **System Check** (Cell 3) - Verify environment
2. ✅ **Load Dataset** (Cell 7) - Load `hf_dataset_enhanced`
3. ✅ **Inspect Data** (Cell 9) - Preview training examples
4. ✅ **Load Model (4-bit)** (Cell 11) - Load Qwen2.5-0.5B with BitsAndBytesConfig
5. ✅ **Configure LoRA** (Cell 13) - Apply LoRA with `prepare_model_for_kbit_training()`
6. ✅ **Training Args** (Cell 15) - Set up 4-bit optimized training
7. ✅ **Initialize Trainer** (Cell 17) - Create SFTTrainer
8. ✅ **Start Training** (Cell 19) - Run `trainer.train()` (expect ~30-60 min)
9. ✅ **Save Model** (Cell 21) - Save checkpoint
10. ✅ **Test Inference** (Cell 23) - Validate on test examples
11. ✅ **Memory Visualization** (Cell 25) - Plot memory usage

---

## Step 6: Monitor Training

### Expected Behavior:
- **First epoch**: Loss ~1.5-2.0
- **Final epoch**: Loss ~0.5-0.8
- **Training time**: 30-60 minutes (3 epochs, 88 examples)
- **Memory usage**: ~15-20GB GPU, ~10-15GB RAM

### Logs to Monitor:
```
Step  | Loss   | Learning Rate
------|--------|---------------
10    | 1.847  | 2.00e-04
50    | 1.234  | 1.96e-04
100   | 0.891  | 1.80e-04
...
264   | 0.523  | 0.00e-00
```

### If Training Fails:

**Error: CUDA Out of Memory**
```python
# Reduce batch size in Cell 15
per_device_train_batch_size=2  # Was 4
gradient_accumulation_steps=8  # Was 4
```

**Error: bitsandbytes not found**
```bash
pip install bitsandbytes --upgrade
```

**Error: Model too large**
```python
# Already using smallest model (0.5B)
# If still fails, contact server admin
```

---

## Step 7: Save and Download Model

After training completes (Cell 21):

```python
# Model saved to: ./qwen_finetuned_output/checkpoint-XXXX
```

### Download Checkpoint:
1. **Compress checkpoint**:
   ```bash
   tar -czf qwen_finetuned_checkpoint.tar.gz qwen_finetuned_output/checkpoint-*
   ```

2. **Download via Jupyter**:
   - Navigate to file browser
   - Right-click `qwen_finetuned_checkpoint.tar.gz`
   - Select "Download"

3. **Extract locally**:
   ```powershell
   tar -xzf qwen_finetuned_checkpoint.tar.gz
   ```

---

## Step 8: Local Inference

Run inference locally (NOT via API):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto",
    torch_dtype="auto",
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(
    base_model,
    "./qwen_finetuned_output/checkpoint-264",  # Your checkpoint
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Test inference
prompt = """You are an expert at finding frequent itemsets in transactional data.

Dataset (5 transactions):
Transaction 1: milk (item_A), bread (item_B), eggs (item_C)
Transaction 2: milk (item_A), bread (item_B), butter (item_D)
Transaction 3: milk (item_A), eggs (item_C), butter (item_D)
Transaction 4: bread (item_B), eggs (item_C), butter (item_D)
Transaction 5: milk (item_A), bread (item_B), eggs (item_C), butter (item_D)

Find all frequent itemsets with minimum support of 3 transactions (60%)."""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## Expected Results

### Ground Truth (Apriori with min_support=3):
```json
[
  {"itemset": ["milk (item_A)"], "support": 0.8, "count": 4},
  {"itemset": ["bread (item_B)"], "support": 0.8, "count": 4},
  {"itemset": ["eggs (item_C)"], "support": 0.8, "count": 4},
  {"itemset": ["butter (item_D)"], "support": 0.6, "count": 3},
  {"itemset": ["milk (item_A)", "bread (item_B)"], "support": 0.6, "count": 3},
  {"itemset": ["milk (item_A)", "eggs (item_C)"], "support": 0.6, "count": 3},
  {"itemset": ["bread (item_B)", "eggs (item_C)"], "support": 0.6, "count": 3},
  {"itemset": ["milk (item_A)", "bread (item_B)", "eggs (item_C)"], "support": 0.6, "count": 3}
]
```

### Fine-Tuned Model Output (Expected):
- Should extract most 3-itemsets with support ≥ 0.6
- May miss some 2-itemsets (acceptable)
- Should include evidence rows for each itemset
- Counts and support should be accurate

---

## Troubleshooting

### Issue: Dataset Not Found
**Solution**: Verify `hf_dataset_enhanced/` uploaded correctly
```bash
ls -R hf_dataset_enhanced/
```

### Issue: Model Download Slow
**Solution**: Pre-download model before training
```python
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
```

### Issue: Training Hangs
**Solution**: Check GPU utilization
```bash
nvidia-smi
```

### Issue: Loss Not Decreasing
**Possible Causes**:
- Learning rate too low → increase to 3e-4
- Dataset corrupted → re-upload `hf_dataset_enhanced/`
- Model not in training mode → verify `model.train()` called

---

## Checkpoints and Logs

Training creates these artifacts:
```
qwen_finetuned_output/
├── checkpoint-100/      # Step 100 checkpoint
├── checkpoint-200/      # Step 200 checkpoint
├── checkpoint-264/      # Final checkpoint (best)
└── runs/                # TensorBoard logs (if enabled)
```

**Best Checkpoint**: Typically the final one (`checkpoint-264` for 3 epochs, 88 examples)

---

## Next Steps After Training

1. **Validate Performance**: Run Cell 23 (Test Inference) on validation examples
2. **Compare to Apriori**: Use `validation_reports/` to compare LLM vs Apriori accuracy
3. **Analyze Errors**: Identify common failure modes (missing itemsets, incorrect counts)
4. **Iterate Dataset**: Add more real-world examples if performance insufficient
5. **Deploy Locally**: Integrate fine-tuned model into `pipeline.py` for batch processing

---

## References

- **Enhanced Dataset**: `hf_dataset_enhanced/` (88 train / 10 validation)
- **Best Practices**: `BEST_PRACTICES.md` (empirical findings)
- **Training Guide**: `FINETUNING_README.md` (comprehensive documentation)
- **Notebook**: `qwen_finetuning_server.ipynb` (26 cells, 4-bit QLoRA)
- **Model Card**: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

---

## Contact

If training fails repeatedly, check:
1. Server GPU availability (`nvidia-smi`)
2. Python version (`python --version` ≥ 3.8)
3. Disk space (`df -h` > 20GB free)
4. Memory (`free -h` > 20GB available)

For persistent issues, consult server administrator or reference `BEST_PRACTICES.md`.
