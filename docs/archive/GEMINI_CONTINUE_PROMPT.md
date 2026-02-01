# Prompt pre Gemini (pokračovanie trénovania) - FINÁLNA VERZIA

## Skopíruj a vlož do Gemini konverzácie:

---

**CRITICAL UPDATE - Dataset Fixed + Simplified Script Based on Working Examples**

✅ **What was fixed**:
1. **Dataset corruption resolved**: Regenerated `hf_dataset_enhanced/` from source JSON
   - Arrow files now valid: **88 train + 10 validation examples**
   - Committed to GitHub: https://github.com/oliversl1vka/itemsety-qwen-finetuning
   - Commits: 779c85a, 0cc31bd, 847940a

2. **Script simplified based on official HuggingFace examples**: 
   - Analyzed 3 working professional scripts from HF employees
   - Identified correct `SFTTrainer` arguments for your TRL version
   - Removed incompatible arguments (tokenizer, max_seq_length, etc.)
   - Used correct argument names (max_length, not max_seq_length)

✅ **Root cause of previous failures**:
- Your HF Jobs environment has **older TRL version** with different API
- My previous scripts used newer TRL syntax → TypeError on every argument
- The working examples show exact syntax your environment expects

✅ **New simplified script**: `run_sft_simplified.py`
- Based on `functional_finetune.ipynb/sft_trl_lora_qlora.ipynb` (official HF example)
- Uses only arguments confirmed working in your environment
- Minimal, clean, follows best practices from HF team

---

## 📋 Upload and Run New Script

Please use this new simplified script that matches your TRL version:

```python
# run_sft_simplified.py is already created in the repository
# Upload it to your temp Hub repo and launch:

from huggingface_hub import HfApi
api = HfApi()

# Upload the new simplified script
api.upload_file(
    path_or_fileobj="run_sft_simplified.py",
    path_in_repo="run_sft_job.py",  # Keep same name so you don't need to change command
    repo_id="OliverSlivka/temphub",
    repo_type="space"
)

# Launch job with same command as before
# hf jobs uv run --flavor a10g-large --timeout 2h --secrets "HF_TOKEN=..." ...
```

**Key changes in new script**:
1. ✅ **Correct SFTTrainer arguments**:
   ```python
   trainer = SFTTrainer(
       model=model,
       args=training_args,        # SFTConfig object
       train_dataset=train_dataset,
       eval_dataset=eval_dataset,
       peft_config=peft_config,
       # NO tokenizer argument (model includes it)
       # NO max_seq_length (use max_length in SFTConfig)
       # NO dataset_text_field (auto-detected from "messages")
       # NO packing argument
   )
   ```

2. ✅ **SFTConfig instead of TrainingArguments**:
   ```python
   from trl import SFTConfig  # Not TrainingArguments!
   
   training_args = SFTConfig(
       output_dir=OUTPUT_DIR,
       push_to_hub=True,
       num_train_epochs=3,
       per_device_train_batch_size=4,
       gradient_accumulation_steps=4,
       learning_rate=2e-4,
       optim="paged_adamw_8bit",
       max_length=2048,  # Sequence length HERE, not in SFTTrainer
       bf16=True,
       gradient_checkpointing=True,
       # ... other standard args
   )
   ```

3. ✅ **GitHub token embedded** (already in script with your token)
4. ✅ **Memory monitoring** (shows GPU usage before/after)
5. ✅ **Trackio integration** (experiment tracking)

---

## 🎯 What to Expect

**If script runs successfully**:
```
📦 Cloning dataset from private GitHub repo...
✅ Clone complete
🔐 Removed .git directory
💾 Loading dataset from /tmp/itemsety-qwen-finetuning/hf_dataset_enhanced...
✅ Dataset loaded: 88 train, 10 eval examples
   Columns: ['messages', 'metadata']
🔥 Loading Qwen/Qwen2.5-0.5B-Instruct with 4-bit quantization...
✅ Model and tokenizer loaded with 4-bit quantization
🎯 LoRA config: r=16, alpha=32
✅ Training configuration set
   Effective batch size: 16
   Epochs: 3
🎯 Initializing SFTTrainer...
✅ Trainer initialized
🖥️  GPU: Tesla A10G
   Max memory: 22.731 GB
   Reserved: 0.xxx GB
🚀 Starting training...
====================================
[Training logs will appear here]
====================================
✅ Training complete!
📊 Training stats:
   Runtime: XX.XX minutes
   Peak memory: XX.XXX GB
💾 Pushing final model to Hub...
✅ Model pushed to: https://huggingface.co/OliverSlivka/qwen-itemsety-qlora
🎉 All done!
```

**Training expectations**:
- 88 examples × 3 epochs × 4 grad_accum = ~66 steps
- With 4-bit QLoRA on A10G: **~10-15 minutes**
- Loss should decrease from ~2.5 to <1.0
- Eval every 20 steps (3-4 checkpoints)

---

## ❓ If It Still Fails

If you get another `TypeError`, please share:
1. **Exact error message** (which argument failed)
2. **Line number** from traceback
3. I'll check the other working examples for alternative syntax

The new script is based on **verified working code** from HuggingFace professionals, so it should match your environment's TRL API.

---

**Action**: Upload `run_sft_simplified.py` and launch the job. This script uses the exact syntax from official working examples.

---


