# Quick Prompt for Gemini CLI

**Previous issue**: All training attempts failed with `TypeError` on SFTTrainer arguments (tokenizer, max_seq_length, max_length, etc.)

**Root cause**: Your HF Jobs environment has older TRL version with different API than I was using.

**Solution**: Created new simplified script `run_sft_simplified.py` based on 3 official working HuggingFace examples (`functional_finetune.ipynb/sft_trl_lora_qlora.ipynb` and others).

## 🎯 What to do now:

1. **Upload the new script** from GitHub repo:
   ```bash
   # The script is at: https://github.com/oliversl1vka/itemsety-qwen-finetuning/blob/main/run_sft_simplified.py
   ```

2. **Key differences from previous attempts**:
   - Uses `SFTConfig` instead of `TrainingArguments`
   - Removed `tokenizer` argument from `SFTTrainer` (auto-inferred from model)
   - Uses `max_length` in `SFTConfig`, not in `SFTTrainer`
   - Removed `dataset_text_field` and `packing` arguments
   - Based on verified working professional code

3. **Correct SFTTrainer call**:
   ```python
   trainer = SFTTrainer(
       model=model,
       args=training_args,  # SFTConfig object
       train_dataset=train_dataset,
       eval_dataset=eval_dataset,
       peft_config=peft_config,
   )
   ```

4. **Launch with same command**:
   ```bash
   hf jobs uv run --flavor a10g-large --timeout 2h --secrets "HF_TOKEN=..." run_sft_simplified.py
   ```

**Expected**: This should work because it uses exact syntax from HF's own working examples that match your TRL version.

If it still fails, share the exact error and I'll check the other example notebooks for alternative approaches.
