# Autonomous Fine-Tuning Prompt for Gemini CLI

Copy this entire prompt into Gemini CLI to run autonomous test → production fine-tuning:

---

## Mission: Autonomous Fine-Tuning of Qwen2.5-3B for Itemset Extraction

You are an autonomous AI agent with access to HuggingFace Skills MCP server. Your mission is to fine-tune Qwen2.5-3B on a real-world dataset through a complete test → production pipeline. You should make technical decisions, debug issues, monitor progress, and iterate until both training runs succeed.

### Project Context

**Task**: Train a language model to extract frequent itemsets from transaction datasets (data mining task).

**Dataset**: `OliverSlivka/itemset-extraction-v2` (already uploaded to HuggingFace Hub)
- 488 training examples (439 for training, 49 for validation)
- Format: ChatML with `messages` column (system/user/assistant)
- Real-world data from 25 domains (e-commerce, healthcare, education, etc.)
- Each example contains: dataset CSV → Apriori itemsets → LLM extraction task

**Target Model**: Qwen2.5-3B-Instruct
- Base: Pre-trained instruction model from Qwen team
- Size: 3B parameters (~6GB in fp16, ~2GB in 4-bit)
- Why 3B: Previous 0.5B model achieved only 6.7% valid JSON; 3B should reach 80-90%

**Expected Outcome**: 
- Test model: `OliverSlivka/qwen2.5-3b-itemset-test` (validation run)
- Production model: `OliverSlivka/qwen2.5-3b-itemset-extractor` (final model)

---

### Phase 1: Dataset Validation (REQUIRED FIRST)

Before any training, validate the dataset:

```
Validate the dataset OliverSlivka/itemset-extraction-v2 for SFT training.
Check format, column structure, and confirm it has the 'messages' field in ChatML format.
Report total examples and split (train/validation).
```

**Expected**: 488 total examples, `messages` column with system/user/assistant structure.

If validation fails, STOP and report the issue. Do not proceed to training.

---

### Phase 2: Test Training Run (Quick Validation)

Once dataset is validated, run a test training with a small subset:

**Objective**: Verify the complete pipeline works before committing to full training.

**Instructions to Gemini**:
```
Fine-tune Qwen/Qwen2.5-3B-Instruct on OliverSlivka/itemset-extraction-v2 for a TEST RUN.

Configuration:
- Method: Supervised Fine-Tuning (SFT)
- Hardware: t4-small (cheapest option, sufficient for test)
- Training samples: 50 examples (not full dataset)
- Validation samples: 49 examples
- Training steps: 12 max steps (quick validation)
- Epochs: 1 epoch
- Batch size: 2 per device
- Gradient accumulation: 8 steps (effective batch size 16)
- Learning rate: 2e-4
- Quantization: 4-bit QLoRA (NF4, double quantization)
- LoRA config: rank=16, alpha=32, dropout=0.05
- Target modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
- Precision: fp16=True, bf16=False (T4 compatibility - CRITICAL)
- Optimizer: paged_adamw_8bit
- Max sequence length: 2048 tokens
- Gradient checkpointing: True
- Warmup steps: 5
- Max grad norm: 0.3
- Output repo: OliverSlivka/qwen2.5-3b-itemset-test
- Push to Hub: Yes (at end)
- Monitoring: Enable Trackio integration

Expected duration: 10-15 minutes
Expected cost: ~$0.15

IMPORTANT: If you encounter any errors (OOM, bf16 issues, dataset format), 
debug and fix them autonomously. Common fixes:
- If OOM: Reduce batch size to 1, increase gradient accumulation to 16
- If bf16 error: Ensure bf16=False, fp16=True (T4 doesn't support bf16)
- If auth error: Verify HF_TOKEN is set and has write permissions
- If dataset error: Check column names match expected format

Submit the test job and monitor progress. Report job ID, monitoring URL, and ETA.
```

**Wait for test completion** before proceeding to Phase 3.

Once test job is running:
```
Monitor the test training job. Check Trackio dashboard every 5 minutes.
Report: current step, training loss trend, any warnings/errors.

When complete, verify:
1. Model pushed to OliverSlivka/qwen2.5-3b-itemset-test
2. Training loss decreased (started ~2.5, should end ~1.5 or lower)
3. No errors in final logs
```

---

### Phase 3: Production Training (Full Dataset)

After test succeeds, proceed to full production training:

**Instructions to Gemini**:
```
Fine-tune Qwen/Qwen2.5-3B-Instruct on OliverSlivka/itemset-extraction-v2 for PRODUCTION.

Configuration:
- Method: Supervised Fine-Tuning (SFT)
- Hardware: t4-small or t4-medium (select based on test performance)
- Training samples: 439 examples (full training set)
- Validation samples: 49 examples
- Training steps: -1 (use epochs instead)
- Epochs: 3 epochs (~81 steps estimated)
- Batch size: 2 per device
- Gradient accumulation: 8 steps (effective batch size 16)
- Learning rate: 2e-4
- Quantization: 4-bit QLoRA (same as test)
- LoRA config: rank=16, alpha=32, dropout=0.05
- Target modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
- Precision: fp16=True, bf16=False (T4 compatibility)
- Optimizer: paged_adamw_8bit
- Max sequence length: 2048 tokens
- Gradient checkpointing: True
- Warmup steps: 10
- Max grad norm: 0.3
- Output repo: OliverSlivka/qwen2.5-3b-itemset-extractor
- Push to Hub: Yes (at end)
- Monitoring: Enable Trackio integration
- Checkpointing: Save every 50 steps, keep 2 best checkpoints
- Evaluation: Every 20 steps

Expected duration: 40-60 minutes
Expected cost: $0.40-0.60 (on t4-small @ $0.60/hour)

Submit the production job and monitor progress. Report job ID, monitoring URL, and cost estimate.
```

**Continuous monitoring during production training**:
```
Monitor the production training job throughout its duration.

Check every 10 minutes and report:
- Current step / total steps
- Training loss (should decrease smoothly)
- Evaluation loss (should track training loss)
- Learning rate schedule
- ETA to completion
- Any warnings or anomalies

If training loss plateaus or increases, recommend adjustments for future runs.
If job fails, debug the error and suggest fixes.
```

---

### Phase 4: Post-Training Verification

After production training completes:

```
Verify the final model OliverSlivka/qwen2.5-3b-itemset-extractor:

1. Confirm model is pushed to Hub
2. Check final training metrics:
   - Final training loss (target: <1.0)
   - Final validation loss (should be close to training loss)
   - Total training time and cost
3. Review model card on Hub (auto-generated by trainer)
4. Test model loading:
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("OliverSlivka/qwen2.5-3b-itemset-extractor")
   tokenizer = AutoTokenizer.from_pretrained("OliverSlivka/qwen2.5-3b-itemset-extractor")

Report summary:
- Test model: [status, metrics, Hub link]
- Production model: [status, metrics, Hub link]
- Total cost: [test + production]
- Total time: [test + production]
- Next steps: [recommendations for evaluation]
```

---

### Phase 5: Optional GGUF Conversion (If Requested)

If local deployment is needed:

```
Convert the fine-tuned model OliverSlivka/qwen2.5-3b-itemset-extractor to GGUF format.

Configuration:
- Quantization: Q4_K_M (good balance of size vs quality)
- Output repo: OliverSlivka/qwen2.5-3b-itemset-extractor-gguf
- Merge LoRA adapters before conversion
- Push to Hub after conversion

After conversion, provide usage instructions:
llama-server -hf OliverSlivka/qwen2.5-3b-itemset-extractor-gguf:Q4_K_M
```

---

## Autonomous Decision-Making Guidelines

You have full autonomy to:

1. **Hardware Selection**: Choose between t4-small and t4-medium based on test results. If test is slow, upgrade to t4-medium for production.

2. **Hyperparameter Tuning**: If test shows poor convergence (loss not decreasing), adjust:
   - Learning rate (try 1e-4 or 3e-4)
   - Warmup steps (increase to 10-15)
   - Gradient accumulation (if OOM, reduce batch size and increase this)

3. **Error Recovery**: 
   - OOM → Reduce batch size, increase gradient accumulation
   - Slow convergence → Increase learning rate slightly
   - Auth errors → Guide user to check HF_TOKEN
   - Dataset errors → Inspect and suggest fixes

4. **Monitoring Strategy**:
   - Check Trackio every 5 min during test
   - Check every 10 min during production
   - Alert if loss stops decreasing for >15 minutes

5. **Cost Optimization**:
   - Use t4-small if test completes in <15 min
   - Suggest t4-medium only if test is slow or shows OOM risk

---

## Expected Timeline

- **Dataset validation**: 2-3 minutes (CPU job)
- **Test training**: 10-15 minutes (~$0.15)
- **Analysis/adjustments**: 5-10 minutes
- **Production training**: 40-60 minutes (~$0.50)
- **Verification**: 5 minutes
- **Total time**: ~70-90 minutes
- **Total cost**: ~$0.65

---

## Success Criteria

✅ **Test Training**:
- Job completes without errors
- Training loss drops from ~2.5 to ~1.5 or lower
- Model pushed to Hub successfully

✅ **Production Training**:
- Job completes 3 epochs without errors
- Final training loss < 1.0
- Validation loss within 0.2 of training loss (no overfitting)
- Model pushed to Hub successfully
- Model can be loaded with transformers library

✅ **Overall**:
- Both models on HuggingFace Hub
- Complete logs and metrics available
- Total cost within budget (~$0.65)
- Ready for evaluation on real-world test datasets

---

## Start Command

Begin with:
```
Start autonomous fine-tuning pipeline for Qwen2.5-3B on itemset extraction task.
Follow the 5-phase plan: Validate → Test → Production → Verify → (Optional GGUF).
Make decisions independently, debug issues as they arise, and report progress at each phase.
```

---

## Additional Context (Technical Details)

**Why 4-bit quantization**: 
- Reduces memory ~4x (6GB → 1.5GB)
- Enables training on t4-small (~$0.60/hr vs a10g-large ~$1.50/hr)
- QLoRA proven as effective as full precision LoRA
- No quality loss for instruction tuning tasks

**Why fp16, not bf16**:
- T4 GPU has limited bf16 support (causes NotImplementedError)
- fp16 fully supported and tested on T4
- Training script explicitly sets bf16=False

**Why LoRA**:
- Full fine-tuning 3B would require >16GB VRAM
- LoRA with rank=16 is 99% as effective, fits easily in 16GB
- Standard approach for models >1B parameters

**Dataset specifics**:
- Synthetic + real hybrid (generated from 25 real CSV sources)
- Anti-hallucination rules in system prompt
- Evidence-based extraction (must cite row numbers)
- Chain-of-thought format option available

**Baseline performance**:
- Qwen2.5-0.5B: 6.7% valid JSON (too small)
- GPT-4o: 100% valid JSON (target baseline)
- Target for 3B: 80-90% valid JSON

---

END OF PROMPT
