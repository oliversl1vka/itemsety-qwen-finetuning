# Training Troubleshooting

**Date:** 2026-03-18  
**Source:** HuggingFace Skills `troubleshooting.md` + project training memory  
**Tags:** #reference #troubleshooting #training #bugs

---

## Quick Diagnosis

| Symptom | Likely Cause | Jump To |
|---------|-------------|---------|
| CUDA OOM | Batch too large / sequences too long | [OOM](#out-of-memory-oom) |
| Loss NaN/Inf | Learning rate too high / dtype issue | [NaN Loss](#nan-or-inf-loss) |
| Loss stuck / not decreasing | LR too low / data issue | [Training Stalls](#training-stalls) |
| Model outputs gibberish | Broken tokenizer / wrong chat template | [Bad Outputs](#bad-model-outputs) |
| Repetition loops | LR too high / no regularization | [Repetition](#repetition-loops) |
| JSON parse failures | Training format mismatch / truncation | [JSON Issues](#json-parse-failures) |
| DPO KL explosion | Beta too low / bad preference data | [DPO Issues](#dpo-specific-issues) |
| GRPO reward collapse | Reward function too sparse | [GRPO Issues](#grpo-specific-issues) |
| Model merge crash | 4-bit merge attempt | [Merge Issues](#model-merge-issues) |
| Slow training | Missing optimizations | [Performance](#slow-training) |

---

## Out of Memory (OOM)

**Solutions (in priority order):**

1. **Reduce batch size:** `per_device_train_batch_size = 1`
2. **Increase gradient accumulation:** `gradient_accumulation_steps = 8`
3. **Reduce sequence length:** `max_seq_length = 1024` (from 2048)
4. **Disable evaluation:** Remove `eval_dataset` and `eval_strategy`
5. **Enable gradient checkpointing:** `gradient_checkpointing = True`
6. **Use Unsloth:** `use_gradient_checkpointing = "unsloth"` (30% savings)
7. **Use smaller LoRA rank:** `r = 16` instead of `r = 32`
8. **Use 4-bit quantization:** `load_in_4bit = True`
9. **Use larger GPU** (see [[Hardware and Memory Guide]])

**Memory guidelines:**
- T4 (16GB): <1B models with LoRA
- A10G (24GB): 1-3B models with LoRA, <1B full fine-tune
- A100 (40GB/80GB): 7B+ models with LoRA, 3B full fine-tune
- H200 (141GB): Comfortable for 7B 4-bit LoRA with room to spare

---

## NaN or Inf Loss

**Causes & fixes:**

| Cause | Fix |
|-------|-----|
| Learning rate too high | Reduce: `1e-4` → `5e-5` → `2e-5` |
| fp16 instability | Switch to `bf16=True` (if GPU supports it) |
| Data has empty messages | Filter dataset: remove entries with empty content |
| Gradient explosion | Add `max_grad_norm=1.0` |
| Mixed precision mismatch | Ensure `bnb_4bit_compute_dtype=torch.bfloat16` matches training dtype |

**macOS-specific:** Keep `fp16=False` on MPS devices — always use float32.

---

## Training Stalls

**Loss not decreasing:**

1. **Learning rate too low:** Try `2e-4` (SFT) or `5e-7` (DPO)
2. **Data too small:** Need 100+ examples for SFT to learn patterns
3. **Labels not masked properly:** Verify `train_on_responses_only()` masks system/user tokens
4. **Wrong data format:** Ensure ChatML messages format matches tokenizer expectations
5. **Frozen layers:** Check `model.print_trainable_parameters()` — should be ~0.5-2%

**Verify label masking:**
```python
# After train_on_responses_only()
sample = trainer.train_dataset[0]
labels = sample["labels"]
# Should see -100 (masked) for system/user, real token IDs for assistant
assert any(l == -100 for l in labels), "No masked tokens — masking not working!"
assert any(l != -100 for l in labels), "All tokens masked — nothing to learn!"
```

---

## Bad Model Outputs

**Model generates gibberish or irrelevant text:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Random tokens | Tokenizer mismatch | Load tokenizer from same checkpoint as model |
| Wrong language | Chat template issue | Verify `apply_chat_template()` produces correct format |
| Cuts off early | `max_new_tokens` too low | Increase to 1024-2048 |
| Repeats prompt | Missing generation prompt | Set `add_generation_prompt=True` |
| No `<think>` tag | Training didn't include CoT | Verify SFT data has `<think>` in assistant messages |

---

## Repetition Loops

**Model repeats same text endlessly:**

This was our **#1 issue in v1-v2 training**.

**Fixes applied in v3:**
1. **Lower learning rate:** `1e-4` instead of `2e-4` (Council recommendation)
2. **Add repetition penalty at inference:** `repetition_penalty=1.2`
3. **Add LoRA dropout:** `lora_dropout=0.05`
4. **Temperature at inference:** `temperature=0.3` (not too high, not too low)
5. **DPO alignment:** Real failure pairs teach model what NOT to do
6. **StoppingCriteria:** Custom stopping when JSON bracket closes

**Inference-time mitigation:**
```python
from transformers import StoppingCriteria

class JsonBracketStop(StoppingCriteria):
    """Stop generation when JSON array closes."""
    def __call__(self, input_ids, scores, **kwargs):
        text = tokenizer.decode(input_ids[0][-50:])
        # Count brackets in generated text
        return text.count(']') > text.count('[')
```

---

## JSON Parse Failures

**Model outputs invalid JSON:**

| Pattern | Cause | Fix |
|---------|-------|-----|
| Text before `[` | Model adds preamble | Strip to first `[` |
| Text after `]` | Model adds explanation | Strip after last `]` |
| Truncated JSON | `max_new_tokens` too low | Increase to 1024+ |
| Missing quotes | Format not learned | More training data with strict JSON |
| Trailing comma | Common LLM mistake | `json.loads(text.rstrip(',]') + ']')` |
| Nested objects wrong | Complex structure | Simplify output format in training |

**Robust JSON extraction:**
```python
import re, json

def extract_json(text):
    """Extract JSON array from model output, handling common issues."""
    # Find the outermost [...] 
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        # Try fixing trailing commas
        fixed = re.sub(r',\s*([}\]])', r'\1', match.group())
        return json.loads(fixed)
```

---

## DPO-Specific Issues

### KL Divergence Explosion
- **Symptom:** DPO loss diverges, model quality drops
- **Fix:** Increase `beta` (0.1 → 0.3) to stay closer to reference model

### Chosen/Rejected Too Similar  
- **Symptom:** DPO has no effect, metrics don't improve
- **Fix:** Ensure rejected examples are clearly wrong (our real failures are good)

### Reference Model Mismatch
- **Symptom:** Training crashes or produces garbage
- **Fix:** DPO reference model must be the SFT checkpoint, not base model

---

## GRPO-Specific Issues

### Reward Collapse (All Zeros)
- **Symptom:** All rewards are 0, no gradient signal
- **Fix:** Make reward function more granular (partial credit, not binary)

### Complete Output Collapse
- **Symptom:** Model outputs empty strings or single tokens
- **Fix:** This happened in our v1 — GRPO was abandoned for now

### ⚠️ GRPO Status
GRPO is **skipped** in current training (Council Decision 2026-03-09). See [[Training Methods Guide]] for details.

---

## Model Merge Issues

### ❌ CRITICAL: Never merge 4-bit models
```python
# ❌ THIS WILL CORRUPT THE MODEL
model = model.merge_and_unload()  # On 4-bit model → garbage weights

# ✅ Save adapter only
model.save_pretrained("adapter-output")
model.push_to_hub("user/model-adapter")

# ✅ For merged model, reload in full precision first
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16)
from peft import PeftModel
merged = PeftModel.from_pretrained(base, "adapter-output").merge_and_unload()
```

This was our **catastrophic v1 bug** — `merge_4bit_forced=True` produced a model that output random tokens.

---

## Slow Training

**Optimization checklist:**

| Optimization | Speedup | How |
|-------------|---------|-----|
| Unsloth | 2× | `FastLanguageModel.from_pretrained()` |
| Flash Attention | 1.5-2× | `pip install flash-attn` (auto-detected) |
| bf16 mixed precision | 1.5× | `bf16=True` in training config |
| Gradient checkpointing | Slower but fits larger models | `gradient_checkpointing=True` |
| `for_inference()` | 2× inference speed | `FastLanguageModel.for_inference(model)` |
| Packing (careful) | Variable | `packing=True` — test first, may break masking |

---

## macOS-Specific Issues (Apple Silicon)

| Problem | Fix |
|---------|-----|
| MPS unsupported op / crash | `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| OOM / system instability | Reduce `MAX_SEQ_LENGTH`, use smaller model |
| fp16 NaN / loss explosion | Keep `fp16=False`, use float32 |
| LoRA "module not found" | Print `model.named_modules()` to find correct target names |
| TRL TypeError on args | Check TRL version; use `SFTConfig` + `processing_class` (TRL ≥0.12) |

### Common LoRA target modules by architecture

| Architecture | target_modules |
|-------------|---------------|
| Qwen/Qwen2.5 | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| LLaMA/Mistral | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Phi-3/4 | `qkv_proj, o_proj, gate_up_proj, down_proj` |
| Gemma | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |

---

## TLJH Server-Specific Issues

From Training Agent memory:

| Problem | Fix |
|---------|-----|
| Wrong GPU selected | `CUDA_VISIBLE_DEVICES=1` (always GPU 1 on our server) |
| Keras 3 crash | `os.environ["TF_USE_LEGACY_KERAS"] = "1"` |
| torch/CUDA mismatch | Check: `torch.version.cuda` matches installed CUDA |
| Package conflicts | Use `!pip install --no-deps unsloth` to avoid torch reinstall |
| Notebook kernel dies | OOM — reduce batch size or sequence length |

---

## See Also

- [[Hardware and Memory Guide]] — VRAM estimation and GPU selection
- [[Training Methods Guide]] — Method-specific configuration
- [[Unsloth Notebook Patterns]] — Proven defaults from 11 notebooks
- [[Training Agent]] — Full history of bugs and fixes
