# Training Agent Memory

Persistent knowledge store for model fine-tuning insights.

**Agent file:** `.github/agents/training-agent.md`  
**Tags:** #agent/training

---

## [2026-02-08] ⚠️ All pre-2026-02-08 pipeline data is INVALID

**Context:** A bug in `pipeline.py` passed singletons (Apriori size-1 itemsets) into the LLM extractor response field instead of the actual LLM output. All validation "passes" before this date were meaningless.  
**Insight:** Training data exported from runs before 2026-02-08 MUST NOT be used. Only post-bugfix pipeline runs produce real training signal.  
**Application:** When running `/export`, verify all source runs are from 2026-02-08 or later.  
**Tags:** #bug #critical

See also: [[References/Pipeline Bug 2026-02-08]]

---

## [2026-02-08] LLM Model Selection for Training Data Generation

**Context:** After fixing the pipeline bug, user tested models on real LLM extraction.

**Validated results (post-bugfix only):**

| Model | Result | API Tier | Daily Limit |
|-------|--------|----------|-------------|
| `gpt-4o-mini` | ❌ **0% validation pass** | Tier 2 (2.5M/day) | Cheap but useless — hallucinated items in rows |
| `gpt-4o` | ✅ **~80% validation pass** | Tier 1 (250k/day) | Works but only ~125 datasets/day |
| `gpt-4.1-mini` | ⏳ **Not yet tested post-bugfix** | Tier 2 (2.5M/day) | ~1250 datasets/day — **test this first** |

**Insight:** Prefer `gpt-4.1-mini` (pending validation). It's in the high-limit tier (2.5M tokens/day) and should handle 500+ datasets in a single day. If it fails, fall back to `gpt-4o` (will take ~4-6 days due to 250k limit).

**Pitfalls:**
- ❌ Never use `gpt-4o-mini` — 0% validation pass, wastes tokens
- ❌ Don't confuse `gpt-4o-mini` (bad) with `gpt-4.1-mini` (expected good)
- ⚠️ Don't trust any "100% pass rate" claims from before 2026-02-08 (pipeline bug)

**Tags:** #experiment #model/gpt-4o #model/gpt-4o-mini #model/gpt-4.1-mini

See also: [[References/Model Comparison]], [[References/API Limits]]

---

## [2026-02-17] ✅ gpt-4.1-mini Full Batch — COMPLETED (40.2% pass rate)

**Context:** Ran full 500-dataset batch with `gpt-4.1-mini`.

**Results:**
- Total: 500 runs | Validated: **201 (40.2%)** | Failed: 299
- Dates: 2026-02-17 → 2026-02-18

**Insight:** 40.2% pass rate is usable for RLHF training. Failures are mostly strict invariant checks (row count mismatches, support math). `gpt-4.1-mini` is the **primary model** for generating training data.
**Tags:** #production #model/gpt-4.1-mini

---

## [2026-02-22] ✅ gpt-4.1-nano Full Batch — COMPLETED (16.2% pass rate)

**Context:** Ran full 500-dataset batch with `gpt-4.1-nano` (cheaper/smaller model).

**Results:**
- Total: 500 runs | Validated: **81 (16.2%)** | Failed: 419
- Date: 2026-02-22

**Insight:** Lower quality than gpt-4.1-mini. 81 validated runs still add diversity to RLHF pairs. Not recommended as primary model — use gpt-4.1-mini instead.
**Tags:** #production #model/gpt-4.1-nano

---

## [2026-02-22] Production DB Summary — Training Data Export Ready

| Model | Total | Valid | Pass Rate |
|-------|-------|-------|-----------|
| `gpt-4.1-mini` | 500 | 201 | **40.2%** |
| `gpt-4.1-nano` | 500 | 81 | **16.2%** |
| **TOTAL** | **1000** | **282** | **28.2%** |

**Note:** These numbers were later corrected — see 2026-03-01 entry below.
**Tags:** #summary #production

---

## [2026-02-22] Training Method: SFT→DPO (Unsloth) — SUPERSEDED

**Context:** Training pipeline was updated from pure DPO to SFT warmup + DPO using Unsloth.

**⚠️ SUPERSEDED by 3-phase approach (2026-03-01).** See below.

**Tags:** #training-method #unsloth #sft-dpo #archived

---

## [2026-03-01] ✅ Corrected DB Numbers + o4-mini Results

**Context:** Re-analyzed runs.db with corrected queries. Also ran o4-mini batch.

**Corrected pipeline results:**

| Model | Total | Valid | Pass Rate |
|-------|-------|-------|-----------|
| `gpt-4.1-mini` | 500 | **135** | **27.0%** |
| `gpt-4.1-nano` | 500 | **79** | **15.8%** |
| `gpt-4o` | 100 | **22** | **22.0%** |
| `o4-mini` | ~500 | **~214** | **~42.8%** |
| **TOTAL** | **~1600** | **~450** | **~28%** |

**Insight:** Previous pass rates (40.2%, 16.2%) were inflated. o4-mini is best performer at ~42.8%.
**Tags:** #correction #production #model/o4-mini

---

## [2026-03-01] ✅ 3-Phase Training Method (SFT-CoT → DPO-Real → GRPO)

**Context:** Complete redesign of training pipeline. Previous SFT→DPO approach used synthetic corruptions for rejected responses. New approach uses:

**3 Phases:**
1. **Phase 1 — SFT-CoT** (3 epochs, lr=2e-4): Train on `<think>` chain-of-thought reasoning. 348 examples → 314 train / 34 val.
2. **Phase 2 — DPO-Real** (2 epochs, lr=5e-5, beta=0.1): Use real LLM failures from 4 models (nano 44.5%, o4-mini 26.6%, gpt-4.1-mini 25%, gpt-4o 3.8%) as rejected. 606 pairs → 546 train / 60 val.
3. **Phase 3 — GRPO** (1 epoch, lr=5e-6): 4 reward functions based on Apriori ground truth (JSON validity, itemset match, evidence accuracy, count correctness). 314 examples → 314 train / 34 val.

**Key differences from previous approach:**
- `<think>` tag for structured CoT reasoning (vs generic step-by-step)
- Real LLM failures for DPO (vs 6 synthetic error types)
- GRPO with 4 Apriori reward functions (vs stub only)
- Qwen2.5-7B target (vs 3B)
- 4096 seq length (vs 2048)
- LoRA r=64, alpha=16 (Unsloth optimized)

**New scripts:**
- `src/training/training_utils.py` — compact system prompt (~150 tokens), CoT generator, CSV loader
- `src/training/generate_cot_sft_data.py` — Phase 1 data (348 examples, avg 1547 tokens)
- `src/training/export_real_dpo_data.py` — Phase 2 data (606 pairs from 4 models)
- `src/training/build_hf_dataset_v2.py` — HF dataset with 3 configs (sft/dpo/grpo)

**Notebook:** `notebooks/training_3phase_7b.ipynb` (21 cells)
**Dataset repo:** `OliverSlivka/itemset-extraction-v2` (3 configs) — ⚠️ NOW FROZEN, current = `itemset-extraction-v3`
**Model repo:** `OliverSlivka/qwen2.5-7b-itemset-extractor`

**Tags:** #training-method #3-phase #sft-cot #dpo-real #grpo #qwen-7b

---

## [2026-03-01] 🎉 FIRST SUCCESSFUL 3-PHASE TRAINING RUN (v1)

**Context:** Full 3-phase training completed on school TLJH Jupyter server (NVIDIA H200 NVL, 150 GB VRAM).

### Hardware & Environment
- **GPU:** NVIDIA H200 NVL (150.1 GB VRAM, 4× GPUs available but only 1 used)
- **CUDA:** 9.0, Toolkit 12.8
- **Torch:** 2.10.0+cu128 (system-installed at `/opt/tljh/user/`)
- **Unsloth:** 2026.2.1
- **Transformers:** 4.57.6
- **Platform:** TLJH (The Littlest JupyterHub), Python 3.12, Linux
- **Server path:** `/home/jupyter-slio02@vse.cz/`
- **User packages:** `~/.local/lib/python3.12/site-packages/`

### TLJH Environment Fixes (CRITICAL for future runs)

**Problem 1: torch version conflict (RESOLVED)**
- Installing `unsloth` pulls `torch 2.10+` to `~/.local/` which shadows the system torch
- Original issue was with CUDA 11.8 (cu118) — but system was **actually updated to cu128**!
- **Fix:** Cell 1 removes core `torch` and `nvidia*` from `~/.local/` after pip install
- **Keep:** `torchvision`, `torchaudio` — unsloth imports them at module level

**Problem 2: Keras 3 crash (RESOLVED)**
- System has TensorFlow + Keras 3 installed
- `transformers` detects TF → tries to import `TFPreTrainedModel` → Keras 3 ValueError
- **Fix:** Set `USE_TF=0` and `USE_JAX=0` env vars BEFORE any transformers import (Cell 3)

**Problem 3: SFTTrainer formatting_func required (RESOLVED)**
- Newer unsloth's `SFTTrainer` requires explicit `formatting_func` (won't auto-detect `messages` column)
- **Fix:** Pre-format dataset with `tokenizer.apply_chat_template()` → `text` column
- Set `dataset_text_field="text"` in SFTConfig
- `train_on_responses_only` then works correctly on the tokenized text

### Training Results

| Phase | Duration | Epochs/Steps | Final Loss | Notes |
|-------|----------|-------------|------------|-------|
| **SFT-CoT** | 4 min 10 sec | 2 epochs (80 steps) | **0.0678** | Val loss: 0.044→0.038 |
| **DPO-Real** | 14 min 1 sec | 2 epochs (274 steps) | **0.0255** | Rewards chosen: 12.67, rejected: -5.53, accuracy: 100% |
| **GRPO** | 45 min 7 sec | 200 steps | **~0.0** | json_format_reward: ~0.95-1.0 avg |
| **Total** | **~63 min** | — | — | + model merge + upload |

### Phase-by-Phase Analysis

**Phase 1 — SFT-CoT (4 min):**
- ✅ Very fast convergence: loss dropped quickly
- ✅ Val loss improved both epochs (0.0446 → 0.0380) — no overfitting
- ✅ 2 epochs is perfect (originally planned 1, changed to 2 in CONFIG)
- ⚠️ Unsloth note: "Qwen2ForCausalLM does not accept `num_items_in_batch`" — harmless warning
- ⚠️ "gradient accumulation will be very slightly less accurate" — known Unsloth note, negligible

**Phase 2 — DPO-Real (14 min):**
- ✅ **Perfect reward accuracy: 1.0** (model always prefers chosen over rejected)
- ✅ Very clean separation: chosen rewards ~12.67, rejected ~-5.53, margin ~18.2
- ✅ Loss basically zero (0.000002 val loss) — model learned DPO perfectly
- ⚠️ This might be **too easy** — the rejected examples might be too obviously bad
- 💡 Consider harder negatives in v2: generate near-miss failures instead of blatant errors

**Phase 3 — GRPO (45 min, longest phase):**
- ✅ json_format_reward consistently 0.95-1.0 (model outputs valid JSON)
- ⚠️ itemset_f1_reward very low (0.0-0.04 most steps) — model not matching Apriori
- ⚠️ count_accuracy_reward also low (0.0-0.2 sporadically)
- ⚠️ thinking_reward nearly always 0.0 — model doesn't produce `<think>` tags during GRPO generation
- 💡 **GRPO seems underperforming** — the model generates valid JSON but wrong itemsets
- 💡 Possible causes: (1) GRPO prompt format doesn't trigger CoT, (2) max_completion_length=2048 too short for `<think>` + JSON, (3) only 200 steps insufficient
- 💡 Clipped ratio ~0% — no completions hit max length (good)

### Inference Test Result
- **⚠️ JSON parse FAILED** on the quick test: `"count": 2wo` (invalid number)
- Only found 1 itemset (bread, milk, eggs) — missed many valid pairs/singles
- This suggests the model needs more training or evaluation on diverse datasets

### Model Upload
- ✅ Pushed to `OliverSlivka/qwen2.5-7b-itemset-extractor` (5.56 GB)
- Model is live and downloadable

### Key Configuration (what worked)

```python
CONFIG = {
    "base_model":        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "max_seq_length":    4096,
    "lora_r":            64,
    "lora_alpha":        16,
    # SFT
    "sft_epochs":        2,
    "sft_lr":            2e-4,
    "sft_batch_size":    2,
    "sft_grad_accum":    4,
    # DPO  
    "dpo_epochs":        2,
    "dpo_lr":            5e-5,
    "dpo_beta":          0.1,
    "dpo_batch_size":    1,
    "dpo_grad_accum":    4,
    # GRPO
    "grpo_max_steps":    200,
    "grpo_lr":           5e-6,
    "grpo_batch_size":   1,
    "grpo_grad_accum":   4,
    "grpo_num_generations": 4,
    "grpo_max_completion_length": 2048,
}
```

- **Trainable params:** 161,480,704 / 7,777,097,216 (2.08%)
- **Optimizer:** adamw_8bit (paged for SFT, standard for DPO)
- **Precision:** bf16
- **Gradient checkpointing:** Unsloth-managed
- **H200 used only ~6-7 GB VRAM** (4-bit quantized) — massive overkill for this model

### Recommendations for v2 Training

1. **Skip GRPO or redesign it:** F1/count rewards are near-zero → model not learning from GRPO. The post-SFT+DPO model outputs valid JSON but wrong content during GRPO rollouts. Consider:
   - Increasing `grpo_max_steps` to 500+
   - Increasing `grpo_num_generations` to 8
   - Adding warmup specifically for GRPO
   - OR just skip GRPO and rely on SFT+DPO (Phase 1+2 only)
   
2. **DPO might be too easy:** 100% accuracy and near-zero loss suggests the rejected examples are trivially bad. For v2, consider mixing in harder negatives (partial correctness, off-by-one counts).

3. **SFT is solid:** 2 epochs with lr=2e-4 on 314 examples converges well. Don't change this.

4. **Run proper evaluation:** The quick inference test showed a JSON error. Need to run `eval_finetuned_model.py` on the full eval set to get F1/P/R numbers.

5. **Consider `<think>` in GRPO prompts:** The model didn't produce `<think>` tags during GRPO — this means GRPO can't learn CoT improvements. May need to add system prompt to GRPO dataset or adjust reward to be gentler.

**Tags:** #training-run #v1 #success #3-phase #h200 #tljh #lessons-learned

See also: [[Experiments/2026-03-01 Qwen2.5-7B v1]]

---

## [2026-03-01] 🔧 TLJH Server Setup Checklist

**For future training runs on the school server:**

1. Download notebook from HF: `wget https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v3/resolve/main/notebooks/training_3phase_7b.ipynb`
2. Run Cell 1 (install) → installs packages, removes user-level torch
3. **RESTART KERNEL** (Kernel → Restart)
4. Run Cell 2+ (system torch is now the active one)
5. If re-running: Cell 1 is idempotent, safe to re-run

**Environment details:**
- Python: 3.12
- System packages: `/opt/tljh/user/lib/python3.12/site-packages/`
- User packages: `~/.local/lib/python3.12/site-packages/`
- User packages have HIGHER priority (shadows system)
- Must delete torch/nvidia from `~/.local/` after unsloth install

**Tags:** #tljh #server #setup #checklist

---

## [2026-02-08] OpenAI API Token Limits

See [[References/API Limits]] for the full breakdown.

**Planning guidance:** At ~2000 tokens/dataset:
- Tier 1: ~125 datasets/day (multi-day batch)
- Tier 2: ~1250 datasets/day (single-day batch)

**Application:** Always prefer Tier 2 models for batch pipeline runs. Use `--llm-model gpt-4.1-mini`.  
**Tags:** #insight #api

---

## [2026-03-07] v2 Training Notebook — Council-Corrected Export

**Context:** v1 training (2026-03-01) catastrophically failed (0% F1). Two LLM Council sessions diagnosed root causes. v2 notebook regenerated with ALL fixes.

### v2 Key Changes (from v1)
| Parameter | v1 (broken) | v2 (fixed) |
|-----------|-------------|------------|
| LoRA alpha/r | 16/64 = 0.25 | 64/32 = **2.0** |
| DPO | Present | **REMOVED** (4/4 council unanimous) |
| SFT epochs | 2 | **5** |
| SFT lr | 2e-4 | **1e-4** |
| packing | default (True) | **False** |
| Save between phases | `merged_4bit_forced` | **Adapters only** |
| GRPO rewards | 4 (broken) | **5 (partial credit)** |
| max_seq_length | 4096 | **6144** |
| weight_decay | None | **0.01** |
| GRPO steps | 200 | **500** |
| GRPO completions | 2048 | **3000** |
| Format gate | None | **Added** |

### Notebook Details
- **File:** `notebooks/training_3phase_2026-03-07_v2.ipynb`
- **Cells:** 18 (14 code, 4 markdown)
- **Phases:** SFT-CoT (5 epochs) → Format Gate → GRPO (500 steps)
- **DPO completely removed** — was killing `<think>` reasoning
- **5 GRPO reward functions:** structure, colval_format, f1, recall_bonus, grounding
- **Format verification gate:** Tests 20 val samples for `<think>`, JSON, `col:val` compliance before GRPO
- **Adapters-only saves** between phases — no more quantization cascade
- **GRPO grounding reward** extracts CSV headers from prompt messages (no csv_input column in dataset)

### Data Generated
- SFT-CoT: 348 examples → `data/sft_cot_v2.json`
- DPO-Real: 606 pairs → `data/dpo_real_v2.json` (kept but NOT used in v2 notebook)
- HF Dataset: SFT 314/34, DPO 546/60, GRPO 314/34 → `data/hf_dataset_v2/`

### Critical Notes for Next Training
1. GRPO dataset has NO `csv_input` column — CSV is embedded in user message of prompt
2. `reward_grounding` function parses `Row 1:` from user message to extract column names
3. TLJH torch cleanup Cell 1 is preserved — must restart kernel after first run
4. If format gate fails (<50%), increase SFT epochs before attempting GRPO
5. If GRPO rewards plateau, check that `reward_f1` is giving non-zero signal (test with pre-flight cell)

**Council sources:** [[../Decisions/2026-03-01 Council Training Advice]], `docs/reports/council_training_advice.json`
**Tags:** #v2 #council-corrected #notebook #export

---

## [2026-03-07] ✅ v2 Training Run — SFT-Only Model Pushed

**Context:** Ran v2 notebook on TLJH (H200 NVL). SFT succeeded, GRPO failed repeatedly. Pushed SFT-only model.

### SFT Training Results (EXCELLENT)
| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.0863 | 0.0588 |
| 2 | 0.0501 | 0.0397 |
| 3 | 0.0388 | 0.0327 |
| 4 | 0.0351 | 0.0297 |
| 5 | 0.0291 | 0.0291 |

- **Final loss:** 0.0572
- **No overfitting** — val loss monotonically decreased
- **Duration:** ~12 minutes (5 epochs, 100 steps total)
- **LoRA config:** r=32, alpha=64 (ratio 2.0), dropout=0.05
- **Unsloth note:** "Dropout = 0.05 causes a performance hit" — only affects speed, not quality
- **Train-on-responses-only:** Successfully applied with ChatML markers

### Format Verification (PASSED with fix)
- ❌ **Initial gate FAILED (0% JSON, 15% think)** — but this was a BUG in the gate!
- **Root cause:** `max_new_tokens=512` was too short (avg SFT response is ~1547 tokens)
- Responses were cut off mid-`<think>`, never reaching `</think>` or JSON
- **After fix (max_new_tokens=2048):** 3/3 samples showed `<think>=True, json=True` ✅
- **Lesson:** ALWAYS set generation max_new_tokens ≥ 2× expected response length

### GRPO Attempts — ALL FAILED (3 separate Unsloth bugs)

**Bug 1: Completion mask mismatch** (lines 3687-3698 in UnslothGRPOTrainer.py)
- Error: `RuntimeError: The size of tensor a (3388) must match the size of tensor b (3000)`
- Cause: `completion_mask` sized to `max_completion_length` but per-token tensors include prompt tokens
- Unsloth's `masked_batch_mean(x)` does `(x * completion_mask)` — shapes don't match
- Fix attempts: Set max_completion_length=max_seq_length → triggered Bug 2

**Bug 2: Model forward attention truncation** (line 958 in llama.py)
- Error: `RuntimeError: The size of tensor a (6829) must match the size of tensor b (6144)`
- Cause: `attention_mask = attention_mask[:, : self.max_seq_length]` but `inputs_embeds` exceeds it
- When max_completion_length=max_seq_length, prompt+completion > max_seq_length → crash
- Fix: Patched `inner_model.max_seq_length = 8192` → but this triggered Bug 3 (even bigger sizes)

**Bug 3: Same as Bug 2 but at 8192** → error at 8877 tokens
- Setting max_seq_length=8192 + max_completion_length=8192 meant model generated up to 8192 new tokens → prompt(800)+completion(8192) > 8192

**Monkey-patch solution (WORKED for the mask bug):**
```python
# Scan ALL 2D tensors in inputs, pad completion_mask to match largest
target_len = max(v.shape[1] for v in inputs.values() if isinstance(v, torch.Tensor) and v.ndim == 2)
if target_len > mask.shape[1]:
    pad = torch.zeros(mask.shape[0], target_len - mask.shape[1], device=mask.device, dtype=mask.dtype)
    inputs["completion_mask"] = torch.cat([pad, mask], dim=1)
```
- Debug output confirmed: `ref_per_token_logps: shape=(6, 3388)` was the largest tensor
- **Key insight:** The inputs dict contains `completion_ids`, `completion_mask`, `prompt_ids`, `prompt_mask`, `ref_per_token_logps`, `advantages`, `max_left_pad`, `num_items_in_batch`

**Bug 4: KL divergence explosion (training instability)**
- Even with monkey-patch working, GRPO destabilized after ~12-15 steps every time
- **Attempt 1:** beta=0.001 → KL=478,944 at step 5, then 985,872 at step 10 (should be <50)
- **Attempt 2:** beta=0.04 → KL=4.8 at step 5 (good!), but 704,436 at step 15 (exploded)
- **Attempt 3:** Not attempted — decided to skip GRPO

**Root cause analysis:** High reward variance + low KL penalty → model finds degenerate completions that game rewards → massive policy update → catastrophic divergence. The task is too complex for stable GRPO with current reward functions.

### Quick Inference Test (SFT-only model)
```
Output: [{"itemset": ["bread", "milk", "eggs"], "count": 3, "rows": ["Row 1", "Row 4", "Row 5"]}, 
         {"itemset": ["bread", "eggs"], "count": 4, "rows": ["Row 1", "Row 4", "Row 5", "Row 2"]}]
```
- ✅ Valid JSON (2 itemsets)
- ❌ No `<think>` tags in output (despite SFT training with think)
- ❌ No `col:val` format (outputs bare "bread" instead of "bread:yes")
- ⚠️ Count accuracy questionable ("bread, eggs" has count 4 but only 3 rows with both)

### Model pushed to HuggingFace
- ✅ `OliverSlivka/qwen2.5-7b-itemset-extractor` — 5.56 GB merged 4-bit
- ⚠️ Method: `merged_4bit_forced` (same as v1 — may have rounding errors)
- ⚠️ No GRPO refinement applied — SFT-only weights

### Configuration Used (v2)
```python
CONFIG = {
    "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "max_seq_length": 8192,  # patched from 6144 during session
    "lora_r": 32, "lora_alpha": 64,  # ratio 2.0
    "lora_dropout": 0.05,
    "sft_epochs": 5, "sft_lr": 1e-4,
    "sft_batch_size": 2, "sft_grad_accum": 8,
    # GRPO: ABANDONED
}
```

### ⚠️ CRITICAL LESSONS FOR v3
1. **Format gate must use max_new_tokens ≥ 2048** (not 512!)
2. **GRPO + Unsloth 2026.2.1 is BROKEN** — at least 3 tensor size bugs + KL instability
3. **Consider using TRL's GRPOTrainer WITHOUT Unsloth** for GRPO phase
4. **Or skip GRPO entirely** — SFT produces valid JSON, just needs better prompting
5. **`merged_4bit_forced` after SFT may lose `<think>` capability** — need to verify
6. **The model doesn't produce `col:val` format at inference** despite training data having it — possible tokenization issue or system prompt mismatch
7. **The model doesn't produce `<think>` tags at inference** — merged_4bit_forced might be destroying these, OR the inference prompt needs to explicitly request it

### Recommendations for v3
- Option A: **SFT only, more epochs (7-10)** with strict format enforcement
- Option B: **SFT + DPO** (re-enable DPO but with HARDER negatives, not easy failures)
- Option C: **SFT + GRPO using vanilla TRL** (bypass Unsloth's broken GRPO wrapper)
- Option D: **SFT + rejection sampling** (generate multiple, pick best by F1)
- **MUST investigate:** Why does inference output lack `<think>` and `col:val`? Check system prompt alignment.

**Tags:** #v2 #training-run #sft-only #grpo-failed #unsloth-bugs #pushed

See also: [[Experiments/2026-03-07 Qwen2.5-7B v2]]

---

## [2026-03-08] 🔬 Adapter-Only Eval — Raw Capture Diagnostic (15 datasets)

**Context:** Council confirmed `merged_4bit_forced` was the problem. Loaded LoRA adapters via `FastLanguageModel.from_pretrained("./sft_checkpoint")` — 392 LoRA parameters confirmed loaded across all 7 target modules. Ran raw capture eval with `max_new_tokens=8192` on 15 datasets sorted by complexity.

### Results Summary

| Dataset | Size | Apriori | Tokens | Limit? | Think | JSON | F1 |
|---------|------|---------|--------|--------|-------|------|----|
| ds_0280 (10×3) | Small | 11 | 2348 | ❌ | ✅ | ✅ | **78.6%** |
| ds_0126 (7×12) | Med | 31 | 2862 | ❌ | ✅ | ✅ | **33.8%** |
| 13 others | Various | 0-349 | 8192 | ⚠️ ALL | Mixed | ❌ | **0%** |

**Only 2/15 finished naturally. 13/15 hit 8192 token limit.**

### 🔴 ROOT CAUSE: Repetition Loops (NOT token limits)

Even 8192 tokens isn't enough — the model enters **degenerate repetition** in 3 distinct patterns:

**Pattern 1 — `<think>` repetition:** Same itemset line repeated 20+ times (ds_0347: `["carinsurance:0", "carloan:0", "default:0"]` repeated endlessly in triples section)

**Pattern 2 — JSON row spam:** `"Row 11", "Row 11", "Row 11"...` repeated HUNDREDS of times in the JSON array (ds_0303: first itemset has 1000+ "Row 11" entries)

**Pattern 3 — Hallucinated numbering:** Model invents items counting up forever: "famrel:52", "famrel:53", "famrel:54"... (ds_0072: generates infinite numbered variants of real items)

### 💡 IMMEDIATE FIX: Add `repetition_penalty=1.3` to generation

The model has NO `repetition_penalty` in its generation config. This is the #1 fix needed:
- Without it: model repeats forever → fills token limit → no JSON → 0% F1
- With it: model should stop repeating and move on to JSON output
- The 2 successful datasets prove the model CAN produce correct `<think>` + JSON when it doesn't loop

### Quality Assessment (when model doesn't repeat)

**ds_0280 (10×3, F1=78.6%):**
- ✅ Correct `<think>` structure (singles → pairs → triples)
- ✅ Valid JSON array with all 11 ground truth itemsets found (recall=100%)
- ⚠️ Over-generates: 17 unique model itemsets vs 11 apriori (precision=64.7%)
- ⚠️ Garbled row refs: "Row 22" (should be "Row 2"), "Row e6" (should be "Row 6")
- ⚠️ Hallucinated counts: some counts don't match row list length

**ds_0126 (7×12, F1=33.8%):**
- ✅ Correct structure and valid JSON
- ⚠️ Low precision (30%) — finds too many non-existent pairs/triples
- ⚠️ Low recall (38.7%) — misses many real itemsets

### Confirmed: Council was RIGHT about merge

| Evidence | Detail |
|----------|--------|
| 392 LoRA params loaded | All 7 target modules across 28 layers |
| `<think>` works | Model produces structured reasoning when it finishes |
| JSON works | Valid JSON arrays with correct schema |
| F1=78.6% on small | Proves SFT training succeeded |
| Merged model F1=0.4% | `merged_4bit_forced` destroyed everything |

### ⛔ CRITICAL: `repetition_penalty` must be added to ALL inference/eval configs

**Standard generation config for this model:**
```python
model.generate(
    max_new_tokens=4096,
    temperature=0.1,
    repetition_penalty=1.3,    # ← ESSENTIAL — prevents infinite loops
    do_sample=True,
)
```

### Next Steps (from before reppenalty test)
1. ~~Re-run eval with `repetition_penalty=1.3` — expect F1 to jump significantly~~ → **DONE, made things WORSE (see below)**
2. ~~If F1 ≥ 40%: merge was the sole problem → fix merge method (16-bit) → deploy~~
3. ✅ If F1 < 40% even with repetition fix: need v3 training with more data + explicit `col:val` prompt → **CONFIRMED**

**Tags:** #adapter-eval #repetition-bug #diagnostic #raw-capture

---

## [2026-03-08] 🔴 Repetition Penalty Experiment — CATASTROPHIC FAILURE

**Config:** `repetition_penalty=1.3`, `no_repeat_ngram_size=3`, `max_new_tokens=4096`, 30 datasets

### Results: 0% F1 Across ALL 30 Datasets

| Metric | Without Penalty (prev) | With Penalty (this run) |
|--------|------------------------|------------------------|
| Datasets | 15 | 30 |
| Completed (no limit) | 2/15 (13%) | 27/30 (90%) ✅ |
| Parse rate | 13% (2/15) | **0% (0/30)** ❌ |
| Avg F1 | 7.5% (78.6% + 33.8% on 2 successes) | **0.0%** |
| Think rate | mixed | **93%** |
| Hit limit | 13/15 (87%) | 3/30 (10%) ✅ |

### ⛔ ROOT CAUSE: `no_repeat_ngram_size=3` Destroys Structured Output

**What happened:** The `no_repeat_ngram_size=3` parameter prevents ANY sequence of 3 tokens from appearing twice in the entire output. For structured output (JSON, `<think>` templates), this is **fatal** because:

- `"Row 1", "Row 2"` → blocked: `", "Row` trigram repeats
- `{"itemset":` → blocked after first occurrence
- `"count":` → blocked after first occurrence
- Every JSON key repeats by design → ALL blocked

**Model's compensation:** Garbled text with:
- Unicode substitutions: `₀`, `⁰`, `¹` instead of `0`, `0`, `1`
- Random underscores: `_row_1+row_2+row__3+r…ow_`
- Misspelled keys: `"	itemst"`, `"	coun"`, `"rowss"`, `"	iemset"`
- Progressive nesting/degradation: increasingly corrupted text deeper in output
- Chinese/math characters: `三 ✓`, `四 ✓`, `∓∓`

**ds_0280 (previously F1=78.6%):** Now 0% — output is `_row_1+row_2+row__3+r…ow_ _six+_r ow_ s_e_v e n` gibberish

**ds_0333 (9×4):** Model enters infinite nested indentation spiral, creating deeper and deeper garbled items

### 🎯 KEY INSIGHT: This is NOT an Inference Config Problem

| Approach | Result | Why |
|----------|--------|-----|
| No penalty | 2/15 work, 13/15 repeat forever | Model doesn't know when to stop |
| `repetition_penalty=1.3` + `no_repeat_ngram_size=3` | 0/30 work, output garbled | Penalties destroy structured format |
| `repetition_penalty` alone (lighter, e.g. 1.1) | **Not tested** | Might help marginally but won't fix root cause |

**The model is fundamentally undertrained:**
- 314 SFT examples is too few for reliable termination behavior
- SFT-only (no DPO/GRPO) means no preference signal for "complete good output" vs "degenerate loop"
- The model has learned the FORMAT but not TERMINATION — knows to list items but not when to stop
- On small/simple datasets (10×3) it sometimes finishes naturally, on anything larger it loops

### ⛔ CRITICAL RULES (Updated)

1. **NEVER use `no_repeat_ngram_size` for structured output tasks** — it destroys JSON/template generation
2. **`repetition_penalty` alone won't fix undertrained models** — you can't inference-param your way out of insufficient training
3. **The v2 SFT adapter works on small datasets** — F1=78.6% on 10×3 proves training succeeded partially
4. **The real fix is v3 training** with more data, DPO phase, and possibly GRPO

### 📊 Model Capability Assessment (Final)

| Capability | Status | Evidence |
|------------|--------|----------|
| `<think>` structure | ✅ Learned | 93% think rate even with penalties |
| JSON schema | ✅ Learned (fragile) | Produces valid JSON on small datasets |
| col:val format | ⚠️ Partially | Some garbling even without penalties |
| Correct counting | ⚠️ Weak | Count mismatches on successful outputs |
| Termination | ❌ Not learned | Loops on 87% of datasets without penalties |
| Large datasets | ❌ Fails | Only works on ≤10×3 complexity |

### 📋 DEFINITIVE PLAN FOR v3

The adapter eval is DONE. We now have full diagnostic data. The path forward:

1. **Consult LLM Council** with both raw capture + reppenalty results
2. **v3 training needs:**
   - 1000+ SFT examples (up from 314) — more diverse output lengths
   - DPO with real failures (already have 606 pairs) — teaches termination
   - Skip GRPO (Unsloth bugs) or use rejection sampling
   - System prompt with explicit `col:val` mention
   - r=64, alpha=128, lr=5e-5
   - `save_method="merged_16bit"` — NEVER `merged_4bit_forced`
3. **Alternative: Try lighter `repetition_penalty=1.05-1.1` without `no_repeat_ngram_size`** as a quick test before v3

**Tags:** #reppenalty-experiment #catastrophic #no-repeat-ngram #v3-planning #council-needed

---

## [2026-03-07] ⚠️ TLJH GPU 0 OOM at unsloth import

**Context:** Cell 7 (`from unsloth import FastLanguageModel`) crashed with `AcceleratorError: CUDA error: out of memory`. GPU 0 had only 238 MiB free (another user's process), while GPUs 1-3 had ~98-99 GB free each.  
**Root cause:** `unsloth_zoo.__init__` runs `torch.cuda.memory.mem_get_info(0)` at import time. If GPU 0 is full, this crashes before the model even loads.  
**Fix applied to v2 notebook:**
1. **Cell 3:** Auto-select GPU with most free VRAM via `nvidia-smi` subprocess, set `CUDA_VISIBLE_DEVICES` **before** `import torch`
2. **Cell 7:** Added `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.reset_peak_memory_stats()` before `from unsloth import`
**Lesson:** On shared TLJH servers, NEVER assume GPU 0 is available. Always auto-detect.  
**Tags:** #tljh #oom #gpu #critical

---

## [2026-03-07] 🔮 LLM Council Analysis — v2 Post-Mortem

**Council:** claude-sonnet-4.6, gemini-3-flash-preview, deepseek-v3.2, grok-4.1-fast  
**Chairman:** claude-opus-4.6  
**Rankings:** Sonnet (1.00) > Grok (2.25) > Gemini (2.75) > DeepSeek (4.00)  
**Full report:** `docs/reports/council_v2_eval_2026-03-07.json`

### 🚨 UNANIMOUS ROOT CAUSE: `merged_4bit_forced` Destroys LoRA Deltas

The SFT training **SUCCEEDED** (format gate passed 3/3 before merge). The merge process:
1. Dequantizes 4-bit NF4 to bf16 (introduces reconstruction error)
2. Adds LoRA deltas (W_merged = W_base + α/r · A·B)
3. Re-quantizes back to 4-bit NF4 (rounds LoRA deltas to zero!)

**Result:** Double quantization error exceeds LoRA signal. Format behaviors (`<think>`, `col:val`) are small-magnitude token probability shifts — exactly what 4-bit re-quantization destroys.

### ⛔ CRITICAL RULE: NEVER USE `merged_4bit_forced` for behavioral fine-tuning

**Correct deployment options:**
1. **LoRA adapters separate** (no merge, load on base) — best for evaluation
2. **16-bit merge** (`save_method="merged_16bit"`) → optionally GPTQ/AWQ quantize after
3. **16-bit merge → GGUF Q4_K_M** for llama.cpp deployment

### 📋 IMMEDIATE NEXT STEP (Non-negotiable)

**Run adapter-only evaluation BEFORE any retraining:**
```python
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", ...)
model = PeftModel.from_pretrained(base_model, "path/to/v2/lora/checkpoint")
# Evaluate on 10 test datasets — expect F1 > 40% if merge was the only problem
```

**Decision gate after diagnostic:**
- Adapter F1 > 40%: Merge was the problem → fix merge method → done
- Adapter F1 < 10%, col:val=✅: Counting wrong → need more data (1000+)
- Adapter F1 < 10%, col:val=❌: CoT data quality issue → regenerate

### 📐 v3 Training Config (Council Consensus)

```python
v3_config = {
    "lora_r": 64, "lora_alpha": 128,  # ratio 2.0, more capacity
    "lr": 5e-5,                        # lower than v2's 1e-4
    "epochs": 3,                       # fewer epochs, more data
    "dataset_size": 1000,              # up from 314
    "save_method": "merged_16bit",     # NEVER merged_4bit_forced
}
```

### 📝 System Prompt Must Explicitly State `col:val` Format

Current prompt NEVER mentions `col:val`. Add explicit instruction:
> "Every item MUST be an exact column:value string from the CSV cells (e.g., age:15, medu:4). NEVER output bare column names."

### 🔄 GRPO: Skip for v3, Use Rejection Sampling Instead

Council unanimously says:
- Unsloth GRPO is broken (3 tensor bugs + KL explosion)
- Use **rejection sampling** for v4 (generate 8 completions, score by F1, retrain on top-k)
- Or use vanilla TRL GRPOTrainer (without Unsloth) only after SFT achieves >50% F1

### 📊 Expected v3 Outcomes (Council Estimate)

| Metric | v2 Actual | v3 Conservative | v3 Optimistic |
|--------|-----------|-----------------|---------------|
| F1     | 0.4%      | 40-60%          | 70-85%        |
| Think  | 0%        | 90%+            | 98%+          |
| col:val| 0%        | 95%+            | 99%+          |
| Parse  | 80%       | 95%+            | 99%+          |
| Halluc | 28%       | 5-10%           | <5%           |

**Tags:** #council #v2-postmortem #merged-4bit-forced #critical #v3-planning

See also: [[Experiments/2026-03-07 Qwen2.5-7B v2]]

---

## [2026-03-09] 🏛️ LLM Council v3 Plan — Adapter Eval + Repetition Loop Fix

**Context:** After v2 adapter evaluation (2026-03-08) showed 78.6% F1 on small datasets but repetition loops on 87% of datasets, ran full LLM Council session with 3 frontier models (Gemini-3-Flash, DeepSeek-v3.2, Grok-4.1-Fast) to decide v3 strategy.

**Council Report:** `docs/reports/council_v3_plan_2026-03-09.json`
**Raw responses:** `docs/reports/council_v3_partial.json`

### Root Cause (UNANIMOUS)

The repetition loop is caused by 3 factors:
1. **(c) CoT format too repetitive** (60%): Rigid `- col:val: N rows → [Row ...] ✓` pattern creates a local attractor
2. **(b) Over-training** (25%): 5 epochs on 348 examples = pattern memorization
3. **(a) Insufficient data diversity** (15%): No termination examples

### v3 Training Config (UPDATED Council Consensus)

```python
v3_config = {
    "lora_r": 32, "lora_alpha": 64,   # r DOWN from 64, alpha ratio 2.0
    "lora_dropout": 0.1,               # NEW regularization
    "sft_lr": 1e-4, "sft_epochs": 3,   # LR down from 2e-4, epochs down from 5
    "max_seq_length": 2048,             # DOWN from 4096
    "dataset_size": 1000,               # UP from 348
    "dpo_lr": 5e-5, "dpo_beta": 0.1, "dpo_epochs": 1,  # NEW DPO phase
    "grpo": "SKIP",
    "save_method": "adapter_only",      # NEVER merge_4bit_forced
}
```

### Critical CoT Format Change

Remove evidence_rows from `<think>` — only counts. Move evidence to JSON. Cuts tokens 1547→700.

### Data Plan: 1000 examples

400 concise CoT + 200 termination + 200 short/direct + 100 edge cases + 100 negative (for DPO)

### Immediate Actions

1. Test v2 adapter with inference fixes (StoppingCriteria at `</think>` + temp=0.3 + top_k=50)
2. Generate concise CoT training data (1000 examples)
3. SFT 3 epochs → DPO 1 epoch → eval

### 🔴 Claude's Late Response — Critical Additions (incorporated 2026-03-09)

Claude-sonnet-4.6 responded after 245s with 47,667 chars — the most detailed response. Key NEW insights:

1. **Loop probability proof**: If P(continue)=0.95 per item, P(terminate over 50 items)=0.95^50≈7.7%. Small datasets work because fewer items = higher termination probability. This is a MATH problem.

2. **alpha/r ratio was fundamentally broken in v2**: alpha/r = 16/64 = 0.25 (very weak LoRA signal). Must fix to 64/32 = 2.0. This is the **#2 most impactful fix** after CoT format.

3. **Temperature 0.1 makes loops WORSE** (counterintuitive): At low temp, model is near-deterministic → locks into highest-probability loop → cannot escape. Recommend temp=0.3 (enough randomness to escape attractors, <3% F1 degradation).

4. **Two-phase generation**: Generate `<think>` with temp=0.3, then switch to temp=0.05 for JSON. StoppingCriteria at `</think>`.

5. **Evaluation gate between SFT and DPO**: Do NOT run DPO if SFT is broken. Gates: parse≥0.70, F1≥0.55, loop_rate≤0.30.

6. **DPO cannot fix broken SFT**: DPO learns preferences, but needs the base to sometimes produce correct outputs. Fix SFT FIRST.

7. **Priority actions by impact**: (1) Compact CoT +25-35% F1, (2) Fix alpha/r ratio +10-15%, (3) Reduce epochs +10-15%, (4) Temp=0.3 +15-20% (FREE), (5) Dynamic token budget +10-20% (FREE)

8. **Key quote**: "The gap between 78.6% and 7.5% mean F1 is almost entirely explained by the repetition loop. Fix the loop → fix the model."

**Updated immediate action #1**: Use temp=0.3 (not 0.05) based on Claude's analysis.
**Updated lora_dropout**: 0.05 (Claude) vs 0.1 (others). Go with 0.05.
**Updated DPO beta consideration**: Claude says 0.05, others say 0.1. Start with 0.1, reduce if instability.

**Tags:** #council #v3-plan #repetition-loop #concise-cot #claude-insights

---

## [2026-03-09] ✅ v3 Implementation — All Council Recommendations Applied

**Context:** Implemented ALL changes from the 2026-03-09 LLM Council session into the codebase. This is a comprehensive update touching training data generation, inference, evaluation, and the notebook.

### Files Modified

| File | Changes |
|------|---------|
| `src/training/training_utils.py` | New concise `generate_cot()` v3 format, `generate_cot_legacy()` preserved, `calculate_token_budget()` added |
| `src/training/generate_cot_sft_data.py` | Default output → `sft_cot_v3.json`, max_tokens → 1800 |
| `src/evaluation/inference_utils.py` | **NEW** — ThinkStoppingCriteria, RepetitionDetector, two-phase generation, dynamic budget |
| `src/evaluation/eval_finetuned_model.py` | temp 0.1→0.3, StoppingCriteria, dynamic budget, --two-phase flag, --temperature flag |
| `notebooks/training_3phase_7b.ipynb` | ALL config changes (see below) |

### Notebook v3 Config Changes

| Parameter | v2 | v3 | Rationale |
|-----------|----|----|-----------|
| `max_seq_length` | 4096 | **2048** | Concise CoT fits in 700-1200 tokens |
| `lora_r` | 64 | **32** | Smaller rank, less overfitting |
| `lora_alpha` | 16 | **64** | Alpha/r = 2.0 (was 0.25 — catastrophically weak) |
| `lora_dropout` | 0 | **0.05** | Regularization against repetition |
| `sft_epochs` | 2 | **3** | More epochs with lower LR |
| `sft_lr` | 2e-4 | **1e-4** | Gentler updates |
| `sft_warmup_ratio` | 0.05 | **0.10** | More warmup for stability |
| `sft_weight_decay` | 0 | **0.01** | L2 regularization |
| `sft_eval_strategy` | epoch | **steps (50)** | Earlier overfitting detection |
| `sft_load_best` | False | **True** | Keep best checkpoint |
| `dpo_epochs` | 2 | **1** | DPO overfits quickly with real failures |
| GRPO | 200 steps | **SKIPPED** | Unsloth GRPO broken + unnecessary |
| Save method | `merged_4bit_forced` | **Adapter-only** | CRITICAL: merge destroys LoRA |
| Inference temp | 0.1 | **0.3** | Escape attractor loops |
| Inference top_k | none | **50** | Prune long tail tokens |
| Inference top_p | none | **0.90** | Nucleus sampling |

### Concise CoT Format (v3)

**Old format (verbose, ~1547 tokens avg):**
```
## Singles (min_support=3)
- col:val: 5 rows → [Row 1, Row 3, Row 5, Row 7, Row 8] ✓
```

**New format (concise, ~700 tokens avg):**
```
## SCAN 1: Singles by column
age: 15=3(R1,R2,R7)✓ | 16=2✗ | 17=4(R3,R4,R5,R6)✓
FREQUENT SINGLES: age:15, age:17, medu:4 (3 items)
## SCAN 2: Pairs
{age:15,medu:4}: R1,R7 → 2✗
## RESULT SUMMARY: 5 singles + 3 pairs + 1 triple = 9 itemsets
```

Key changes: column-grouped (all values for a column on one line), compact notation (R1 not Row 1), RESULT SUMMARY termination signal.

### Inference Utilities (NEW module)

`src/evaluation/inference_utils.py` implements:
1. `ThinkStoppingCriteria` — stops at `</think>` token sequence
2. `RepetitionDetector` — catches repeated line patterns (3× consecutive)
3. `generate_two_phase()` — think at temp=0.3, JSON at temp=0.05
4. `generate_with_guards()` — single-phase with both stopping criteria
5. `extract_and_repair_json()` — robust JSON extraction from messy output
6. `calculate_dynamic_budget()` — dataset-specific max_new_tokens
7. `V3_INFERENCE_CONFIG` — reference config dict

### What Still Needs to Happen

1. **Regenerate SFT data** with concise CoT format: `python src/training/generate_cot_sft_data.py --db runs.db --output data/sft_cot_v3.json`
2. **Rebuild HF dataset** with v3 data
3. **Push to HF Hub**
4. **Train on TLJH server** with updated notebook
5. **Evaluate** with new inference utils (temp=0.3, StoppingCriteria)

### Expected v3 Impact (from council estimates)

| Fix | Expected Impact | Cost |
|-----|----------------|------|
| Concise CoT format | +25-35% F1 | Requires retraining |
| Fix alpha/r ratio (already done in v2) | +10-15% F1 | Already applied |
| Reduce epochs 5→3 + lower LR | +10-15% F1 | Requires retraining |
| temp=0.3 at inference | +15-20% F1 | FREE — no retraining |
| Dynamic token budget | +10-20% F1 | FREE — no retraining |
| StoppingCriteria at </think> | Prevents 87% of timeouts | FREE — no retraining |

**Conservative estimate:** 50-65% F1 (up from 7.5% mean)
**Optimistic estimate:** 75-85% F1

**Tags:** #v3 #implementation #council-applied #all-files-updated

---

## [2026-03-17] 🔬 Unsloth Notebook Mining — Validated Training Defaults

**Source:** Diamond knowledge extraction from 11 reviewed Unsloth × Qwen notebooks (`knowledge_extraction/unsloth_notebooks/notes/DIAMOND_KNOWLEDGE.md`)

### Optimizer Upgrade: `paged_adamw_8bit`

**What:** Replace `adamw_8bit` with `paged_adamw_8bit` in both SFT and DPO phases.
**Why:** Pages optimizer states to CPU when GPU RAM is tight. Strictly better — same convergence, lower peak VRAM. Used in Unsloth's own Qwen2.5-Coder notebook as the recommended optimizer.
**Evidence:** `original_template/Qwen2.5_Coder_(14B)-Conversational.ipynb` Cell 17 uses `paged_adamw_8bit` explicitly.
**Applied to:** `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 8 (SFT) and Cell 12 (DPO).

### Label Masking Verification (New Diagnostic Cell)

**What:** After creating SFTTrainer with `train_on_responses_only`, decode one sample's labels to visually confirm masking works correctly.
**Why:** Wrong `instruction_part`/`response_part` strings cause masking to fail SILENTLY — the model trains on prompts instead of just responses, wasting capacity and potentially memorizing inputs. This was observed across Qwen3-4B-Thinking and Qwen2.5-Coder notebooks.
**Pattern:**
```python
space = tokenizer(" ", add_special_tokens=False).input_ids[0]
sample = sft_trainer.train_dataset[5]
labels = sample["labels"] if "labels" in sample else sample.get("label_ids", [])
print(tokenizer.decode([space if x == -100 else x for x in labels]))
```
**Applied to:** New cell added after SFT trainer creation in notebook.

### SFT Format Verification Gate (Generate Before Proceeding)

**What:** After SFT training, generate on 2 validation samples and check: (a) `<think>` tags present, (b) JSON parseable, (c) col:val format used.
**Why:** v2 had a gate but used `max_new_tokens=512` (too short) → false negatives. Diamond knowledge confirms: ALWAYS verify format learning before proceeding to next phase. The Qwen3-4B-GRPO notebook uses this exact pattern (intermediate verification after SFT, before GRPO).
**Applied to:** Added actual generation test to notebook Cell 9 (was just a print statement).

### Validated LoRA/Training Defaults (Cross-Notebook Consensus)

These defaults appeared in 8+ of 11 reviewed notebooks — they are the canonical Unsloth configuration:

| Parameter | SFT Value | GRPO Value | Our v3 | Status |
|-----------|-----------|------------|--------|--------|
| `lora_r` | 16 | 32 | 32 | ✅ Matches GRPO default |
| `lora_alpha` | =r | =2r | 64 (=2r) | ✅ Correct |
| `lora_dropout` | 0 | 0 | 0.05 | ✅ Council override |
| `learning_rate` | 2e-4 | 5e-6 | 1e-4 | ✅ Council override |
| `optim` | adamw_8bit | adamw_8bit | **paged_adamw_8bit** | 🔄 Upgraded |
| `gradient_checkpointing` | "unsloth" | "unsloth" | "unsloth" | ✅ Matches |
| `seed` | 3407 | 3407 | 42 | ℹ️ Different (our choice) |

### Future: GRPO with GSPO (When Re-enabled)

If GRPO is re-enabled in a future version (after Unsloth bugs are fixed), use GSPO over vanilla GRPO:
```python
GRPOConfig(
    # ... standard params ...
    importance_sampling_level = "sequence",   # GSPO
    mask_truncated_completions = False,        # GSPO
    loss_type = "dr_grpo",                     # GSPO
    learning_rate = 5e-6,
    max_grad_norm = 0.1,        # Aggressive clipping for stability
    adam_beta2 = 0.99,
    weight_decay = 0.1,         # Heavy regularization
    lr_scheduler_type = "cosine",
)
```
**Source:** Both Vision-GRPO notebooks use GSPO as the current Unsloth recommendation.

### Future: Pre-Finetuning Before GRPO

If starting GRPO from scratch (or from base model), SFT on ~50-60 format examples first:
- Teaches output structure before reward optimization
- Without it, GRPO wastes hundreds of steps learning format
- Our existing SFT phase already serves this purpose — no extra step needed for our workflow

### Packing: Potential 5× Speedup (NOT Applied — Needs Testing)

`packing=True` concatenates multiple short samples to fill `max_seq_length`, reducing padding waste. Our CSV prompts are short (~500-700 tokens after concise CoT). Potential 5× throughput improvement.
**NOT applied** because: (a) v3 Council explicitly set `packing=False`, (b) packing can corrupt `train_on_responses_only` masking boundaries. Test on a small subset before enabling.

**Tags:** #diamond-knowledge #unsloth-mining #validated-defaults #paged-adamw #label-verification

---

## [2026-03-09] v3 Training Results + v3.1 merge_and_unload Fix

### v3 Training on TLJH (H200 NVL, 150 GB)

**SFT Phase:** 3 epochs, 90 steps, 3:35 duration
- Final loss: 0.0709, val loss: 0.033624
- ✅ SFT converged well

**DPO Phase:** 1 epoch, 137 steps, 5:43 duration
- Final loss: 0.0373, accuracy: 1.000 (100%)
- Rewards: chosen=8.686, rejected=-21.749, margin=30.436
- ✅ DPO converged perfectly (clear preference separation)

**Inference Test:** ❌ FAILED
- Output: `"count": 2wo` — JSON parse failure
- No `<think>` tags in output
- Root cause: **merge_and_unload() on 4-bit model** (same bug as v2!)

### Root Cause Analysis

The v3 notebook still had `merge_and_unload()` in:
- **Cell 11 (DPO setup):** Merged SFT adapter into 4-bit base before DPO
- **Cell 19 (inference):** Merged both SFT + DPO adapters into 4-bit base

This is the **exact same bug** that destroyed v2 — double quantization
(dequant→merge→requant) rounds LoRA deltas to zero.

### v3.1 Fix Applied

**Strategy: Option A — Continue training same adapter**

Cell 11 (DPO setup):
```python
# OLD (broken): model.merge_and_unload() → get_peft_model() → DPO
# NEW (v3.1): PeftModel.from_pretrained(sft_dir, is_trainable=True) → DPO
```
- SFT adapter loaded as TRAINABLE — DPO continues training same LoRA weights
- Single adapter checkpoint after DPO (contains SFT + DPO knowledge)
- DPOTrainer auto-uses base (adapter disabled) as ref_model

Cell 19 (inference):
```python
# OLD (broken): merge_and_unload(SFT) → merge_and_unload(DPO) → generate
# NEW (v3.1): PeftModel.from_pretrained(dpo_dir) → for_inference → generate
```
- DPO adapter loaded directly on base — NO merge
- Adapter computes at inference time (LoRA math preserved in float16)

**Why Option A over Option B (nested adapters):**
- Option B: Frozen SFT + fresh LoRA on top → DPO adapter alone doesn't contain SFT knowledge
- Option A: DPO continues training SFT adapter → single checkpoint has everything
- Option A is simpler, more robust, proven with DPOTrainer

### CRITICAL RULE (NEVER FORGET)
**NEVER call `merge_and_unload()` on a 4-bit NF4 model.**
- It dequantizes (4-bit→bf16), adds LoRA deltas, requantizes (bf16→4-bit)
- The requantization rounds LoRA deltas to ZERO
- Model reverts to untrained base → corrupted outputs like `"count": 2wo`
- Always use adapter-only loading: `PeftModel.from_pretrained(base, adapter_dir)`

### Next Steps
1. User re-downloads v3.1 notebook from HF Hub
2. Re-trains on TLJH (should take ~9 min again)
3. Inference test should now produce valid JSON with `<think>` tags
4. If F1 > 0.60, proceed to Stage 6 (/validate)

**Tags:** #v3 #v3.1 #fix #merge-and-unload #critical #training-results

---

## [2026-03-17] ⚠️ TLJH stale `CUDA_VISIBLE_DEVICES` caused Cell 3 Torch crash

**Context:** Cell 3 crashed on TLJH with `DeferredCudaCallError` / `device=3, num_gpus=3` while calling `torch.cuda.get_device_name(0)`. The notebook process inherited a stale GPU visibility mapping that referenced 4 GPUs even though only 3 were visible to the process.

**Root cause:** `torch` was imported before sanitizing `CUDA_VISIBLE_DEVICES`, so lazy CUDA init saw an invalid visible-device mapping and failed internally.

**Fix applied to v3 notebook:**
1. In Cell 3, query `nvidia-smi` **before** `import torch`
2. Parse the current `CUDA_VISIBLE_DEVICES`
3. If it is invalid for the GPUs visible to the process, replace it with the GPU that has the most free VRAM
4. Only then import `torch` and call `torch.cuda.init()`
5. If `torch` was already imported, raise a clear message telling the user to restart the kernel and rerun from Cell 2

**Lesson:** On shared TLJH/Jupyter environments, do not trust inherited GPU visibility. Always sanitize `CUDA_VISIBLE_DEVICES` before the first `import torch` in the notebook.

**Tags:** #tljh #gpu #cuda-visible-devices #torch #notebook #critical

---

## [2026-03-17] ⚠️ SFT gate must use deterministic decoding + explicit truncation diagnostics

**Context:** The Cell 9 SFT format gate kept reporting failures even after the JSON parser was fixed to ignore bracket counts inside `<think>`. Raw outputs showed two distinct cases:
- generations ending before `</think>`
- ambiguous warnings where it was unclear whether the model stopped early or simply hit `max_new_tokens`

**Root cause:** The gate was still using sampled decoding for a binary format check, and it did not report whether generation had actually exhausted the token budget. That made parser issues, decoding noise, and genuine model corruption look the same.

**Fix applied to v3 notebook:**
1. Switch Cell 9 gate generation to `do_sample=False` (deterministic decoding)
2. Size `max_new_tokens` from the reference validation completion length instead of using one fixed cap
3. Save per-sample metadata JSON with target length, generated length, and whether the budget was hit
4. Print a clear distinction between:
    - `generation hit token budget before </think>`
    - `model stopped early before </think>`

**Lesson:** Format gates should be deterministic and diagnostic-first. If the goal is to verify learned structure, do not use stochastic decoding as the primary pass/fail signal.

**Tags:** #sft #gate #diagnostics #decoding #notebook #v3.6

---

## [2026-03-17] 🚨 Likely root cause of malformed SFT outputs: notebook seq length too small

**Context:** The improved Cell 9 gate showed a decisive split:
- one sample hit the generation budget before closing `</think>`
- another sample closed `</think>` without hitting the budget, but still emitted malformed JSON

This ruled out the earlier "parser only" hypothesis. Separate dataset inspection then showed that many SFT examples are much longer than the notebook's `max_seq_length=2048` setting.

**Evidence:**
- local SFT dataset stats: 348 examples, avg estimated length ~1547 tokens, max 3744
- 119 examples exceed 1800 estimated tokens
- 96 examples exceed 2048 estimated tokens
- gate sample 1 expected completion alone was ~1903 tokens, which leaves no room for system + user prompt within a 2048 training window

**Implication:** With `max_seq_length=2048`, the notebook silently truncates many assistant completions during SFT. That teaches the model to produce cut-off reasoning blocks and malformed JSON endings.

**Fix applied to notebook v3.7:**
1. Restore `CONFIG["max_seq_length"] = 4096`
2. Add explicit token-length auditing in the SFT prep cell
3. Assert that zero train/validation samples exceed the configured length before training starts

**Lesson:** Never reduce notebook sequence length without auditing the actual tokenized SFT examples first. For this project, CoT + final JSON can easily exceed 2048 even when the examples look "concise".

**Tags:** #sft #truncation #seq-length #root-cause #notebook #v3.7

---

## [2026-03-17] ⚠️ Cell 9 gate can reintroduce false truncation if its completion cap ignores CONFIG

**Context:** After restoring `CONFIG["max_seq_length"] = 4096`, the gate still reported both validation samples as truncated. The reason was not the model alone: Cell 9 still computed `generation_budget = min(2048, ...)`, so the gate itself was hard-capped at 2048 tokens.

**Root cause:** The notebook's training window and gate window diverged. Even with a 4096-token model context, Cell 9 could only generate 2048 new tokens, which was insufficient for long `<think>` traces plus final JSON on some validation samples.

**Fix applied:**
1. Upgrade Cell 9 to v3.7
2. Compute `prompt_token_count` first
3. Derive `available_completion_budget` from `CONFIG["max_seq_length"] - prompt_token_count - margin`
4. Cap generation using the available context window, not a fixed 2048 ceiling
5. Persist prompt length, available budget, requested budget, and final budget to the per-sample metadata JSON

**Lesson:** Whenever `CONFIG["max_seq_length"]` changes, verify that every diagnostic/inference gate uses the same effective context budget. Otherwise the notebook can appear broken even after the training config is fixed.

**Tags:** #gate #seq-length #truncation #notebook #v3.7

---

## [2026-03-17] ✅ Notebook gate should reuse evaluation-time two-phase decoding

**Context:** Even after fixing Cell 9's budget math, the SFT gate still showed runaway reasoning loops and malformed JSON tails. The repo already had a more robust inference path in `src/evaluation/inference_utils.py`, but the notebook gate and quick inference cell were still using custom one-shot generation.

**Fix applied in notebook v3.8:**
1. Cell 9 now imports and uses `generate_two_phase()`
2. Cell 9 now parses output with `extract_and_repair_json()`
3. Cell 19 now uses the same guarded two-phase decode path as evaluation
4. The gate now measures separate think/json budgets instead of one flat completion budget

**Why this matters:** The notebook must test the adapter under the same decoding strategy used in real evaluation. Otherwise the notebook can report failures that are partly artifacts of weaker generation settings rather than the adapter alone.

**Lesson:** When an inference utility already exists in the repo, prefer reusing it in notebooks over re-implementing local generation logic.

**Tags:** #gate #inference #two-phase #notebook #evaluation #v3.8

---

## [2026-03-17] 🔬 Root Cause: R-Shorthand Tokenization Ambiguity + Dataset Size Mismatch

**Context:** SFT gate showed 0/2 pass with specific failure patterns: row hallucination (R22, R23, R24 in a 10-row dataset), value corruption (rating:55 instead of :5), duplicate evidence rows, and schema drift. Training data audit showed 0 contamination — training data was perfectly clean. Deep investigation revealed two compounding root causes.

### Root Cause 1: R-Shorthand Tokenization Ambiguity

The v3 concise CoT format used packed R-shorthand in think blocks: `R1,R10,R2,R3`. Qwen tokenizer splits this as:

| Format | Tokens | Quality |
|--------|--------|---------|
| `R1,R10,R2,R3` | `R` `1` `,R` `1` `0` `,R` `2` `,R` `3` | ❌ AMBIGUOUS — R10 shares token boundary with R1 |
| `R1, R10, R2, R3` | `R` `1` `,` `ĠR` `1` `0` `,` `ĠR` `2` `,` `ĠR` `3` | ✅ CLEAN — each R-ref starts with ĠR token |

The packed format had NO clean boundary between R10 and the preceding comma — the model saw `R` `1` then `,R` `1` `0` and couldn't distinguish whether digits belong to the current or next R-ref. This directly caused:
- **Row hallucination**: R2 → R22, R3 → R23, R4 → R24 (digit concatenation)
- **Value corruption**: `rating:5` → `rating:55` (same digit-doubling mechanism)
- **48% of training examples** (112/233) had adjacent single+double digit R-refs, teaching the model this ambiguous pattern

### Root Cause 2: HF Dataset Size Mismatch (26% data loss)

| Source | Examples | Format |
|--------|----------|--------|
| `sft_cot_v2.json` (local, verbose) | **348** | `Row N` format (clean tokenization) |
| `sft_cot_v3.json` (local, concise) | **258** | `R-shorthand` packed (ambiguous) |
| HF Hub (what model trained on) | **258** (233+25) | Built from v3, NOT v2 |

The HF dataset was built from `sft_cot_v3.json` (258 examples, max_tokens=1800) not `sft_cot_v2.json` (348 examples, max_tokens=3500). The model trained on 26% fewer examples than available, AND those examples used the ambiguous R-shorthand.

### Fix Applied (v3.10)

1. **Spaced R-refs in `generate_cot()`**: Changed `",".join(row_nums)` to `", ".join(row_nums)` in `src/training/training_utils.py` (singles, pairs, triples sections). Also added space before `+Nmore` suffix.

2. **Increased max_tokens**: Default filter raised from 1800 → 3500 in `generate_cot_sft_data.py` to capture more examples.

3. **Added tokenizer-based filter**: New `scripts/filter_sft_by_tokens.py` uses actual Qwen tokenizer to verify every example ≤ 4096 tokens (chars/4 estimate was off by ~2×).

4. **Regenerated + rebuilt + pushed**:
   - `sft_cot_v3.json`: 334 → filtered to **272** examples (all ≤ 4096 actual tokens)
   - HF dataset: SFT 245 train / 27 val (up from 233/25)
   - Notebook bumped to **v3.10**
   - All pushed to HF Hub

### Key Numbers: Before vs After

| Metric | Before (v3.9) | After (v3.10) |
|--------|---------------|---------------|
| SFT examples | 258 (233+25) | **272** (245+27) |
| Packed R-refs (ambiguous) | 258/258 (100%) | **0/272 (0%)** |
| Spaced R-refs (clean) | 0/258 (0%) | **272/272 (100%)** |
| Examples > 4096 tokens | Unknown (no audit) | **0/272 (verified)** |
| Token length range | 497–3820 | **503–4070** |

### ⚠️ CRITICAL RULE: Never Use Packed R-Shorthand

**ALWAYS** use spaced delimiters in structured data that will be tokenized:
- ✅ `R1, R10, R2, R3` (comma + space = clean token boundary)
- ❌ `R1,R10,R2,R3` (no space = ambiguous boundary for subword tokenizers)

This applies to ANY delimiter in training data where adjacent values could share digit boundaries.

### ⚠️ CRITICAL RULE: Always Filter SFT Data by Actual Token Count

The `estimate_tokens()` function (chars/4) underestimates by ~2× for this content. **Always run `scripts/filter_sft_by_tokens.py`** after generating SFT data to ensure NO examples exceed `max_seq_length`.

### Next Steps
1. User re-downloads v3.10 notebook from HF Hub
2. Re-trains on TLJH (SFT on 245 examples with clean tokenization)
3. Expect SFT gate to show improved row references (no R22/R23 hallucination)
4. If gate passes, proceed to DPO phase

**Tags:** #v3.10 #tokenization #r-shorthand #root-cause #fix #dataset-size-mismatch

---

## [2026-03-18] ✅ Proper HF Versioning — Separate Repos for v2 and v3

**Context:** Previously, every rebuild of the HF dataset overwrote the same repo (`OliverSlivka/itemset-extraction-v2`), destroying the training history. After v3 data was pushed to the v2 repo, the original v2 training data was lost from HF.

### Problem

| Time | Repo content | Issue |
|------|-------------|-------|
| 2026-03-07 | v2 data (348 SFT, verbose "Row N") | ✅ Correct |
| 2026-03-09 | v3 data (258 SFT, packed R-refs) | ❌ Overwrote v2! |
| 2026-03-17 | v3.10 data (272 SFT, spaced R-refs) | ❌ Still overwriting v2! |

### Fix Applied (v3.11)

Created separate HF repos — each version is now FROZEN:

| Repo | SFT (train/val) | Format | Notebook |
|------|-----------------|--------|----------|
| `OliverSlivka/itemset-extraction-v2` | 314/34 | Verbose `Row N` | `training_3phase_2026-03-07_v2.ipynb` |
| `OliverSlivka/itemset-extraction-v3` | 245/27 | Concise spaced `R1, R10` | `training_3phase_2026-03-09_v3.ipynb` (v3.11) |

Both repos share DPO (546/60) and GRPO (matching SFT split).

### Files Changed

| File | Change |
|------|--------|
| `notebooks/training_3phase_2026-03-09_v3.ipynb` | CONFIG `hf_dataset` → `OliverSlivka/itemset-extraction-v3`, version → v3.11 |
| `scripts/push_versioned_datasets.py` | NEW — restores v2 + creates v3 in one script |
| `scripts/verify_hf_repos.py` | NEW — verifies both repos by downloading from Hub |

### Verified from Hub

- v2: SFT train=314, val=34, format has `Row N` ✅
- v3: SFT train=245, val=27, format has spaced R-refs ✅
- Both repos have DPO 546/60, GRPO matching SFT

### ⚠️ CRITICAL RULE: Never Overwrite Previous Version Repos

For future versions (v4, v5, ...):
1. Create NEW repo: `OliverSlivka/itemset-extraction-v{N}`
2. Push data + notebook to new repo
3. Update notebook CONFIG to point to new repo
4. Previous repos remain FROZEN — never overwrite them

**Tags:** #v3.11 #versioning #hf-repos #critical-rule #frozen-versions
