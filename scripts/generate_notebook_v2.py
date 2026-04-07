#!/usr/bin/env python3
"""Generate the corrected v2 training notebook (SFT→GRPO, no DPO).

Incorporates ALL LLM Council fixes from 2026-03-01:
- LoRA alpha/r = 64/32 = 2.0 (was 0.25)
- DPO removed entirely
- SFT 5 epochs, lr=1e-4, packing=False
- 5 GRPO reward functions with partial credit
- Format verification gate
- Save adapters only between phases
- max_seq_length=6144
"""

import json, sys
from pathlib import Path

def make_cell(cell_type, source, **kwargs):
    """Create a notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else source.split("\n"),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def md(source):
    # Split into lines, add \n to all but last
    lines = source.split("\n")
    result = [l + "\n" for l in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    else:
        # trailing newline already handled
        pass
    return make_cell("markdown", result)

def code(source):
    lines = source.split("\n")
    result = [l + "\n" for l in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return make_cell("code", result)


cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 1: Markdown Title
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
# 2-Phase Frequent Itemset Extractor Training v2 (Qwen 2.5-7B)

**Pipeline:** SFT-CoT (5 epochs) → GRPO with 5 Reward Functions
**Date:** 2026-03-07 | **Version:** v2 (council-corrected)

## Changes from v1 (catastrophic failure — 0% F1)

| Issue | v1 (broken) | v2 (fixed) |
|-------|-------------|------------|
| LoRA alpha/r ratio | 16/64 = **0.25** | 64/32 = **2.0** |
| DPO phase | Present (killed `<think>`) | **REMOVED** |
| SFT epochs | 2 | **5** |
| SFT learning rate | 2e-4 | **1e-4** |
| Packing | Default (True) | **False** |
| Save between phases | `merged_4bit_forced` | **Adapters only** |
| GRPO rewards | 4 (broken, F1≈0) | **5 (partial credit)** |
| GRPO steps | 200 | **500** |
| Format gate | None | **Added** |
| max_seq_length | 4096 | **6144** |
| weight_decay | None | **0.01** |
| max_grad_norm | None | **1.0** |

## Root causes (LLM Council — 4/4 unanimous)
1. LoRA ratio 0.25 → weights barely updated during training
2. DPO destroyed `<think>` reasoning via contrastive loss
3. GRPO rewards gave zero signal (no normalization, no partial credit)
4. `merged_4bit_forced` between phases caused quantization cascade

## Phases
1. **SFT-CoT** (5 epochs) — Learn `<think>` reasoning + JSON with `col:val` items
2. **GRPO** (500 steps) — 5 reward functions: structure, col:val, F1, recall, grounding"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 2: Install dependencies
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 1: Install dependencies (run ONCE, then restart kernel) ──────────────
# ⚠️ TLJH server: installing unsloth pulls torch to ~/.local which SHADOWS
#    system torch. Fix: install → delete user-level torch/nvidia → restart.

import os, glob, shutil, subprocess

USER_SP = os.path.expanduser("~/.local/lib/python3.12/site-packages")

# Step 1: Install if not present
unsloth_dir = os.path.join(USER_SP, "unsloth")
if not os.path.isdir(unsloth_dir):
    print("📦 First run — installing ML packages...")
    subprocess.check_call(
        "pip install unsloth trl datasets transformers accelerate "
        "bitsandbytes huggingface_hub peft safetensors sentencepiece protobuf -q".split()
    )
    print("📦 Install complete.")
else:
    print("✅ Packages already installed — skipping pip install")

# Step 2: Remove ONLY core torch + nvidia (keep torchvision/torchaudio)
removed = []
for pattern in ["torch", "torch-*"]:
    for p in glob.glob(os.path.join(USER_SP, pattern)):
        basename = os.path.basename(p)
        if basename.startswith("torchvision") or basename.startswith("torchaudio"):
            continue
        shutil.rmtree(p, ignore_errors=True)
        removed.append(basename)

for p in glob.glob(os.path.join(USER_SP, "nvidia*")):
    shutil.rmtree(p, ignore_errors=True)
    removed.append(os.path.basename(p))

if removed:
    print(f"🗑️  Cleaned from ~/.local: {removed}")
else:
    print("✅ No user-level torch/nvidia to clean")

# Step 3: Verify system torch
assert not os.path.exists(os.path.join(USER_SP, "torch")), \\
    f"❌ FAILED: torch still in {USER_SP}/torch"

r = subprocess.run(
    ["python3", "-c", "import torch; print(torch.__version__, torch.__file__)"],
    capture_output=True, text=True
)
if r.returncode == 0:
    print(f"✅ System torch: {r.stdout.strip()}")
else:
    print(f"⚠️  System torch not found: {r.stderr.strip()[:200]}")

print("\\n" + "=" * 60)
print("⚠️  RESTART THE KERNEL (Kernel → Restart)")
print("   Then run Cell 2 (CONFIG) onwards.")
print("=" * 60)"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 3: CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 2: CONFIG — edit this cell only ─────────────────────────────────────
# v2 corrected per LLM Council analysis (2026-03-01)
# Key fixes: LoRA ratio 2.0 (was 0.25), no DPO, 5 SFT epochs, 5 reward funcs

CONFIG = {
    # ── Model ────────────────────────────────────────────────────────────────
    "base_model":        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "max_seq_length":    6144,      # was 4096 — long CSVs + full answers
    "load_in_4bit":      True,

    # ── Dataset (HuggingFace Hub) ─────────────────────────────────────────────
    "hf_dataset":        "OliverSlivka/itemset-extraction-v2",
    "hf_token":          "",    # paste HF token here, or set env HF_TOKEN

    # ── LoRA — FIXED (alpha/r = 2.0, was 0.25 in v1) ────────────────────────
    "lora_r":            32,        # was 64
    "lora_alpha":        64,        # was 16 → ratio 0.25→2.0 (CRITICAL FIX)
    "lora_dropout":      0.05,      # was 0 — helps generalization
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],

    # ── Phase 1: SFT with CoT (5 epochs, lower LR) ──────────────────────────
    "sft_epochs":        5,         # was 2 — need format memorization
    "sft_lr":            1e-4,      # was 2e-4 — lower for stability
    "sft_batch_size":    2,         # reduce to 1 if OOM
    "sft_grad_accum":    8,         # was 4 — effective batch = 16
    "sft_output_dir":    "./sft_checkpoint",

    # ── Phase 2: GRPO — NO DPO (council: 4/4 unanimous) ─────────────────────
    "grpo_max_steps":    500,       # was 200
    "grpo_lr":           2e-6,      # was 5e-6 — lower for RL stability
    "grpo_batch_size":   1,
    "grpo_grad_accum":   8,         # was 4
    "grpo_num_generations": 6,      # was 4 — more samples = better variance
    "grpo_max_completion_length": 3000,  # was 2048 — allow full answers
    "grpo_beta":         0.001,     # KL penalty (low = allow exploration)
    "grpo_temperature":  0.7,
    "grpo_output_dir":   "./grpo_checkpoint",

    # ── Output / Push ─────────────────────────────────────────────────────────
    "hf_model_repo":     "OliverSlivka/qwen2.5-7b-itemset-extractor",
    "push_to_hub":       True,
}

print("✅ CONFIG loaded (v2 — council-corrected)")
print(f"   Model: {CONFIG['base_model']}")
print(f"   Seq length: {CONFIG['max_seq_length']}")
print(f"   LoRA: r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']} (ratio={CONFIG['lora_alpha']/CONFIG['lora_r']:.1f})")
print(f"   SFT: {CONFIG['sft_epochs']} epochs, lr={CONFIG['sft_lr']}")
print(f"   GRPO: {CONFIG['grpo_max_steps']} steps, lr={CONFIG['grpo_lr']}")
print(f"   DPO: REMOVED (council-recommended)")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 4: GPU check + imports
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 3: GPU check + imports ───────────────────────────────────────────────
import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gc, json, re, torch
from datasets import load_dataset
from huggingface_hub import login

# HF login
hf_token = CONFIG["hf_token"] or os.environ.get("HF_TOKEN", "")
if hf_token:
    login(token=hf_token)
    print("✅ HuggingFace logged in")
else:
    print("⚠️  No HF token — set hf_token in CONFIG or run `huggingface-cli login`")
    try:
        login()
    except Exception:
        pass

# GPU info
if torch.cuda.is_available():
    print(f"\\n✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    !nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else:
    raise RuntimeError("❌ No GPU found — connect a GPU runtime.")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 5: Load datasets
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 4: Load datasets from HuggingFace ────────────────────────────────────
# v2: Only SFT + GRPO (DPO removed)
print("📂 Loading datasets from HuggingFace Hub...")

sft_dataset  = load_dataset(CONFIG["hf_dataset"], "sft")
grpo_dataset = load_dataset(CONFIG["hf_dataset"], "grpo")

print(f"✅ SFT:  {len(sft_dataset['train']):>4d} train / {len(sft_dataset['validation']):>3d} val")
print(f"✅ GRPO: {len(grpo_dataset['train']):>4d} train / {len(grpo_dataset['validation']):>3d} val")
print(f"ℹ️  DPO: SKIPPED (removed per council recommendation)")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 6: Data preview
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 5: Quick data preview ────────────────────────────────────────────────
example = sft_dataset["train"][0]
print("═" * 60)
print("SFT Example (messages format):")
print("═" * 60)
for msg in example["messages"]:
    role = msg["role"].upper()
    content = msg["content"]
    if len(content) > 300:
        content = content[:300] + f"... ({len(content)} chars total)"
    print(f"\\n[{role}]")
    print(content)

print("\\n" + "═" * 60)
print("GRPO Example:")
print("═" * 60)
grpo_ex = grpo_dataset["train"][0]
print(f"Prompt: {len(grpo_ex['prompt'])} messages")
gt_preview = grpo_ex['ground_truth'][:200] + "..." if len(grpo_ex['ground_truth']) > 200 else grpo_ex['ground_truth']
print(f"Ground truth: {gt_preview}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Markdown: Phase 1 — SFT
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Phase 1 — SFT with Chain-of-Thought 🎓

Teaches the model to **reason step-by-step** using `<think>` tags, then output JSON with `col:val` items.

**v2 fixes applied:**
- LoRA alpha/r = 64/32 = **2.0** (was 0.25 — weights barely updated in v1)
- **5 epochs** (was 2 — insufficient for format memorization)
- Learning rate **1e-4** (was 2e-4 — more stable with more epochs)
- **`packing=False`** (prevents cross-example contamination)
- **`weight_decay=0.01`** + **`max_grad_norm=1.0`** (regularization)
- **`load_best_model_at_end=True`** (pick best epoch by val loss)
- Effective batch size = 2 × 8 = **16** (was 8)"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 7: Load model + LoRA
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 7: Load model + LoRA (FIXED alpha/r ratio) ──────────────────────────
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = CONFIG["base_model"],
    max_seq_length = CONFIG["max_seq_length"],
    load_in_4bit   = CONFIG["load_in_4bit"],
    dtype          = None,  # auto: bfloat16 on Ampere+
)

model = FastLanguageModel.get_peft_model(
    model,
    r                         = CONFIG["lora_r"],       # 32 (was 64)
    lora_alpha                = CONFIG["lora_alpha"],    # 64 (was 16) → ratio 2.0
    target_modules            = CONFIG["lora_target_modules"],
    lora_dropout              = CONFIG["lora_dropout"],  # 0.05 (was 0)
    bias                      = "none",
    use_gradient_checkpointing = "unsloth",
    random_state              = 42,
)

model.print_trainable_parameters()
ratio = CONFIG["lora_alpha"] / CONFIG["lora_r"]
print(f"\\n✅ LoRA loaded — alpha/r ratio: {ratio:.1f} (v1 was 0.25, now {ratio:.1f})")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 8: SFT training
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 8: SFT training (5 epochs, packing=False, weight_decay) ─────────────
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

# Pre-format: apply chat template to messages → text column
def apply_template(examples):
    return {
        "text": [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples["messages"]
        ]
    }

sft_train_fmt = sft_dataset["train"].map(
    apply_template, batched=True,
    remove_columns=sft_dataset["train"].column_names
)
sft_val_fmt = sft_dataset["validation"].map(
    apply_template, batched=True,
    remove_columns=sft_dataset["validation"].column_names
)

# Verify data format before training
sample = sft_train_fmt[0]["text"]
assert "<think>" in sample, f"SFT data missing <think>!\\n{sample[:500]}"
assert ":" in sample, f"SFT data missing col:val format!\\n{sample[:500]}"
print(f"✅ Dataset formatted — train: {len(sft_train_fmt)}, val: {len(sft_val_fmt)}")
print(f"   Sample (first 200): {sample[:200]}")

sft_trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = sft_train_fmt,
    eval_dataset    = sft_val_fmt,
    args            = SFTConfig(
        dataset_text_field          = "text",
        max_seq_length              = CONFIG["max_seq_length"],
        packing                     = False,               # CRITICAL: no packing
        num_train_epochs            = CONFIG["sft_epochs"],
        per_device_train_batch_size = CONFIG["sft_batch_size"],
        gradient_accumulation_steps = CONFIG["sft_grad_accum"],
        learning_rate               = CONFIG["sft_lr"],
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.10,                # was 0.05
        weight_decay                = 0.01,                # ADDED (v2)
        max_grad_norm               = 1.0,                 # ADDED (v2)
        optim                       = "adamw_8bit",
        bf16                        = True,
        fp16                        = False,
        logging_steps               = 10,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,                # ADDED (v2)
        metric_for_best_model       = "eval_loss",
        output_dir                  = CONFIG["sft_output_dir"],
        report_to                   = "none",
        seed                        = 42,
        dataset_num_proc            = 4,
    ),
)

# Only train on assistant responses — mask system + user tokens
sft_trainer = train_on_responses_only(
    sft_trainer,
    instruction_part = "<|im_start|>user\\n",
    response_part    = "<|im_start|>assistant\\n",
)

print("🎓 Starting SFT training (5 epochs, lr=1e-4, packing=False)...")
sft_result = sft_trainer.train()
print(f"✅ SFT done! Final loss: {sft_result.training_loss:.4f}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 9: Save SFT (ADAPTERS ONLY)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 9: Save SFT checkpoint (ADAPTERS ONLY — no destructive merge) ───────
# ⚠️ v1 BUG: used merged_4bit_forced → quantization cascade destroyed weights
# v2 FIX: Save LoRA adapters only. Model stays in memory for GRPO.

model.save_pretrained(CONFIG["sft_output_dir"])
tokenizer.save_pretrained(CONFIG["sft_output_dir"])
print(f"💾 SFT adapters saved → {CONFIG['sft_output_dir']}")

# Free trainer memory but KEEP model for GRPO (no reload needed)
del sft_trainer
gc.collect()
torch.cuda.empty_cache()
vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
print(f"🧹 Trainer freed. VRAM available: {vram_free:.1f} GB")
print("ℹ️  Model kept in memory — continuing to format gate → GRPO")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 10: Format verification gate
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 10: Format verification gate ─────────────────────────────────────────
# ⚠️ Do NOT proceed to GRPO unless SFT model reliably produces correct format.
# Generates from 20 val samples and checks: <think>, valid JSON, col:val items.

FastLanguageModel.for_inference(model)

n_samples = min(20, len(sft_val_fmt))
results = {"think": 0, "json_valid": 0, "colval": 0, "total": 0}
print(f"🔍 Format verification gate — testing {n_samples} validation samples...\\n")

for i in range(n_samples):
    text = sft_val_fmt[i]["text"]
    user_match = re.search(r'<\\|im_start\\|>user\\n(.*?)<\\|im_end\\|>', text, re.DOTALL)
    if not user_match:
        continue

    user_content = user_match.group(1)
    sys_match = re.search(r'<\\|im_start\\|>system\\n(.*?)<\\|im_end\\|>', text, re.DOTALL)
    sys_content = sys_match.group(1) if sys_match else ""

    messages = []
    if sys_content:
        messages.append({"role": "system", "content": sys_content})
    messages.append({"role": "user", "content": user_content})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512, temperature=0.1, do_sample=False,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    results["total"] += 1

    # Check <think> block with real content
    if "<think>" in response and "</think>" in response:
        think_content = response.split("<think>", 1)[1].split("</think>", 1)[0]
        if len(think_content.strip()) > 20:
            results["think"] += 1

    # Check valid JSON array with col:val items
    json_text = response.split("</think>")[-1].strip() if "</think>" in response else response
    json_match = re.search(r'\\[.*\\]', json_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list) and len(parsed) > 0:
                results["json_valid"] += 1
                all_items = [item for x in parsed if isinstance(x, dict)
                             for item in x.get("itemset", [])]
                if all_items and all(":" in str(item) for item in all_items):
                    results["colval"] += 1
        except json.JSONDecodeError:
            pass

    if (i + 1) % 5 == 0:
        print(f"  Tested {i+1}/{n_samples}...")

# Report
print("\\n" + "=" * 50)
print("FORMAT VERIFICATION RESULTS")
print("=" * 50)
for key in ["think", "json_valid", "colval"]:
    rate = results[key] / max(results["total"], 1)
    status = "✅" if rate >= 0.8 else "⚠️" if rate >= 0.5 else "❌"
    print(f"  {status} {key:>12}: {results[key]}/{results['total']} ({rate:.0%})")

overall = min(results[k] / max(results["total"], 1) for k in ["think", "json_valid", "colval"])
if overall >= 0.80:
    print(f"\\n✅ GATE PASSED — compliance {overall:.0%} ≥ 80%. Safe to proceed to GRPO!")
elif overall >= 0.50:
    print(f"\\n⚠️  GATE MARGINAL — compliance {overall:.0%}. GRPO may struggle.")
else:
    print(f"\\n❌ GATE FAILED — compliance {overall:.0%} < 50%. Increase SFT epochs!")

FastLanguageModel.for_training(model)"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Markdown: Phase 2 — GRPO
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Phase 2 — GRPO with 5 Reward Functions 🔬

**No DPO phase** — removed per LLM Council (4/4 unanimous). DPO's contrastive loss
destroyed `<think>` reasoning in v1.

GRPO (Group Relative Policy Optimization) generates completions and optimizes for reward:

| # | Reward Function | What it checks | Range |
|---|----------------|----------------|-------|
| 1 | `reward_structure` | `<think>` + JSON schema | [0, 1] |
| 2 | `reward_colval_format` | Items use `col:val` format | [0, 1] |
| 3 | `reward_f1` | F1 vs Apriori (sqrt scaling) | [0, 1] |
| 4 | `reward_recall_bonus` | 70% recall + 30% precision | [0, 1] |
| 5 | `reward_grounding` | Items exist in CSV headers | [0, 1] |

**v2 fixes:** All rewards give **partial credit** (not binary 0/1).
v1 BUG: `itemset_f1_reward` returned 0.0 for ALL samples → zero learning signal."""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 12: GRPO reward functions
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 12: GRPO reward functions (5 council-designed, partial credit) ───────
import re, json
from typing import List, Optional

def _extract_json_from_completion(text: str) -> Optional[list]:
    \"\"\"Extract JSON array from model completion, handling <think> tags.\"\"\"
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    m = re.search(r'\\[.*\\]', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            return parsed if isinstance(parsed, list) else None
        except json.JSONDecodeError:
            return None
    return None


def _normalize_itemset(items):
    \"\"\"Normalize itemset items for comparison: strip, lowercase, frozenset.\"\"\"
    return frozenset(str(i).strip().lower() for i in items)


def _parse_ground_truth(gt_str):
    \"\"\"Parse ground truth JSON string into set of frozensets.\"\"\"
    try:
        gt = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
    except (json.JSONDecodeError, TypeError):
        return set()
    result = set()
    for x in gt:
        if isinstance(x, dict) and "itemset" in x and isinstance(x["itemset"], list):
            result.add(_normalize_itemset(x["itemset"]))
    return result


def _parse_predicted(parsed_json):
    \"\"\"Parse predicted JSON into set of frozensets.\"\"\"
    if not parsed_json:
        return set()
    result = set()
    for x in parsed_json:
        if isinstance(x, dict) and "itemset" in x and isinstance(x["itemset"], list):
            result.add(_normalize_itemset(x["itemset"]))
    return result


# ── Reward 1: Output structure compliance ─────────────────────────────────────
def reward_structure(completions: List[str], **kwargs) -> List[float]:
    \"\"\"
    Has <think> with content → +0.3 | Has JSON array → +0.4 | Correct schema → +0.3
    Range: [0.0, 1.0]
    \"\"\"
    rewards = []
    for text in completions:
        score = 0.0
        if "<think>" in text and "</think>" in text:
            think = text.split("<think>", 1)[1].split("</think>", 1)[0]
            if len(think.strip()) > 20:
                score += 0.3
        parsed = _extract_json_from_completion(text)
        if parsed is not None and len(parsed) > 0:
            score += 0.4
            if all(isinstance(x, dict) and "itemset" in x and "count" in x for x in parsed):
                score += 0.3
        rewards.append(score)
    return rewards


# ── Reward 2: col:val format compliance ───────────────────────────────────────
def reward_colval_format(completions: List[str], **kwargs) -> List[float]:
    \"\"\"
    Fraction of items using correct col:val format.
    Directly targets the v1 failure: 'age' vs 'age:16'.
    Range: [0.0, 1.0]
    \"\"\"
    rewards = []
    for text in completions:
        parsed = _extract_json_from_completion(text)
        if not parsed:
            rewards.append(0.0)
            continue
        all_items = [item for x in parsed if isinstance(x, dict)
                     for item in x.get("itemset", [])]
        if not all_items:
            rewards.append(0.0)
            continue
        correct = sum(1 for item in all_items
                      if isinstance(item, str) and ":" in item
                      and len(item.split(":", 1)) == 2
                      and all(p.strip() for p in item.split(":", 1)))
        rewards.append(correct / len(all_items))
    return rewards


# ── Reward 3: F1 with partial credit (sqrt scaling) ──────────────────────────
def reward_f1(completions: List[str], ground_truth: List[str] = None, **kwargs) -> List[float]:
    \"\"\"
    F1 score vs Apriori ground truth with sqrt scaling.
    v1 BUG: raw frozenset with NO normalization → always 0.
    v2 FIX: normalize (strip, lowercase) + f1^0.5 for partial credit.
    Range: [0.0, 1.0]
    \"\"\"
    if ground_truth is None:
        return [0.0] * len(completions)
    rewards = []
    for text, gt_str in zip(completions, ground_truth):
        parsed = _extract_json_from_completion(text)
        if not parsed:
            rewards.append(0.0)
            continue
        pred_sets = _parse_predicted(parsed)
        true_sets = _parse_ground_truth(gt_str)
        if not true_sets:
            rewards.append(0.5 if not pred_sets else 0.0)
            continue
        if not pred_sets:
            rewards.append(0.0)
            continue
        tp = len(pred_sets & true_sets)
        precision = tp / len(pred_sets)
        recall = tp / len(true_sets)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rewards.append(f1 ** 0.5)  # sqrt: F1=0.1→0.32 (not flat 0)
    return rewards


# ── Reward 4: Recall-weighted signal ──────────────────────────────────────────
def reward_recall_bonus(completions: List[str], ground_truth: List[str] = None, **kwargs) -> List[float]:
    \"\"\"
    70% recall + 30% precision. Combats the low-recall problem
    (v1: 1-14 itemsets vs 3-387 ground truth).
    Range: [0.0, 1.0]
    \"\"\"
    if ground_truth is None:
        return [0.0] * len(completions)
    rewards = []
    for text, gt_str in zip(completions, ground_truth):
        parsed = _extract_json_from_completion(text)
        if not parsed:
            rewards.append(0.0)
            continue
        pred_sets = _parse_predicted(parsed)
        true_sets = _parse_ground_truth(gt_str)
        if not true_sets:
            rewards.append(0.5)
            continue
        if not pred_sets:
            rewards.append(0.0)
            continue
        tp = len(pred_sets & true_sets)
        precision = tp / len(pred_sets)
        recall = tp / len(true_sets)
        rewards.append(0.7 * recall + 0.3 * precision)
    return rewards


# ── Reward 5: Anti-hallucination (grounding) ─────────────────────────────────
def reward_grounding(completions: List[str], csv_header: List[str] = None, **kwargs) -> List[float]:
    \"\"\"
    Fraction of predicted column names that exist in the input CSV.
    Targets v1's 88.8% hallucination rate.
    Range: [0.0, 1.0]
    \"\"\"
    if csv_header is None:
        return [0.5] * len(completions)  # no header → neutral score
    rewards = []
    for text, header in zip(completions, csv_header):
        parsed = _extract_json_from_completion(text)
        if not parsed:
            rewards.append(0.0)
            continue
        csv_columns = set(col.strip().lower() for col in header.split(',') if col.strip())
        if not csv_columns:
            rewards.append(0.5)
            continue
        pred_cols = set()
        for x in parsed:
            if isinstance(x, dict) and "itemset" in x:
                for item in x["itemset"]:
                    if isinstance(item, str) and ":" in item:
                        pred_cols.add(item.split(":")[0].strip().lower())
        if not pred_cols:
            rewards.append(0.0)
            continue
        valid = pred_cols & csv_columns
        rewards.append(len(valid) / len(pred_cols))
    return rewards


print("✅ GRPO reward functions defined (v2 — 5 functions, partial credit):")
print("   1. reward_structure     — <think> + JSON schema")
print("   2. reward_colval_format — col:val format check")
print("   3. reward_f1            — F1 with sqrt scaling")
print("   4. reward_recall_bonus  — recall-weighted signal")
print("   5. reward_grounding     — anti-hallucination")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 13: Test reward functions
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 13: Pre-flight reward function test ──────────────────────────────────
# ⚠️ v1 BUG: Reward functions were never tested → returned 0 for everything.
# v2 FIX: Test on known good/bad examples BEFORE training.

GOOD_OUTPUT = \"\"\"<think>
Dataset: 3 rows × 3 columns, min_support=2

Singles (count≥2):
- ["age:25"]: Row 1, Row 3 → 2 ✓
- ["health:good"]: Row 1, Row 2 → 2 ✓
- ["income:high"]: Row 1, Row 3 → 2 ✓

Pairs (count≥2):
- ["age:25", "income:high"]: Row 1, Row 3 → 2 ✓
</think>
[{"itemset": ["age:25"], "count": 2, "rows": ["Row 1", "Row 3"]},
 {"itemset": ["health:good"], "count": 2, "rows": ["Row 1", "Row 2"]},
 {"itemset": ["income:high"], "count": 2, "rows": ["Row 1", "Row 3"]},
 {"itemset": ["age:25", "income:high"], "count": 2, "rows": ["Row 1", "Row 3"]}]\"\"\"

BAD_OUTPUT_BARE_COLS = \"\"\"[{"itemset": ["age", "health", "income"], "count": 3, "rows": ["Row 1"]}]\"\"\"

BAD_OUTPUT_NO_JSON = "I found some frequent itemsets: age appears 3 times."

GROUND_TRUTH = '[{"itemset": ["age:25"], "count": 2, "rows": ["Row 1", "Row 3"]}, {"itemset": ["health:good"], "count": 2, "rows": ["Row 1", "Row 2"]}, {"itemset": ["income:high"], "count": 2, "rows": ["Row 1", "Row 3"]}, {"itemset": ["age:25", "income:high"], "count": 2, "rows": ["Row 1", "Row 3"]}]'
CSV_HEADER = "age,health,income"

print("═" * 60)
print("REWARD FUNCTION PRE-FLIGHT TEST")
print("═" * 60)

for label, output in [("GOOD (correct)", GOOD_OUTPUT),
                       ("BAD (bare cols)", BAD_OUTPUT_BARE_COLS),
                       ("BAD (no JSON)", BAD_OUTPUT_NO_JSON)]:
    print(f"\\n── {label} ──")
    print(f"  structure:     {reward_structure([output])[0]:.3f}")
    print(f"  colval_format: {reward_colval_format([output])[0]:.3f}")
    print(f"  f1:            {reward_f1([output], [GROUND_TRUTH])[0]:.3f}")
    print(f"  recall_bonus:  {reward_recall_bonus([output], [GROUND_TRUTH])[0]:.3f}")
    print(f"  grounding:     {reward_grounding([output], [CSV_HEADER])[0]:.3f}")

# Verify good output scores high, bad output scores low
good_f1 = reward_f1([GOOD_OUTPUT], [GROUND_TRUTH])[0]
bad_f1 = reward_f1([BAD_OUTPUT_BARE_COLS], [GROUND_TRUTH])[0]
assert good_f1 > 0.8, f"GOOD output F1 reward too low: {good_f1}"
assert bad_f1 < 0.3, f"BAD output F1 reward too high: {bad_f1}"
print(f"\\n✅ Reward functions verified — good={good_f1:.2f}, bad={bad_f1:.2f}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 14: GRPO training
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 14: GRPO training (500 steps, 5 reward functions) ───────────────────
from trl import GRPOTrainer, GRPOConfig

# Preprocess GRPO dataset: format prompts + extract CSV headers for grounding
def format_grpo(examples):
    prompts = []
    csv_headers = []
    for prompt_msgs in examples["prompt"]:
        # Apply chat template for prompt
        prompts.append(
            tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        )
        # Extract CSV column names from user message for grounding reward
        user_msg = next((m["content"] for m in prompt_msgs if m["role"] == "user"), "")
        row1_match = re.search(r'Row 1:(.+?)(?:\\n|$)', user_msg)
        if row1_match:
            items = row1_match.group(1).strip().split(', ')
            cols = list(set(item.split(':')[0].strip().lower() for item in items if ':' in item))
            csv_headers.append(','.join(cols))
        else:
            csv_headers.append('')
    return {
        "prompt": prompts,
        "ground_truth": examples["ground_truth"],
        "csv_header": csv_headers,
    }

grpo_train = grpo_dataset["train"].map(
    format_grpo, batched=True, remove_columns=grpo_dataset["train"].column_names
)
print(f"✅ GRPO dataset formatted — {len(grpo_train)} examples")
print(f"   Columns: {grpo_train.column_names}")
print(f"   Sample CSV header: {grpo_train[0]['csv_header'][:100]}")

# Create GRPO trainer — model already has SFT LoRA weights active
grpo_trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    reward_funcs     = [
        reward_structure,           # 1. <think> + JSON
        reward_colval_format,       # 2. col:val format
        reward_f1,                  # 3. F1 (main signal)
        reward_recall_bonus,        # 4. recall incentive
        reward_grounding,           # 5. anti-hallucination
    ],
    args             = GRPOConfig(
        max_steps                   = CONFIG["grpo_max_steps"],
        per_device_train_batch_size = CONFIG["grpo_batch_size"],
        gradient_accumulation_steps = CONFIG["grpo_grad_accum"],
        learning_rate               = CONFIG["grpo_lr"],
        num_generations             = CONFIG["grpo_num_generations"],
        max_completion_length       = CONFIG["grpo_max_completion_length"],
        max_prompt_length           = CONFIG["max_seq_length"] - CONFIG["grpo_max_completion_length"],
        warmup_ratio                = 0.05,
        weight_decay                = 0.01,
        max_grad_norm               = 0.5,      # conservative for RL
        optim                       = "adamw_8bit",
        bf16                        = True,
        fp16                        = False,
        logging_steps               = 5,
        save_steps                  = 100,
        output_dir                  = CONFIG["grpo_output_dir"],
        report_to                   = "none",
        seed                        = 42,
        beta                        = CONFIG["grpo_beta"],
        temperature                 = CONFIG["grpo_temperature"],
        top_p                       = 0.9,
    ),
    train_dataset    = grpo_train,
)

print(f"\\n🔬 Starting GRPO training for {CONFIG['grpo_max_steps']} steps...")
print(f"   Rewards: structure, colval, f1, recall, grounding")
print(f"   Generations per prompt: {CONFIG['grpo_num_generations']}")
print(f"   Beta (KL penalty): {CONFIG['grpo_beta']}")
grpo_result = grpo_trainer.train()
print(f"✅ GRPO done!")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 15: Save + Push
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 15: Save final model + push to HuggingFace Hub ──────────────────────
# For final deployment, merged_4bit_forced is fine (no more phases after this)
import os

model.save_pretrained_merged(
    CONFIG["grpo_output_dir"] + "/final",
    tokenizer,
    save_method = "merged_4bit_forced",
)
print(f"💾 Final model saved → {CONFIG['grpo_output_dir']}/final")

if CONFIG["push_to_hub"]:
    hf_token = CONFIG["hf_token"] or os.environ.get("HF_TOKEN", "")
    print(f"\\n🚀 Pushing to HF Hub: {CONFIG['hf_model_repo']}")
    model.push_to_hub_merged(
        CONFIG["hf_model_repo"],
        tokenizer,
        save_method = "merged_4bit_forced",
        token       = hf_token,
    )
    tokenizer.push_to_hub(
        CONFIG["hf_model_repo"],
        token = hf_token,
    )
    print(f"✅ Model live at: https://huggingface.co/{CONFIG['hf_model_repo']}")
else:
    print("ℹ️  push_to_hub=False — model saved locally only")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Markdown: Inference test
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Quick Inference Test 🧪

Sanity check on a sample CSV to verify the model produces `<think>` reasoning + valid JSON with `col:val` items."""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 17: Quick inference test
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# ── Cell 17: Quick inference test ─────────────────────────────────────────────
FastLanguageModel.for_inference(model)

SYSTEM_PROMPT = (
    "You are a frequent itemset extractor. Given CSV transaction data and a "
    "minimum support count, identify all itemsets whose items co-occur in at "
    "least that many rows.\\n\\n"
    "Rules:\\n"
    "1. Scan single items, pairs, and triples (up to size 3)\\n"
    "2. Count = number of distinct rows containing ALL items in the itemset\\n"
    "3. Only report itemsets with count >= min_support\\n"
    "4. Canonicalize items: lowercase, trimmed, sorted alphabetically\\n"
    '5. Row references: "Row N" format, 1-based indexing\\n\\n'
    "Think step by step inside <think> tags, then output ONLY a JSON array:\\n"
    '[{\\"itemset\\": [\\"item1\\", \\"item2\\"], \\"count\\": N, \\"rows\\": [\\"Row 1\\", \\"Row 3\\"]}]'
)

SAMPLE_CSV = \"\"\"Row 1: bread:yes, milk:yes, eggs:yes
Row 2: bread:yes, butter:yes, jam:yes
Row 3: milk:yes, eggs:yes, cheese:yes
Row 4: bread:yes, milk:yes, eggs:yes, butter:yes
Row 5: bread:yes, eggs:yes\"\"\"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Find all frequent itemsets with minimum support count = 2 in this dataset:\\n\\n{SAMPLE_CSV}"},
]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids      = inputs,
    max_new_tokens = 1024,
    temperature    = 0.1,
    do_sample      = True,
    pad_token_id   = tokenizer.eos_token_id,
)

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print("─── Model Output ───")
print(response)

# Validate
try:
    json_text = response
    if "</think>" in json_text:
        json_text = json_text.split("</think>", 1)[-1].strip()
    parsed = json.loads(re.search(r'\\[.*\\]', json_text, re.DOTALL).group())
    print(f"\\n✅ Valid JSON — {len(parsed)} itemsets found")

    # Check format
    has_think = "<think>" in response and "</think>" in response
    all_items = [item for x in parsed if isinstance(x, dict) for item in x.get("itemset", [])]
    has_colval = all(":" in str(item) for item in all_items) if all_items else False

    print(f"   <think> tags: {'✅' if has_think else '❌'}")
    print(f"   col:val format: {'✅' if has_colval else '❌'}")
    print(f"   Items: {all_items[:10]}{'...' if len(all_items) > 10 else ''}")
except Exception as e:
    print(f"\\n⚠️  JSON parse failed: {e}")"""))


# ═══════════════════════════════════════════════════════════════════════════════
# Assemble notebook
# ═══════════════════════════════════════════════════════════════════════════════
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

outpath = Path("notebooks/training_3phase_2026-03-07_v2.ipynb")
outpath.parent.mkdir(parents=True, exist_ok=True)
outpath.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"✅ Notebook written → {outpath}")
print(f"   Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type'] == 'code')} code, "
      f"{sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown)")

# Also update notebook_versions.json
versions_path = Path("notebooks/notebook_versions.json")
versions = []
if versions_path.exists():
    versions = json.loads(versions_path.read_text())
versions.append({
    "version": "v2",
    "date": "2026-03-07",
    "filename": outpath.name,
    "phases": "SFT-CoT → GRPO (no DPO)",
    "changes": [
        "LoRA alpha/r = 64/32 = 2.0 (was 16/64 = 0.25)",
        "DPO removed entirely (council 4/4 unanimous)",
        "SFT 5 epochs (was 2), lr=1e-4 (was 2e-4)",
        "packing=False, weight_decay=0.01, max_grad_norm=1.0",
        "Save adapters only between phases (no merged_4bit_forced)",
        "5 GRPO reward functions with partial credit",
        "Format verification gate between SFT and GRPO",
        "max_seq_length=6144 (was 4096)",
        "GRPO: 500 steps, lr=2e-6, num_gen=6, completion=3000, beta=0.001",
    ],
    "based_on_council": "docs/reports/council_training_advice.json",
})
versions_path.write_text(json.dumps(versions, indent=2, ensure_ascii=False))
print(f"✅ Version log updated → {versions_path}")
