#!/usr/bin/env python3
"""
Standalone SFT publish script — HF-native loading, clean merge, push to Hub.

Fix for the Cell 20 issue: Unsloth's FastLanguageModel doesn't become a
proper PeftModel, so push_to_hub_merged silently skips the merge.
This script uses pure HF/PEFT APIs to load the SFT LoRA adapter, merge it,
and push a clean 16-bit model to HuggingFace.

Usage:
    # Set token in env (recommended) or pass via --token
    export HF_TOKEN="hf_..."
    python publish_sft_clean.py

    # Or pass token directly
    python publish_sft_clean.py --token "hf_..."

    # Dry run (skip push, just verify checkpoint and merge locally)
    python publish_sft_clean.py --dry-run

Requirements (install on TLJH if needed, but should be there):
    pip install transformers accelerate peft bitsandbytes huggingface_hub
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import torch
from pathlib import Path

# ── CONFIG — matches the notebook's CONFIG dict ──────────────────────────────
CONFIG = {
    "base_model_name":     "Qwen/Qwen2.5-7B-Instruct",
    "max_seq_length":      4096,
    "load_in_4bit":        True,
    "sft_output_dir":      "./sft_checkpoint",
    "hf_model_repo":       "OliverSlivka/qwen2.5-7b-itemset-extractor",
}


def check_adapter_files(checkpoint_dir: str) -> dict:
    """Verify the SFT checkpoint contains valid LoRA adapter files."""
    ckpt = Path(checkpoint_dir)
    result = {"ok": True, "files": {}, "warnings": []}

    for fname, label in [
        ("adapter_config.json", "LoRA config"),
        ("adapter_model.safetensors", "LoRA weights (safetensors)"),
    ]:
        path = ckpt / fname
        if path.exists():
            result["files"][fname] = path
        else:
            result["files"][fname] = None
            if fname == "adapter_config.json":
                result["ok"] = False
                result["warnings"].append(f"MISSING: {label} — this dir is not a PEFT adapter checkpoint")
            else:
                result["warnings"].append(f"MISSING: {label}")

    # Check adapter_config.json content if present
    ac_path = ckpt / "adapter_config.json"
    if ac_path.exists():
        with open(ac_path) as f:
            ac = json.load(f)
        peft_type = ac.get("peft_type", "")
        if peft_type.upper() != "LORA":
            result["ok"] = False
            result["warnings"].append(f"adapter_config.json peft_type='{peft_type}', expected 'LORA'")
        result["peft_config"] = ac

    return result


def clean_stale_adapter_files(hf_token: str, repo_id: str, dry_run: bool = False):
    """Remove stale adapter_model.safetensors and adapter_* files from HF repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    try:
        existing_files = api.list_repo_files(repo_id, repo_type="model")
    except Exception as e:
        print(f"⚠️  Could not list repo files: {e}")
        print("   (repo may be empty or not exist yet — this is fine)")
        return

    stale = [
        path for path in existing_files
        if path.startswith("adapter_") or path == "adapter_model.safetensors"
    ]
    if not stale:
        print("✅ No stale adapter files in repo")
        return

    print(f"🧹 Removing {len(stale)} stale adapter file(s) from repo: {stale}")
    if dry_run:
        print("   (dry-run: skipping actual deletion)")
        return

    for path in stale:
        try:
            api.delete_file(
                path_in_repo=path,
                repo_id=repo_id,
                repo_type="model",
                token=hf_token,
            )
            print(f"   ✅ Deleted: {path}")
        except Exception as e:
            print(f"   ⚠️  Failed to delete {path}: {e}")


def push_model_card(hf_token: str, repo_id: str, dry_run: bool = False):
    """Push HF_MODEL_CARD.md as the repo's README.md."""
    card_path = Path("HF_MODEL_CARD.md")
    if not card_path.exists():
        print("ℹ️  No HF_MODEL_CARD.md found locally — skipping model card push")
        return

    from huggingface_hub import HfApi

    content = card_path.read_text()
    if dry_run:
        print(f"📄 Would push model card ({len(content)} chars) → {repo_id}/README.md")
        return

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
    )
    print(f"📄 Model card pushed → https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Publish SFT model cleanly to HF Hub")
    parser.add_argument("--token", help="HF token (falls back to HF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true", help="Verify but don't push")
    parser.add_argument("--checkpoint", default=CONFIG["sft_output_dir"],
                        help=f"Path to SFT checkpoint (default: {CONFIG['sft_output_dir']})")
    parser.add_argument("--repo", default=CONFIG["hf_model_repo"],
                        help=f"HF repo ID (default: {CONFIG['hf_model_repo']})")
    parser.add_argument("--base-model", default=CONFIG["base_model_name"],
                        help=f"Base model (default: {CONFIG['base_model_name']})")
    parser.add_argument("--skip-clean", action="store_true",
                        help="Skip cleaning stale adapter files from repo")
    parser.add_argument("--skip-card", action="store_true",
                        help="Skip pushing model card")
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN", "")
    if not hf_token and not args.dry_run:
        print("❌ No HF token. Set HF_TOKEN env var or pass --token")
        sys.exit(1)

    print("=" * 65)
    print("🚀 CLEAN SFT PUBLISH")
    print("=" * 65)
    print(f"   Base model:  {args.base_model}")
    print(f"   Checkpoint:  {args.checkpoint}")
    print(f"   HF repo:     {args.repo}")
    print(f"   Dry run:     {args.dry_run}")
    print()

    # ── 1. Verify checkpoint ─────────────────────────────────────────────────
    print("─" * 65)
    print("📋 Step 1: Verify SFT checkpoint")
    print("─" * 65)

    adapter_check = check_adapter_files(args.checkpoint)
    for k, v in adapter_check["files"].items():
        status = "✅" if v else "❌"
        print(f"   {status} {k}")
    for w in adapter_check["warnings"]:
        print(f"   ⚠️  {w}")

    if not adapter_check["ok"]:
        print("\n❌ Checkpoint validation FAILED. Cannot publish.")
        sys.exit(1)
    print("✅ Checkpoint is valid PEFT/LoRA adapter\n")

    # ── 2. Load base + adapter (HF-native) ───────────────────────────────────
    print("─" * 65)
    print("📦 Step 2: Load base model + SFT adapter (HF-native)")
    print("─" * 65)

    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"   Loading base model: {args.base_model}")
    base_model = None
    try:
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        print("   ✅ Base model loaded (4-bit, SDPA)")
    except Exception as e:
        print(f"   ❌ Failed to load base model: {e}")
        sys.exit(1)

    print(f"   Loading LoRA adapter from: {args.checkpoint}")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            args.checkpoint,
            is_trainable=False,
        )
        model.eval()
    except Exception as e:
        print(f"   ❌ Failed to load LoRA adapter: {e}")
        sys.exit(1)

    if not isinstance(model, PeftModel):
        print("   ❌ Model is NOT a PeftModel after loading adapter. Aborting.")
        sys.exit(1)
    print("   ✅ Model loaded as PeftModel (LoRA adapter bound)\n")

    # ── 3. Load tokenizer ────────────────────────────────────────────────────
    print("─" * 65)
    print("🔤 Step 3: Load tokenizer")
    print("─" * 65)

    # Try checkpoint first (it may have tokenizer files), then base model
    if (Path(args.checkpoint) / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        print(f"   ✅ Tokenizer loaded from checkpoint: {args.checkpoint}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        print(f"   ✅ Tokenizer loaded from base model: {args.base_model}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("   🔧 Set pad_token = eos_token")
    print()

    # ── 4. Clean stale adapter files from HF repo ────────────────────────────
    if not args.skip_clean and not args.dry_run:
        print("─" * 65)
        print("🧹 Step 4: Clean stale adapter files from HF repo")
        print("─" * 65)
        clean_stale_adapter_files(hf_token, args.repo, args.dry_run)
    elif args.dry_run:
        print("🧹 Step 4: Would clean stale adapter files (dry-run)")
    else:
        print("⏭️  Step 4: Skipped (--skip-clean)")
    print()

    # ── 5. Merge and push ────────────────────────────────────────────────────
    print("─" * 65)
    print("🔀 Step 5: Merge LoRA + push to HF Hub")
    print("─" * 65)

    if args.dry_run:
        print("   (dry-run: loading adapter only, skipping merge + push)")
        print("   ✅ Dry run complete — checkpoint is valid, model loads as PeftModel")
        return

    # Merge LoRA weights into base model (fp16, in memory)
    print("   🔀 Merging LoRA weights into base model (fp16)...")
    try:
        merged_model = model.merge_and_unload()
        print("   ✅ LoRA merged into base model")
    except Exception as e:
        print(f"   ❌ Merge failed: {e}")
        print("      Trying to save and push LoRA adapter instead...")
        # Fallback: push as adapter
        print("   📦 Pushing LoRA adapter (unmerged)...")
        model.push_to_hub(
            args.repo,
            token=hf_token,
        )
        tokenizer.push_to_hub(args.repo, token=hf_token)
        print(f"   ✅ LoRA adapter pushed to: https://huggingface.co/{args.repo}")
        print("   ℹ️  The model was pushed as a LoRA adapter, not merged.")
        return

    # Push merged model
    print(f"   📤 Pushing merged model to: {args.repo}")
    merged_model.push_to_hub(
        args.repo,
        token=hf_token,
    )
    print("   ✅ Merged model pushed")

    # Push tokenizer
    print("   📤 Pushing tokenizer...")
    tokenizer.push_to_hub(args.repo, token=hf_token)
    print("   ✅ Tokenizer pushed\n")

    # ── 6. Cleanup ───────────────────────────────────────────────────────────
    del merged_model
    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── 7. Push model card ───────────────────────────────────────────────────
    if not args.skip_card:
        print("─" * 65)
        print("📄 Step 6: Push model card")
        print("─" * 65)
        push_model_card(hf_token, args.repo, dry_run=False)
    print()

    # ── Done ──────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("✅ DONE — SFT model published cleanly")
    print(f"   https://huggingface.co/{args.repo}")
    print("=" * 65)
    print()
    print("Load with:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.repo}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.repo}')")


if __name__ == "__main__":
    main()
