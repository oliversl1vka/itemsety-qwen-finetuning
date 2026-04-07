#!/usr/bin/env python3
"""Verify both HF repos have correct data after push."""
from datasets import load_dataset
import os

# Use token from env
token = os.environ.get("HF_TOKEN")

print("=" * 60)
print("VERIFYING HF REPOS (downloading from Hub)")
print("=" * 60)

# ── V2 ──
print("\n📦 OliverSlivka/itemset-extraction-v2")
for config in ["sft", "dpo", "grpo"]:
    ds = load_dataset("OliverSlivka/itemset-extraction-v2", config, token=token)
    print(f"  {config}: train={len(ds['train'])}, val={len(ds['validation'])}")
    if config == "sft":
        sample = ds["train"][0]["messages"][-1]["content"]
        has_row_n = "Row " in sample
        has_r_ref = ", R" in sample[:200]
        print(f"    ✓ Has 'Row N' format: {has_row_n}")
        print(f"    ✓ Has R-ref format: {has_r_ref}")
        assert has_row_n, "ERROR: v2 SFT should have verbose 'Row N' format!"
        print(f"    Sample (first 200 chars): {sample[:200]}")

# ── V3 ──
print(f"\n📦 OliverSlivka/itemset-extraction-v3")
for config in ["sft", "dpo", "grpo"]:
    ds = load_dataset("OliverSlivka/itemset-extraction-v3", config, token=token)
    print(f"  {config}: train={len(ds['train'])}, val={len(ds['validation'])}")
    if config == "sft":
        sample = ds["train"][0]["messages"][-1]["content"]
        has_row_n = "Row " in sample
        has_r_ref = ", R" in sample[:300]
        print(f"    ✓ Has verbose 'Row N': {has_row_n}")
        print(f"    ✓ Has spaced R-refs: {has_r_ref}")
        print(f"    Sample (first 200 chars): {sample[:200]}")

# ── Summary ──
print(f"\n{'=' * 60}")
print("✅ VERIFICATION COMPLETE")
print(f"{'=' * 60}")
print("  v2: 314/34 SFT (verbose Row N) — FROZEN for v2 training history")
print("  v3: 245/27 SFT (concise spaced R-refs) — CURRENT for v3 training")
