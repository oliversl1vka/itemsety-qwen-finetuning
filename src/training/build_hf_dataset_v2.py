#!/usr/bin/env python3
"""
Build a unified HuggingFace dataset for 3-phase training: SFT → DPO → GRPO.

Reads the outputs from:
  - generate_cot_sft_data.py  → SFT examples with CoT
  - export_real_dpo_data.py   → DPO pairs with real LLM failures

Creates a HuggingFace DatasetDict with three configs:
  - sft:  {"messages": [...]}  — for SFTTrainer
  - dpo:  {"prompt": [...], "chosen": [...], "rejected": [...]}  — for DPOTrainer
  - grpo: {"prompt": [...], "ground_truth": "json_string"}  — for GRPOTrainer

Usage:
    # Step 1: Generate data
    python src/training/generate_cot_sft_data.py --db runs.db --output data/sft_cot_v2.json
    python src/training/export_real_dpo_data.py --db runs.db --output data/dpo_real_v2.json

    # Step 2: Build HF dataset
    python src/training/build_hf_dataset_v2.py \\
        --sft-data data/sft_cot_v2.json \\
        --dpo-data data/dpo_real_v2.json \\
        --output data/hf_dataset_v2 \\
        --push OliverSlivka/itemset-extraction-v2
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import argparse

from datasets import Dataset, DatasetDict


def load_json(path: str) -> List[Dict]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sft_split(sft_data: List[Dict], val_ratio: float = 0.1) -> DatasetDict:
    """
    Build SFT dataset split.

    Format: {"messages": [system, user, assistant_with_cot]}
    """
    # Shuffle deterministically
    random.seed(42)
    data = sft_data.copy()
    random.shuffle(data)

    val_size = max(1, int(len(data) * val_ratio))
    train_data = data[val_size:]
    val_data = data[:val_size]

    def extract_sft(examples: List[Dict]) -> Dict:
        return {"messages": [ex["messages"] for ex in examples]}

    train_ds = Dataset.from_dict(extract_sft(train_data))
    val_ds = Dataset.from_dict(extract_sft(val_data))

    return DatasetDict({"train": train_ds, "validation": val_ds})


def build_dpo_split(dpo_data: List[Dict], val_ratio: float = 0.1) -> DatasetDict:
    """
    Build DPO dataset split.

    Format: {"prompt": [msgs], "chosen": [msgs], "rejected": [msgs]}
    """
    random.seed(42)
    data = dpo_data.copy()
    random.shuffle(data)

    val_size = max(1, int(len(data) * val_ratio))
    train_data = data[val_size:]
    val_data = data[:val_size]

    def extract_dpo(examples: List[Dict]) -> Dict:
        return {
            "prompt": [ex["prompt"] for ex in examples],
            "chosen": [ex["chosen"] for ex in examples],
            "rejected": [ex["rejected"] for ex in examples],
        }

    train_ds = Dataset.from_dict(extract_dpo(train_data))
    val_ds = Dataset.from_dict(extract_dpo(val_data))

    return DatasetDict({"train": train_ds, "validation": val_ds})


def build_grpo_split(sft_data: List[Dict], val_ratio: float = 0.1) -> DatasetDict:
    """
    Build GRPO dataset split from SFT data (reuse prompts + ground truth).

    Format: {"prompt": [system, user], "ground_truth": "json_string"}
    The ground_truth is passed as kwargs to reward functions.
    """
    random.seed(42)
    data = sft_data.copy()
    random.shuffle(data)

    val_size = max(1, int(len(data) * val_ratio))
    train_data = data[val_size:]
    val_data = data[:val_size]

    def extract_grpo(examples: List[Dict]) -> Dict:
        prompts = []
        ground_truths = []
        for ex in examples:
            # prompt = system + user messages (no assistant)
            prompt_msgs = [m for m in ex["messages"] if m["role"] != "assistant"]
            prompts.append(prompt_msgs)
            ground_truths.append(ex.get("ground_truth", "[]"))
        return {"prompt": prompts, "ground_truth": ground_truths}

    train_ds = Dataset.from_dict(extract_grpo(train_data))
    val_ds = Dataset.from_dict(extract_grpo(val_data))

    return DatasetDict({"train": train_ds, "validation": val_ds})


def build_all(
    sft_path: str,
    dpo_path: str,
    output_dir: str,
    val_ratio: float = 0.1,
    push_to_hub: str = None,
    hf_token: str = None,
) -> None:
    """Build all three dataset configs and save."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print("📂 Loading data...")
    sft_data = load_json(sft_path)
    dpo_data = load_json(dpo_path)
    print(f"   SFT examples: {len(sft_data)}")
    print(f"   DPO pairs: {len(dpo_data)}")

    # Build SFT split
    print("\n🎓 Building SFT dataset...")
    sft_ds = build_sft_split(sft_data, val_ratio)
    sft_ds.save_to_disk(str(output_path / "sft"))
    print(f"   Train: {len(sft_ds['train'])} | Val: {len(sft_ds['validation'])}")

    # Build DPO split
    print("\n🎯 Building DPO dataset...")
    dpo_ds = build_dpo_split(dpo_data, val_ratio)
    dpo_ds.save_to_disk(str(output_path / "dpo"))
    print(f"   Train: {len(dpo_ds['train'])} | Val: {len(dpo_ds['validation'])}")

    # Build GRPO split (reuses SFT prompts + ground truth)
    print("\n🔬 Building GRPO dataset...")
    grpo_ds = build_grpo_split(sft_data, val_ratio)
    grpo_ds.save_to_disk(str(output_path / "grpo"))
    print(f"   Train: {len(grpo_ds['train'])} | Val: {len(grpo_ds['validation'])}")

    # Push to HuggingFace Hub
    if push_to_hub:
        from huggingface_hub import login

        if hf_token:
            login(token=hf_token)

        print(f"\n🚀 Pushing to HuggingFace Hub: {push_to_hub}")

        # Push each config separately
        sft_ds.push_to_hub(push_to_hub, config_name="sft", private=False)
        print(f"   ✅ Pushed SFT config")

        dpo_ds.push_to_hub(push_to_hub, config_name="dpo", private=False)
        print(f"   ✅ Pushed DPO config")

        grpo_ds.push_to_hub(push_to_hub, config_name="grpo", private=False)
        print(f"   ✅ Pushed GRPO config")

        print(f"\n🎉 Dataset live at: https://huggingface.co/datasets/{push_to_hub}")

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Dataset build complete!")
    print(f"   Output: {output_path}")
    print(f"   SFT:  {len(sft_ds['train'])} train / {len(sft_ds['validation'])} val")
    print(f"   DPO:  {len(dpo_ds['train'])} train / {len(dpo_ds['validation'])} val")
    print(f"   GRPO: {len(grpo_ds['train'])} train / {len(grpo_ds['validation'])} val")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Build unified HF dataset for 3-phase training"
    )
    parser.add_argument(
        "--sft-data", default="data/sft_cot_v2.json",
        help="Path to SFT CoT training data JSON"
    )
    parser.add_argument(
        "--dpo-data", default="data/dpo_real_v2.json",
        help="Path to DPO real failures data JSON"
    )
    parser.add_argument(
        "--output", default="data/hf_dataset_v2",
        help="Output directory for HF dataset"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--push", default=None,
        help="HuggingFace repo to push to (e.g., OliverSlivka/itemset-extraction-v2)"
    )
    parser.add_argument("--hf-token", default=None, help="HuggingFace API token")

    args = parser.parse_args()

    build_all(
        sft_path=args.sft_data,
        dpo_path=args.dpo_data,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        push_to_hub=args.push,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
