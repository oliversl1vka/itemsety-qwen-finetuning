#!/usr/bin/env python3
"""
Inspect and validate training data before fine-tuning.
Checks format, statistics, and potential issues.
"""

import json
from pathlib import Path
import argparse


def check_json_validity(examples) -> bool:
    """Check if all assistant responses are valid JSON"""
    try:
        for ex in examples:
            assistant_msg = ex["messages"][2]["content"]
            json.loads(assistant_msg)
        return True
    except Exception:
        return False


def inspect_hf_dataset(dataset_path: str = "hf_dataset") -> None:
    """Inspect HuggingFace dataset structure and statistics"""

    try:
        from datasets import load_from_disk
    except ImportError:
        print("❌ Error: 'datasets' package not installed")
        print("   Run: pip install datasets")
        return

    dataset = load_from_disk(dataset_path)

    print("=" * 60)
    print("📊 DATASET INSPECTION REPORT")
    print("=" * 60)

    # Basic stats
    print("\n🔢 SPLIT SIZES:")
    print(f"   Train: {len(dataset['train'])} examples")
    print(f"   Validation: {len(dataset['validation'])} examples")
    total = len(dataset["train"]) + len(dataset["validation"])
    print(f"   Total: {total} examples")

    # Sample first example
    print("\n📝 SAMPLE EXAMPLE (train[0]):")
    sample = dataset["train"][0]
    print(f"   Messages: {len(sample['messages'])} turns")
    for i, msg in enumerate(sample["messages"]):
        role = msg["role"]
        content = msg["content"]
        content_preview = content[:100] + "..." if len(content) > 100 else content
        print(f"     {i+1}. {role}: {content_preview}")

    # Metadata stats
    print("\n📈 METADATA STATISTICS:")
    if "metadata" in sample:
        itemset_counts = [ex["metadata"]["itemset_count"] for ex in dataset["train"]]
        print(
            f"   Itemset count range: " f"{min(itemset_counts)} - {max(itemset_counts)}"
        )
        avg_count = sum(itemset_counts) / len(itemset_counts)
        print(f"   Average itemsets: {avg_count:.1f}")

    # Message length stats
    print("\n📏 MESSAGE LENGTH STATISTICS:")
    user_lengths = [len(ex["messages"][1]["content"]) for ex in dataset["train"]]
    assistant_lengths = [len(ex["messages"][2]["content"]) for ex in dataset["train"]]

    print("   User message (CSV context):")
    print(f"     Min: {min(user_lengths)} chars")
    print(f"     Max: {max(user_lengths)} chars")
    print(f"     Avg: {sum(user_lengths) / len(user_lengths):.0f} chars")

    print("   Assistant message (ground truth):")
    print(f"     Min: {min(assistant_lengths)} chars")
    print(f"     Max: {max(assistant_lengths)} chars")
    avg_asst = sum(assistant_lengths) / len(assistant_lengths)
    print(f"     Avg: {avg_asst:.0f} chars")

    # Token estimate (rough: chars / 4)
    avg_total_chars = (sum(user_lengths) + sum(assistant_lengths)) / len(
        dataset["train"]
    )
    estimated_tokens = avg_total_chars / 4
    print(f"\n🎯 ESTIMATED TOKENS PER EXAMPLE: ~{estimated_tokens:.0f}")
    print("   (Rough estimate: total_chars / 4)")

    # Validation checks
    print("\n✅ VALIDATION CHECKS:")
    checks = {
        "All examples have 3 messages": all(
            len(ex["messages"]) == 3 for ex in dataset["train"]
        ),
        "All messages have role+content": all(
            all("role" in msg and "content" in msg for msg in ex["messages"])
            for ex in dataset["train"]
        ),
        "Assistant responses are valid JSON": check_json_validity(dataset["train"]),
        "No empty messages": all(
            all(len(msg["content"]) > 0 for msg in ex["messages"])
            for ex in dataset["train"]
        ),
    }

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect training dataset")
    parser.add_argument("--dataset", default="hf_dataset", help="Path to HF dataset")

    args = parser.parse_args()

    inspect_hf_dataset(args.dataset)


if __name__ == "__main__":
    main()
