#!/usr/bin/env python3
"""
Convert exported training data to HuggingFace Dataset format.
Supports conversational format for SFT training.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import argparse


def create_conversational_example(
    example: Dict[str, Any], system_prompt_path: str = "extractor_system_prompt.md"
) -> Dict[str, Any]:
    """Convert raw example to conversational format for SFT"""

    # Load system prompt
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # Format user message
    user_message = f"{example['csv_context']}\n\n"
    user_message += (
        f"Find all frequent itemsets with minimum support count = "
        f"{example['min_support']}."
    )

    # Format assistant response (ground truth)
    assistant_response = json.dumps(example["ground_truth"], ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response},
        ],
        "metadata": {
            "dataset_id": example["id"],
            "dataset_name": example["dataset_name"],
            "itemset_count": len(example["ground_truth"]),
            "min_support": example["min_support"],
        },
    }


def create_hf_dataset(
    input_file: str = "training_data/all_training_examples.json",
    output_dir: str = "hf_dataset",
    train_split: float = 0.9,
    system_prompt_path: str = "extractor_system_prompt.md",
) -> None:
    """Create HuggingFace dataset from training examples"""

    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        print("❌ Error: 'datasets' package not installed")
        print("   Run: pip install datasets")
        return

    # Load examples
    with open(input_file, "r", encoding="utf-8") as f:
        examples = json.load(f)

    print(f"📦 Loaded {len(examples)} training examples")

    # Convert to conversational format
    conversational_examples = []
    for example in examples:
        conv_example = create_conversational_example(example, system_prompt_path)
        conversational_examples.append(conv_example)

    # Shuffle and split
    import random

    random.seed(42)
    random.shuffle(conversational_examples)

    split_idx = int(len(conversational_examples) * train_split)
    train_examples = conversational_examples[:split_idx]
    val_examples = conversational_examples[split_idx:]

    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    # Create dataset dict
    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    print(f"\n✅ HuggingFace Dataset Created!")
    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Validation: {len(val_dataset)} examples")
    print(f"   Output: {output_path}")
    print(f"\nTo load:")
    print(f"   from datasets import load_from_disk")
    print(f"   dataset = load_from_disk('{output_path}')")


def main():
    parser = argparse.ArgumentParser(description="Create HuggingFace dataset")
    parser.add_argument("--input", default="training_data/all_training_examples.json")
    parser.add_argument("--output", default="hf_dataset")
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--system-prompt", default="extractor_system_prompt.md")

    args = parser.parse_args()

    create_hf_dataset(
        input_file=args.input,
        output_dir=args.output,
        train_split=args.train_split,
        system_prompt_path=args.system_prompt,
    )


if __name__ == "__main__":
    main()
