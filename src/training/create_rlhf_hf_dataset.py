"""
Convert exported RLHF training data to HuggingFace Dataset format.
Supports DPO (Direct Preference Optimization) and PPO-style RLHF training.

Based on formats from:
- HH-RLHF: https://github.com/anthropics/hh-rlhf
- Stanford SHP: https://huggingface.co/datasets/stanfordnlp/SHP
- OpenAI WebGPT: https://huggingface.co/datasets/openai/webgpt_comparisons
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import argparse


def create_dpo_example(
    rlhf_pair: Dict[str, Any], 
    system_prompt_path: str = "extractor_system_prompt.md"
) -> Dict[str, Any]:
    """
    Convert RLHF pair to DPO format (Direct Preference Optimization).
    
    DPO format:
    {
        "prompt": [{"role": "system", ...}, {"role": "user", ...}],
        "chosen": [{"role": "assistant", ...}],
        "rejected": [{"role": "assistant", ...}]
    }
    """
    
    # Load system prompt
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    # Create messages
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rlhf_pair["prompt"]},
    ]
    
    chosen_messages = [
        {"role": "assistant", "content": rlhf_pair["chosen"]}
    ]
    
    rejected_messages = [
        {"role": "assistant", "content": rlhf_pair["rejected"]}
    ]
    
    return {
        "prompt": prompt_messages,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
        "error_type": rlhf_pair["error_type"],
        "metadata": rlhf_pair["metadata"],
    }


def create_ppo_example(
    rlhf_pair: Dict[str, Any],
    system_prompt_path: str = "extractor_system_prompt.md"
) -> Dict[str, Any]:
    """
    Convert RLHF pair to PPO format (for reward model training).
    
    PPO format (for reward modeling):
    {
        "query": "Full prompt text",
        "responses": [chosen_response, rejected_response],
        "scores": [1.0, 0.0],  # Binary preference
    }
    """
    
    # Load system prompt
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    # Combine system + user prompt
    full_prompt = f"{system_prompt}\n\n{rlhf_pair['prompt']}"
    
    return {
        "query": full_prompt,
        "responses": [rlhf_pair["chosen"], rlhf_pair["rejected"]],
        "scores": [1.0, 0.0],  # Chosen = 1.0, Rejected = 0.0
        "error_type": rlhf_pair["error_type"],
        "metadata": rlhf_pair["metadata"],
    }


def create_conversational_pair(
    rlhf_pair: Dict[str, Any],
    system_prompt_path: str = "extractor_system_prompt.md"
) -> Dict[str, Any]:
    """
    Convert RLHF pair to conversational format for TRL library.
    
    Conversational format:
    {
        "chosen": [full conversation with good response],
        "rejected": [full conversation with bad response],
    }
    """
    
    # Load system prompt
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    # Chosen conversation
    chosen_conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rlhf_pair["prompt"]},
        {"role": "assistant", "content": rlhf_pair["chosen"]},
    ]
    
    # Rejected conversation
    rejected_conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rlhf_pair["prompt"]},
        {"role": "assistant", "content": rlhf_pair["rejected"]},
    ]
    
    return {
        "chosen": chosen_conversation,
        "rejected": rejected_conversation,
        "error_type": rlhf_pair["error_type"],
        "metadata": rlhf_pair["metadata"],
    }


def create_rlhf_hf_dataset(
    input_file: str = "data/rlhf_training_v1/all_rlhf_pairs.json",
    output_dir: str = "data/hf_rlhf_dataset_v1",
    train_split: float = 0.9,
    system_prompt_path: str = "extractor_system_prompt.md",
    format_type: str = "dpo",  # dpo, ppo, or conversational
) -> None:
    """
    Create HuggingFace RLHF dataset from preference pairs.
    
    Args:
        format_type: 
            - "dpo": Direct Preference Optimization format
            - "ppo": PPO reward modeling format
            - "conversational": TRL conversational format
    """

    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        print("❌ Error: 'datasets' package not installed")
        print("   Run: pip install datasets")
        return

    # Load RLHF pairs
    with open(input_file, "r", encoding="utf-8") as f:
        rlhf_pairs = json.load(f)

    print(f"📦 Loaded {len(rlhf_pairs)} RLHF preference pairs")
    
    # Count unique datasets and error types
    unique_datasets = len(set(pair["metadata"]["dataset_id"] for pair in rlhf_pairs))
    error_types = {}
    for pair in rlhf_pairs:
        err_type = pair["error_type"]
        error_types[err_type] = error_types.get(err_type, 0) + 1
    
    print(f"   Unique datasets: {unique_datasets}")
    print(f"   Error types: {dict(error_types)}")

    # Convert to specified format
    formatted_examples = []
    
    if format_type == "dpo":
        print(f"\n🔄 Converting to DPO format...")
        for pair in rlhf_pairs:
            formatted_examples.append(create_dpo_example(pair, system_prompt_path))
    
    elif format_type == "ppo":
        print(f"\n🔄 Converting to PPO format...")
        for pair in rlhf_pairs:
            formatted_examples.append(create_ppo_example(pair, system_prompt_path))
    
    elif format_type == "conversational":
        print(f"\n🔄 Converting to Conversational format...")
        for pair in rlhf_pairs:
            formatted_examples.append(create_conversational_pair(pair, system_prompt_path))
    
    else:
        raise ValueError(f"Unknown format_type: {format_type}. Use 'dpo', 'ppo', or 'conversational'")

    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(formatted_examples)

    split_idx = int(len(formatted_examples) * train_split)
    train_examples = formatted_examples[:split_idx]
    val_examples = formatted_examples[split_idx:]

    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    # Create dataset dict
    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    print(f"\n✅ HuggingFace RLHF Dataset Created!")
    print(f"   Format: {format_type.upper()}")
    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Validation: {len(val_dataset)} examples")
    print(f"   Output: {output_path}")
    
    # Save metadata
    metadata = {
        "format": format_type,
        "total_examples": len(formatted_examples),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "unique_datasets": unique_datasets,
        "error_type_distribution": error_types,
        "train_split": train_split,
    }
    
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Error type distribution:")
    for err_type, count in error_types.items():
        pct = count / len(rlhf_pairs) * 100
        print(f"      - {err_type}: {count} ({pct:.1f}%)")
    
    print(f"\n💡 Usage:")
    print(f"   from datasets import load_from_disk")
    print(f"   dataset = load_from_disk('{output_path}')")
    print(f"   train_data = dataset['train']")
    print(f"   val_data = dataset['validation']")


def main():
    parser = argparse.ArgumentParser(
        description="Create HuggingFace RLHF dataset from preference pairs"
    )
    parser.add_argument(
        "--input",
        default="data/rlhf_training_v1/all_rlhf_pairs.json",
        help="Input JSON file with RLHF pairs",
    )
    parser.add_argument(
        "--output",
        default="data/hf_rlhf_dataset_v1",
        help="Output directory for HF dataset",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Train/val split ratio (default: 0.9)",
    )
    parser.add_argument(
        "--system-prompt",
        default="extractor_system_prompt.md",
        help="Path to system prompt file",
    )
    parser.add_argument(
        "--format",
        choices=["dpo", "ppo", "conversational"],
        default="dpo",
        help="Output format: dpo (Direct Preference Optimization), ppo (PPO reward model), or conversational (TRL)",
    )

    args = parser.parse_args()

    create_rlhf_hf_dataset(
        input_file=args.input,
        output_dir=args.output,
        train_split=args.train_split,
        system_prompt_path=args.system_prompt,
        format_type=args.format,
    )


if __name__ == "__main__":
    main()
