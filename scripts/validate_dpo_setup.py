#!/usr/bin/env python3
"""Validate DPO training setup and dataset format."""

from datasets import load_from_disk
from pathlib import Path
import sys

def validate_dataset():
    """Check RLHF dataset format."""
    print("📊 Validating RLHF dataset format...")
    
    dataset_path = Path("data/hf_rlhf_dataset_v1")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    try:
        dataset = load_from_disk(str(dataset_path))
        print(f"✅ Dataset loaded successfully")
        print(f"   Train: {len(dataset['train'])} examples")
        print(f"   Validation: {len(dataset['validation'])} examples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Check format
    print("\n📋 Validating DPO format...")
    example = dataset["train"][0]
    required_fields = ["prompt", "chosen", "rejected"]
    
    for field in required_fields:
        if field in example:
            print(f"   ✓ {field} field present")
        else:
            print(f"   ✗ {field} field MISSING")
            return False
    
    # Check message structure
    print("\n📝 Validating message structure...")
    try:
        assert isinstance(example["prompt"], list), "prompt should be list"
        assert isinstance(example["chosen"], list), "chosen should be list"
        assert isinstance(example["rejected"], list), "rejected should be list"
        
        assert len(example["prompt"]) >= 1, "prompt should have messages"
        assert len(example["chosen"]) == 1, "chosen should have 1 message"
        assert len(example["rejected"]) == 1, "rejected should have 1 message"
        
        print(f"   ✓ Prompt: {len(example['prompt'])} messages")
        print(f"   ✓ Chosen: {len(example['chosen'])} messages")
        print(f"   ✓ Rejected: {len(example['rejected'])} messages")
    except AssertionError as e:
        print(f"   ✗ {e}")
        return False
    
    print("\n✅ RLHF dataset format valid for DPO training!")
    return True


def validate_script():
    """Check DPO training script exists."""
    print("\n📄 Validating DPO training script...")
    
    script_path = Path("src/training/run_dpo_training.py")
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    print(f"✅ Training script found: {script_path}")
    
    # Check script has main components
    content = script_path.read_text()
    required = ["DPOTrainer", "DPOConfig", "beta", "ScriptArguments"]
    
    for component in required:
        if component in content:
            print(f"   ✓ {component}")
        else:
            print(f"   ✗ {component} missing")
            return False
    
    print("\n✅ DPO training script validated!")
    return True


def main():
    """Run all validations."""
    print("=" * 60)
    print("DPO TRAINING VALIDATION")
    print("=" * 60)
    
    results = []
    
    # Validate dataset
    results.append(validate_dataset())
    
    # Validate script
    results.append(validate_script())
    
    # Summary
    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL VALIDATIONS PASSED")
        print("\nReady for DPO training!")
        print("\nTo train:")
        print("  python src/training/run_dpo_training.py \\")
        print("    --dataset_path data/hf_rlhf_dataset_v1 \\")
        print("    --use_4bit --use_lora --beta 0.1")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
