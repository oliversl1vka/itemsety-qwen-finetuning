#!/usr/bin/env python3
"""Quick test to verify dataset loading works correctly."""

from datasets import load_from_disk

print("=" * 60)
print("TESTING DATASET LOADING")
print("=" * 60)

# Load dataset
print("\nLoading from: ./hf_dataset_enhanced")
dataset = load_from_disk("./hf_dataset_enhanced")

print(f"\n✅ Dataset loaded successfully!")
print(f"   Train examples: {len(dataset['train'])}")
print(f"   Validation examples: {len(dataset['validation'])}")
print(f"   Columns: {dataset['train'].column_names}")

# Verify structure
example = dataset['train'][0]
print(f"\n✅ First example structure:")
print(f"   Keys: {list(example.keys())}")
print(f"   Messages count: {len(example['messages'])}")
print(f"   First message role: {example['messages'][0]['role']}")
print(f"   Content preview: {example['messages'][0]['content'][:100]}...")

# Check metadata
if 'metadata' in example:
    print(f"\n✅ Metadata present:")
    for key in example['metadata'].keys():
        print(f"   - {key}: {example['metadata'][key]}")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED - READY FOR TRAINING!")
print("=" * 60)
