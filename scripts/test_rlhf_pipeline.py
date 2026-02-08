#!/usr/bin/env python3
"""
Test script to validate RLHF dataset creation pipeline.
"""

import json
from pathlib import Path


def test_rlhf_export():
    """Test RLHF preference pair export"""
    print("=" * 60)
    print("🧪 Testing RLHF Export Pipeline")
    print("=" * 60)
    
    # Check if runs.db exists
    if not Path("runs.db").exists():
        print("❌ runs.db not found!")
        print("   Run: python pipeline.py --data-dir data/datasets_v2 --llm-full")
        return False
    
    print("✅ Found runs.db")
    
    # Check validated runs count
    import sqlite3
    conn = sqlite3.connect("runs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM runs WHERE validation_passed = 1")
    count = cursor.fetchone()[0]
    conn.close()
    
    print(f"✅ Found {count} validated runs")
    
    if count == 0:
        print("❌ No validated runs in database!")
        return False
    
    return True


def test_error_generation():
    """Test rejected response generation"""
    print("\n" + "=" * 60)
    print("🧪 Testing Error Generation")
    print("=" * 60)
    
    # Sample ground truth
    sample_gt = [
        {"itemset": ["A", "B"], "count": 5, "rows": ["Row 1", "Row 2", "Row 3", "Row 4", "Row 5"]},
        {"itemset": ["A", "C"], "count": 4, "rows": ["Row 1", "Row 2", "Row 6", "Row 7"]},
        {"itemset": ["B", "C"], "count": 3, "rows": ["Row 2", "Row 3", "Row 8"]},
    ]
    
    # Import error generator
    import sys
    sys.path.insert(0, "src/training")
    from export_rlhf_training_data import generate_rejected_responses
    
    rejected_variants = generate_rejected_responses(sample_gt, min_support=3, num_variants=3)
    
    print(f"\n✅ Generated {len(rejected_variants)} error variants:")
    for i, (rejected, error_type) in enumerate(rejected_variants, 1):
        print(f"\n   Variant {i}: {error_type}")
        print(f"      Original itemsets: {len(sample_gt)}")
        print(f"      Rejected itemsets: {len(rejected)}")
        
        # Check differences
        if error_type == "hallucination":
            if len(rejected) > len(sample_gt):
                print(f"      ✅ Added fake itemsets")
            else:
                print(f"      ⚠️  Expected more itemsets")
        
        elif error_type == "missing_itemsets":
            if len(rejected) < len(sample_gt):
                print(f"      ✅ Removed valid itemsets")
            else:
                print(f"      ⚠️  Expected fewer itemsets")
        
        elif error_type == "wrong_counts":
            count_diffs = sum(1 for j in range(min(len(sample_gt), len(rejected))) 
                            if sample_gt[j]["count"] != rejected[j]["count"])
            print(f"      ✅ Corrupted {count_diffs} counts")
    
    return True


def test_dpo_format():
    """Test DPO dataset format"""
    print("\n" + "=" * 60)
    print("🧪 Testing DPO Format")
    print("=" * 60)
    
    # Check if system prompt exists
    if not Path("extractor_system_prompt.md").exists():
        print("❌ extractor_system_prompt.md not found!")
        return False
    
    print("✅ Found system prompt")
    
    # Create sample RLHF pair
    sample_pair = {
        "prompt": "Dataset: test.csv\nFind itemsets...",
        "chosen": '[{"itemset": ["A", "B"], "count": 5, "rows": ["Row 1"]}]',
        "rejected": '[{"itemset": ["X", "Y"], "count": 3, "rows": ["Row 99"]}]',
        "error_type": "hallucination",
        "metadata": {"dataset_id": "test_001"}
    }
    
    # Import formatter
    import sys
    sys.path.insert(0, "src/training")
    from create_rlhf_hf_dataset import create_dpo_example
    
    # Mock tokenizer with apply_chat_template
    class MockTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            text = ""
            for msg in messages:
                text += f"[{msg['role'].upper()}]\n{msg['content']}\n\n"
            if add_generation_prompt:
                text += "[ASSISTANT]\n"
            return text
    
    # Create DPO example
    dpo_example = create_dpo_example(sample_pair, "extractor_system_prompt.md")
    
    # Validate structure
    required_keys = ["prompt", "chosen", "rejected", "error_type", "metadata"]
    missing_keys = [k for k in required_keys if k not in dpo_example]
    
    if missing_keys:
        print(f"❌ Missing keys: {missing_keys}")
        return False
    
    print("✅ DPO example structure valid")
    print(f"   Keys: {list(dpo_example.keys())}")
    print(f"   Prompt messages: {len(dpo_example['prompt'])}")
    print(f"   Chosen messages: {len(dpo_example['chosen'])}")
    print(f"   Rejected messages: {len(dpo_example['rejected'])}")
    
    return True


def test_full_pipeline():
    """Test complete RLHF pipeline"""
    print("\n" + "=" * 60)
    print("🧪 Testing Full RLHF Pipeline")
    print("=" * 60)
    
    steps = [
        ("Database check", test_rlhf_export),
        ("Error generation", test_error_generation),
        ("DPO format", test_dpo_format),
    ]
    
    results = []
    for step_name, test_func in steps:
        try:
            result = test_func()
            results.append((step_name, result))
        except Exception as e:
            print(f"\n❌ {step_name} failed: {e}")
            results.append((step_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    for step_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {step_name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\n📝 Next steps:")
        print("   1. python src/training/export_rlhf_training_data.py")
        print("   2. python src/training/create_rlhf_hf_dataset.py --format dpo")
        print("   3. python src/training/run_dpo_training.py")
    else:
        print("\n⚠️  Some tests failed. Fix issues before proceeding.")
    
    return all_passed


if __name__ == "__main__":
    success = test_full_pipeline()
    exit(0 if success else 1)
