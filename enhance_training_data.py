"""
Enhance training data with human-readable context and triplet format.
Based on findings: synthetic-only data fails, real-word context improves performance.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import random

# Real-world item mappings for context
ITEM_CONTEXT_MAPPINGS = {
    "grocery": [
        "milk", "bread", "eggs", "butter", "cheese", "yogurt", "apple", "banana",
        "orange", "tomato", "potato", "onion", "chicken", "beef", "fish", "rice",
        "pasta", "cereal", "coffee", "tea", "sugar", "flour", "salt", "pepper"
    ],
    "electronics": [
        "laptop", "phone", "tablet", "monitor", "keyboard", "mouse", "headphones",
        "speaker", "camera", "charger", "cable", "battery", "memory_card", "router",
        "printer", "scanner", "webcam", "microphone", "smartwatch", "earbuds"
    ],
    "clothing": [
        "shirt", "pants", "dress", "skirt", "jacket", "coat", "shoes", "boots",
        "sneakers", "hat", "cap", "scarf", "gloves", "socks", "belt", "tie",
        "sweater", "hoodie", "jeans", "shorts"
    ],
    "household": [
        "detergent", "soap", "shampoo", "toothpaste", "tissue", "towel", "plate",
        "cup", "fork", "knife", "spoon", "bowl", "pan", "pot", "cleaning_spray",
        "sponge", "trash_bag", "light_bulb", "battery", "tape"
    ]
}

def generate_item_context(item: str, domain: str = None) -> str:
    """Generate human-readable context for synthetic items"""
    if domain is None:
        domain = random.choice(list(ITEM_CONTEXT_MAPPINGS.keys()))
    
    # Map synthetic item to real-world equivalent
    items = ITEM_CONTEXT_MAPPINGS[domain]
    idx = hash(item) % len(items)
    real_item = items[idx]
    
    return f"{real_item} ({item})"

def enhance_example_with_context(
    example: Dict[str, Any],
    add_triplets: bool = True
) -> Dict[str, Any]:
    """
    Enhance training example with:
    1. Human-readable item context
    2. Triplet format (if requested)
    """
    
    # Choose domain for this example
    domain = random.choice(list(ITEM_CONTEXT_MAPPINGS.keys()))
    
    # Enhance CSV context with real-world items
    csv_lines = example['csv_context'].strip().split('\n')
    enhanced_csv = []
    
    for line in csv_lines:
        parts = line.split(',')
        if len(parts) > 1:  # Has items
            # Map items to real-world context
            enhanced_items = []
            for item in parts[1:]:
                item = item.strip()
                if item:
                    enhanced_items.append(generate_item_context(item, domain))
            enhanced_csv.append(f"{parts[0]},{','.join(enhanced_items)}")
        else:
            enhanced_csv.append(line)
    
    enhanced_context = '\n'.join(enhanced_csv)
    
    # Enhance ground truth itemsets
    enhanced_ground_truth = []
    for itemset_obj in example['ground_truth']:
        enhanced_itemset = {
            'itemset': [
                generate_item_context(item, domain)
                for item in itemset_obj['itemset']
            ],
            'count': itemset_obj['count'],
            'evidence_rows': itemset_obj.get('evidence_rows', []),
            'explanation': itemset_obj.get('explanation', f"Frequent itemset of size {len(itemset_obj['itemset'])}")
        }
        
        # Add triplet format if requested and size >= 3
        if add_triplets and len(enhanced_itemset['itemset']) >= 3:
            enhanced_itemset['triplet'] = enhanced_itemset['itemset'][:3]
        
        enhanced_ground_truth.append(enhanced_itemset)
    
    return {
        'id': example['id'],
        'dataset_name': example['dataset_name'],
        'csv_context': enhanced_context,
        'ground_truth': enhanced_ground_truth,
        'min_support': example['min_support'],
        'domain': domain,  # Add domain metadata
    }

def enhance_training_data(
    input_file: str = "training_data/all_training_examples.json",
    output_file: str = "training_data/all_training_examples_enhanced.json",
    add_triplets: bool = True,
):
    """Enhance all training examples with context and triplets"""
    
    print("=" * 60)
    print("ENHANCING TRAINING DATA")
    print("=" * 60)
    
    # Load original examples
    with open(input_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    print(f"\n📦 Loaded {len(examples)} training examples")
    
    # Enhance each example
    enhanced_examples = []
    for i, example in enumerate(examples, 1):
        enhanced = enhance_example_with_context(example, add_triplets)
        enhanced_examples.append(enhanced)
        
        if i % 10 == 0:
            print(f"   Enhanced {i}/{len(examples)} examples...")
    
    # Save enhanced examples
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_examples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Enhanced training data saved!")
    print(f"   Output: {output_path}")
    print(f"   Total examples: {len(enhanced_examples)}")
    
    # Show example
    print("\n" + "=" * 60)
    print("SAMPLE ENHANCED EXAMPLE")
    print("=" * 60)
    example = enhanced_examples[0]
    print(f"\nDomain: {example['domain']}")
    print(f"\nOriginal CSV (first 3 lines):")
    print('\n'.join(example['csv_context'].split('\n')[:3]))
    print(f"\nFirst itemset:")
    print(json.dumps(example['ground_truth'][0], indent=2, ensure_ascii=False))
    
    return enhanced_examples

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhance training data with context")
    parser.add_argument("--input", default="training_data/all_training_examples.json")
    parser.add_argument("--output", default="training_data/all_training_examples_enhanced.json")
    parser.add_argument("--triplets", action="store_true", default=True, help="Add triplet format")
    
    args = parser.parse_args()
    
    enhance_training_data(
        input_file=args.input,
        output_file=args.output,
        add_triplets=args.triplets,
    )

if __name__ == "__main__":
    main()
