"""
Generate evaluation datasets with REPEATING items (like grocery basket data).
These are designed to test the fine-tuned itemset extraction model.
"""

import csv
import random
from pathlib import Path

# Jednoduché, opakovateľné položky (ako v úspešnom teste)
ITEM_POOLS = {
    "grocery": ["milk", "bread", "eggs", "butter", "cheese", "yogurt", "juice", "coffee", "tea", "sugar"],
    "letters": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "fruits": ["apple", "banana", "orange", "grape", "mango", "kiwi", "pear", "peach"],
    "colors": ["red", "blue", "green", "yellow", "black", "white", "pink", "gray"],
}

def generate_eval_dataset(
    output_path: Path,
    num_rows: int,
    items_per_row: int,
    item_pool: list,
    repeat_probability: float = 0.7,  # Vysoká pravdepodobnosť opakovania
):
    """Generate a dataset with repeating items."""
    
    # Vyber subset položiek pre tento dataset
    num_items = min(len(item_pool), items_per_row + 3)
    selected_items = random.sample(item_pool, num_items)
    
    rows = []
    for _ in range(num_rows):
        # Náhodný počet položiek v tomto riadku
        row_size = random.randint(2, items_per_row)
        
        # Vyber položky s vysokou pravdepodobnosťou opakovania
        if random.random() < repeat_probability:
            # Preferuj prvých niekoľko položiek (vytvára frekvenčné vzory)
            weights = [3.0 if i < 4 else 1.0 for i in range(len(selected_items))]
            row_items = random.choices(selected_items, weights=weights, k=row_size)
        else:
            row_items = random.sample(selected_items, min(row_size, len(selected_items)))
        
        # Odstráň duplikáty v riadku
        row_items = list(dict.fromkeys(row_items))
        rows.append(row_items)
    
    # Ulož ako CSV
    max_cols = max(len(r) for r in rows)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Hlavička
        writer.writerow([f"item{i+1}" for i in range(max_cols)])
        # Dáta
        for row in rows:
            padded = row + [''] * (max_cols - len(row))
            writer.writerow(padded)
    
    return rows


def main():
    output_dir = Path("eval_datasets")
    output_dir.mkdir(exist_ok=True)
    
    random.seed(42)
    
    # Definícia datasetov - od najmenších po najväčšie
    datasets = [
        # Veľmi malé (ako v úspešnom teste)
        {"rows": 5, "items": 4, "pool": "grocery", "name": "tiny_grocery"},
        {"rows": 5, "items": 4, "pool": "letters", "name": "tiny_letters"},
        
        # Malé
        {"rows": 10, "items": 5, "pool": "grocery", "name": "small_grocery"},
        {"rows": 10, "items": 5, "pool": "fruits", "name": "small_fruits"},
        
        # Stredné
        {"rows": 15, "items": 6, "pool": "colors", "name": "medium_colors"},
        {"rows": 15, "items": 6, "pool": "grocery", "name": "medium_grocery"},
        
        # Väčšie
        {"rows": 20, "items": 7, "pool": "letters", "name": "large_letters"},
        {"rows": 20, "items": 7, "pool": "fruits", "name": "large_fruits"},
        
        # Ešte väčšie
        {"rows": 25, "items": 8, "pool": "grocery", "name": "xlarge_grocery"},
        {"rows": 30, "items": 8, "pool": "colors", "name": "xlarge_colors"},
    ]
    
    print(f"📊 Generating {len(datasets)} evaluation datasets...")
    print("=" * 60)
    
    for i, cfg in enumerate(datasets, 1):
        pool = ITEM_POOLS[cfg["pool"]]
        filename = f"eval_{i:02d}_{cfg['name']}_{cfg['rows']}x{cfg['items']}.csv"
        filepath = output_dir / filename
        
        rows = generate_eval_dataset(
            filepath,
            num_rows=cfg["rows"],
            items_per_row=cfg["items"],
            item_pool=pool,
        )
        
        # Spočítaj frekventné položky
        item_counts = {}
        for row in rows:
            for item in row:
                item_counts[item] = item_counts.get(item, 0) + 1
        frequent = sum(1 for c in item_counts.values() if c >= 3)
        
        print(f"   {i:2d}. {filename}")
        print(f"       {cfg['rows']} rows × {cfg['items']} items, {frequent} items with count≥3")
    
    print("=" * 60)
    print(f"✅ Generated {len(datasets)} datasets in {output_dir}/")
    print("\n📋 Datasets sorted by size (smallest first) for incremental testing")


if __name__ == "__main__":
    main()
