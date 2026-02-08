#!/usr/bin/env python3
"""
Create synthetic source CSV datasets for dataset generator.

Generates 25 realistic categorical datasets simulating common domains:
- Retail (grocery, electronics, clothing)
- Healthcare (symptoms, medications, procedures)
- Education (courses, grades, activities)
- Entertainment (movies, music, books)
- Travel (destinations, activities, transportation)
"""

import random
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Seed for reproducibility
random.seed(42)

# =============================================================================
# DOMAIN TEMPLATES
# =============================================================================

DOMAINS = {
    "grocery_store": {
        "items": ["milk", "bread", "butter", "eggs", "cheese", "yogurt", "cereal", 
                 "coffee", "tea", "sugar", "flour", "rice", "pasta", "chicken", 
                 "beef", "fish", "apples", "bananas", "oranges", "tomatoes"],
        "rows": (50, 100),
        "cols": (8, 15),
    },
    "electronics_store": {
        "items": ["laptop", "mouse", "keyboard", "monitor", "headphones", "charger",
                 "cable", "phone", "tablet", "printer", "speaker", "webcam", "router",
                 "hard_drive", "usb_stick", "battery", "case", "adapter"],
        "rows": (40, 80),
        "cols": (6, 12),
    },
    "clothing_store": {
        "items": ["shirt", "pants", "dress", "skirt", "jacket", "coat", "shoes",
                 "boots", "sneakers", "hat", "scarf", "gloves", "belt", "socks",
                 "tie", "sweater", "jeans", "shorts"],
        "rows": (45, 90),
        "cols": (7, 14),
    },
    "pharmacy": {
        "items": ["aspirin", "ibuprofen", "bandages", "thermometer", "vitamins",
                 "cold_medicine", "cough_syrup", "antiseptic", "gauze", "tape",
                 "antibiotic", "pain_relief", "allergy_med", "antacid"],
        "rows": (35, 70),
        "cols": (5, 10),
    },
    "restaurant_orders": {
        "items": ["burger", "fries", "pizza", "salad", "soup", "sandwich", "pasta",
                 "steak", "chicken", "fish", "rice", "vegetables", "dessert", "coffee",
                 "soda", "water", "juice", "wine", "beer"],
        "rows": (60, 120),
        "cols": (8, 16),
    },
    "bookstore": {
        "items": ["fiction", "non_fiction", "mystery", "romance", "scifi", "fantasy",
                 "biography", "history", "science", "art", "cookbook", "textbook",
                 "children", "poetry", "self_help"],
        "rows": (40, 80),
        "cols": (6, 12),
    },
    "movie_rentals": {
        "items": ["action", "comedy", "drama", "horror", "thriller", "romance",
                 "scifi", "fantasy", "documentary", "animation", "musical", "western"],
        "rows": (50, 100),
        "cols": (5, 10),
    },
    "gym_membership": {
        "items": ["weights", "cardio", "yoga", "pilates", "spinning", "swimming",
                 "basketball", "tennis", "boxing", "martial_arts", "dance", "aerobics"],
        "rows": (35, 70),
        "cols": (5, 9),
    },
    "online_courses": {
        "items": ["python", "java", "javascript", "sql", "data_science", "machine_learning",
                 "web_dev", "mobile_dev", "cloud", "devops", "security", "design"],
        "rows": (40, 80),
        "cols": (6, 11),
    },
    "streaming_music": {
        "items": ["pop", "rock", "jazz", "classical", "country", "hip_hop", "electronic",
                 "blues", "reggae", "folk", "metal", "indie", "rnb"],
        "rows": (55, 110),
        "cols": (7, 13),
    },
    "travel_bookings": {
        "items": ["flight", "hotel", "car_rental", "tour", "insurance", "restaurant",
                 "museum", "beach", "mountain", "city", "cruise", "safari"],
        "rows": (45, 90),
        "cols": (6, 12),
    },
    "hospital_visits": {
        "items": ["checkup", "xray", "bloodtest", "vaccination", "surgery", "therapy",
                 "consultation", "prescription", "emergency", "diagnosis"],
        "rows": (40, 80),
        "cols": (5, 9),
    },
    "university_enrollment": {
        "items": ["math", "physics", "chemistry", "biology", "english", "history",
                 "art", "music", "sports", "computer_science", "psychology", "economics"],
        "rows": (50, 100),
        "cols": (6, 12),
    },
    "pet_store": {
        "items": ["dog_food", "cat_food", "leash", "collar", "toy", "bed", "crate",
                 "treats", "shampoo", "brush", "bowl", "litter", "cage"],
        "rows": (35, 70),
        "cols": (5, 10),
    },
    "hardware_store": {
        "items": ["hammer", "screwdriver", "drill", "saw", "nails", "screws", "tape",
                 "paint", "brush", "ladder", "wrench", "pliers", "sandpaper"],
        "rows": (40, 80),
        "cols": (6, 12),
    },
    "beauty_salon": {
        "items": ["haircut", "coloring", "highlights", "styling", "manicure", "pedicure",
                 "facial", "massage", "waxing", "makeup"],
        "rows": (45, 90),
        "cols": (5, 9),
    },
    "coffee_shop": {
        "items": ["espresso", "latte", "cappuccino", "americano", "mocha", "tea",
                 "pastry", "sandwich", "muffin", "cookie", "bagel"],
        "rows": (60, 120),
        "cols": (6, 11),
    },
    "game_store": {
        "items": ["action_game", "rpg", "strategy", "sports", "racing", "puzzle",
                 "adventure", "simulation", "fighting", "platform"],
        "rows": (40, 80),
        "cols": (5, 10),
    },
    "car_parts": {
        "items": ["oil", "filter", "battery", "tires", "brakes", "spark_plugs",
                 "belts", "wipers", "lights", "mirrors"],
        "rows": (35, 70),
        "cols": (5, 9),
    },
    "office_supplies": {
        "items": ["paper", "pens", "pencils", "folders", "binders", "stapler",
                 "tape", "scissors", "calculator", "notebook", "markers"],
        "rows": (45, 90),
        "cols": (6, 11),
    },
    "garden_center": {
        "items": ["flowers", "seeds", "soil", "fertilizer", "pots", "tools",
                 "hose", "gloves", "plants", "mulch"],
        "rows": (40, 80),
        "cols": (5, 10),
    },
    "toy_store": {
        "items": ["action_figures", "dolls", "puzzles", "board_games", "lego",
                 "building_blocks", "stuffed_animals", "cars", "trains"],
        "rows": (50, 100),
        "cols": (6, 12),
    },
    "bakery": {
        "items": ["bread", "croissant", "bagel", "muffin", "cake", "pie", "cookies",
                 "donuts", "rolls", "baguette"],
        "rows": (55, 110),
        "cols": (6, 11),
    },
    "sports_equipment": {
        "items": ["basketball", "football", "soccer_ball", "tennis_racket", "golf_clubs",
                 "baseball_bat", "hockey_stick", "bike", "skateboard", "helmet"],
        "rows": (40, 80),
        "cols": (5, 10),
    },
    "jewelry_store": {
        "items": ["ring", "necklace", "bracelet", "earrings", "watch", "pendant",
                 "chain", "brooch"],
        "rows": (35, 70),
        "cols": (4, 8),
    },
}


def generate_binary_dataset(domain_name: str, config: Dict, num_rows: int, num_cols: int) -> pd.DataFrame:
    """Generate a binary (0/1) transaction dataset."""
    items = random.sample(config["items"], min(num_cols, len(config["items"])))
    
    data = {}
    for item in items:
        # Each item has 20-60% probability of appearing in a transaction
        prob = random.uniform(0.2, 0.6)
        data[item] = [1 if random.random() < prob else 0 for _ in range(num_rows)]
    
    return pd.DataFrame(data)


def generate_categorical_dataset(domain_name: str, config: Dict, num_rows: int, num_cols: int) -> pd.DataFrame:
    """Generate a categorical dataset with column-specific values."""
    items = config["items"]
    
    # Create columns with different categorical values
    data = {}
    for col_idx in range(num_cols):
        col_name = f"attr_{col_idx + 1}"
        # Each column gets 3-6 possible values
        num_values = random.randint(3, min(6, len(items)))
        possible_values = random.sample(items, num_values)
        
        # Generate column data with some None values (10-30% missing)
        col_data = []
        for _ in range(num_rows):
            if random.random() < random.uniform(0.1, 0.3):
                col_data.append(None)
            else:
                col_data.append(random.choice(possible_values))
        
        data[col_name] = col_data
    
    return pd.DataFrame(data)


def generate_source_dataset(domain_name: str, config: Dict, dataset_format: str) -> pd.DataFrame:
    """Generate a single source dataset."""
    min_rows, max_rows = config["rows"]
    min_cols, max_cols = config["cols"]
    
    num_rows = random.randint(min_rows, max_rows)
    num_cols = random.randint(min_cols, max_cols)
    
    if dataset_format == "binary":
        return generate_binary_dataset(domain_name, config, num_rows, num_cols)
    else:
        return generate_categorical_dataset(domain_name, config, num_rows, num_cols)


def main():
    """Generate all source datasets."""
    output_dir = Path("real_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🏭 Generating Source Datasets")
    print("=" * 60)
    
    domain_names = list(DOMAINS.keys())
    
    # Generate 25 datasets (one per domain, with binary/categorical mix)
    dataset_idx = 1
    for domain_name in domain_names:
        config = DOMAINS[domain_name]
        
        # Alternate between binary and categorical formats
        dataset_format = "binary" if dataset_idx % 2 == 1 else "categorical"
        
        print(f"\n[{dataset_idx:02d}/25] {domain_name} ({dataset_format})...", end=" ")
        
        df = generate_source_dataset(domain_name, config, dataset_format)
        
        # Save to CSV
        filename = f"source_{dataset_idx:02d}_{domain_name}.csv"
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)
        
        print(f"✅ Saved {len(df)} rows × {len(df.columns)} cols → {filename}")
        
        dataset_idx += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Generated {len(domain_names)} source datasets in {output_dir}/")
    print("\nNext step: Run generate_datasets_v2.py to create 500 training datasets")


if __name__ == "__main__":
    main()
