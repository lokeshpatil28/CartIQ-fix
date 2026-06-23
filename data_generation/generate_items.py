"""
Generates the item catalog (items.csv).

FIX vs. original: cuisine list is now canonical and consistent across the
entire pipeline (data_generation, training, API, frontend). The original
project used "Pizza" here but "Italian" everywhere else (orders, city-bias
maps, frontend dropdown), which silently broke any flow touching Italian
cuisine. There is exactly one source of truth for this list now: CUISINES
below. Every other file imports it from here rather than re-typing it.
"""

import pandas as pd
import numpy as np
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Single source of truth for cuisines - import this instead of re-declaring it.
CUISINES = ["North Indian", "Mughlai", "Chinese", "Italian", "South Indian"]

CATEGORIES = {
    "Main": 0.4,
    "Side": 0.25,
    "Drink": 0.2,
    "Dessert": 0.15,
}

PRICE_RANGES = {
    "Main": (180, 450),
    "Side": (80, 200),
    "Drink": (40, 150),
    "Dessert": (90, 220),
}

N_ITEMS = 150
# Minimum items guaranteed per (cuisine, category) cell so every combination
# used downstream (candidate generation, order building) always has stock.
MIN_PER_CELL = 3


def generate_price(category: str) -> int:
    lo, hi = PRICE_RANGES[category]
    return int(np.random.randint(lo, hi))


def build_items() -> pd.DataFrame:
    items = []
    item_id = 1

    # Guarantee coverage: at least MIN_PER_CELL items per cuisine x category.
    for cuisine in CUISINES:
        for category in CATEGORIES:
            for _ in range(MIN_PER_CELL):
                items.append({
                    "item_id": item_id,
                    "item_name": f"{cuisine.replace(' ', '')}_{category}_{item_id}",
                    "cuisine": cuisine,
                    "category": category,
                    "veg_flag": int(np.random.choice([0, 1], p=[0.6, 0.4])),
                    "price": generate_price(category),
                    "popularity_score": round(float(np.random.uniform(0.1, 1.0)), 3),
                    "margin": round(float(np.random.uniform(0.2, 0.6)), 3),
                })
                item_id += 1

    # Fill remaining slots randomly until we hit N_ITEMS.
    while item_id <= N_ITEMS:
        cuisine = random.choice(CUISINES)
        category = np.random.choice(list(CATEGORIES.keys()), p=list(CATEGORIES.values()))
        items.append({
            "item_id": item_id,
            "item_name": f"{cuisine.replace(' ', '')}_{category}_{item_id}",
            "cuisine": cuisine,
            "category": category,
            "veg_flag": int(np.random.choice([0, 1], p=[0.6, 0.4])),
            "price": generate_price(category),
            "popularity_score": round(float(np.random.uniform(0.1, 1.0)), 3),
            "margin": round(float(np.random.uniform(0.2, 0.6)), 3),
        })
        item_id += 1

    return pd.DataFrame(items)


if __name__ == "__main__":
    item_df = build_items()
    item_df.to_csv("items.csv", index=False)
    print("Items generated:", len(item_df))
    print(item_df.groupby(["cuisine", "category"]).size())
