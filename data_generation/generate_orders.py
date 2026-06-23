"""
Generates raw orders (orders.csv): one row per order, with the full final
cart as a list of item_ids, built up in a fixed Main -> Side -> Drink ->
Dessert sequence (mirrors a real food-delivery cart-building flow: you
always pick a main first, then optionally add on).

FIX vs. original: "Italian" is used consistently (matches generate_items.py
and the city-bias maps below), so sampling never hits an empty
cuisine/category slice.
"""

import pandas as pd
import numpy as np
import random

from generate_items import CUISINES

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N_ORDERS = 10000

# Soft prior over cuisine choice per city - used only to make the order mix
# look like a real city (e.g. Hyderabad skews Mughlai). This is NOT used as
# a label-generating rule anywhere downstream; it just shapes which carts
# get created in the first place.
CITY_CUISINE_PRIOR = {
    "Bangalore":  {"Italian": 0.30, "South Indian": 0.30, "Chinese": 0.20, "North Indian": 0.10, "Mughlai": 0.10},
    "Delhi":      {"Mughlai": 0.35, "North Indian": 0.30, "Chinese": 0.15, "Italian": 0.10, "South Indian": 0.10},
    "Mumbai":     {"North Indian": 0.25, "Chinese": 0.25, "Italian": 0.20, "Mughlai": 0.15, "South Indian": 0.15},
    "Hyderabad":  {"Mughlai": 0.40, "North Indian": 0.25, "Chinese": 0.15, "Italian": 0.10, "South Indian": 0.10},
    # ── New cities added for 10-city expansion ──────────────────────────
    # Each distribution reflects real culinary identity, not copy-paste.
    "Chennai":    {"South Indian": 0.45, "Chinese": 0.15, "North Indian": 0.15, "Mughlai": 0.10, "Italian": 0.15},
    "Kolkata":    {"Chinese": 0.25, "North Indian": 0.25, "Mughlai": 0.20, "South Indian": 0.10, "Italian": 0.20},
    "Pune":       {"North Indian": 0.20, "Italian": 0.25, "Chinese": 0.25, "South Indian": 0.15, "Mughlai": 0.15},
    "Ahmedabad":  {"North Indian": 0.30, "South Indian": 0.15, "Chinese": 0.20, "Italian": 0.20, "Mughlai": 0.15},
    "Jaipur":     {"North Indian": 0.40, "Mughlai": 0.25, "Chinese": 0.15, "Italian": 0.10, "South Indian": 0.10},
    "Lucknow":    {"Mughlai": 0.40, "North Indian": 0.30, "Chinese": 0.10, "Italian": 0.10, "South Indian": 0.10},
}
CITIES = list(CITY_CUISINE_PRIOR.keys())
MEAL_TIMES = ["Breakfast", "Lunch", "Snacks", "Dinner"]
MEAL_TIME_WEIGHTS = [0.2, 0.35, 0.2, 0.25]


def add_side_probability(cuisine: str) -> float:
    return {"Mughlai": 0.7, "North Indian": 0.6, "Chinese": 0.5, "Italian": 0.4}.get(cuisine, 0.3)


def add_drink_probability(cuisine: str) -> float:
    return {"Italian": 0.7, "Chinese": 0.6}.get(cuisine, 0.4)


def add_dessert_probability(meal_time: str) -> float:
    return 0.5 if meal_time == "Dinner" else 0.2


def build_orders(item_df: pd.DataFrame) -> pd.DataFrame:
    orders = []

    for order_id in range(N_ORDERS):
        city = random.choice(CITIES)
        meal_time = random.choices(MEAL_TIMES, weights=MEAL_TIME_WEIGHTS, k=1)[0]

        cuisine_probs = CITY_CUISINE_PRIOR[city]
        cuisine = np.random.choice(list(cuisine_probs.keys()), p=list(cuisine_probs.values()))

        cuisine_items = item_df[item_df["cuisine"] == cuisine]

        main_item = cuisine_items[cuisine_items["category"] == "Main"].sample(1).iloc[0]
        cart = [int(main_item["item_id"])]

        if random.random() < add_side_probability(cuisine):
            side = cuisine_items[cuisine_items["category"] == "Side"].sample(1).iloc[0]
            cart.append(int(side["item_id"]))

        if random.random() < add_drink_probability(cuisine):
            drink = cuisine_items[cuisine_items["category"] == "Drink"].sample(1).iloc[0]
            cart.append(int(drink["item_id"]))

        if random.random() < add_dessert_probability(meal_time):
            dessert = cuisine_items[cuisine_items["category"] == "Dessert"].sample(1).iloc[0]
            cart.append(int(dessert["item_id"]))

        orders.append({
            "order_id": order_id,
            "city": city,
            "meal_time": meal_time,
            "cuisine": cuisine,
            "items": cart,
        })

    return pd.DataFrame(orders)


if __name__ == "__main__":
    item_df = pd.read_csv("items.csv")
    orders_df = build_orders(item_df)
    orders_df.to_csv("orders.csv", index=False)
    print("Orders generated:", len(orders_df))
    print("Avg cart size:", orders_df["items"].apply(len).mean().round(2))
