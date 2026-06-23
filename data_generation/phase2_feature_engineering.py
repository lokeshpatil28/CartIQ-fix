"""
Builds the feature matrix used for training and (via the same functions,
imported by the API) for serving.

============================================================================
WHY THIS FILE WAS REWRITTEN
============================================================================
The original version computed features like `is_dinner_dessert` and
`city_cuisine_match` using literally the same if/else predicates as the
label generator, and even reconstructed the label generator's weighted sum
verbatim as a feature (`contextual_score_boost`). That's not feature
engineering, it's copying the answer key into the exam.

This version only uses information a real system would actually have at
serving time: the cart's contents, the candidate's metadata, and coarse
context (city/meal_time/cuisine as categorical descriptors, not as
predicates pre-matched to a label rule). It does NOT know about, or try to
reverse-engineer, generate_training_data.py's hidden affinity matrices -
by design, since in a real system there is no generator to peek at.

Train/serve consistency: build_features() and aggregate_cart() here are the
single source of truth, imported directly by backend/api.py. There is now
exactly one feature-computation code path, not two independently
maintained copies (which was the original train/serve-skew risk).
============================================================================
"""

import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder

MEAL_MAP = {"Breakfast": 0, "Lunch": 1, "Snacks": 2, "Dinner": 3}
CITY_LIST = ["Bangalore", "Delhi", "Hyderabad", "Mumbai", "Chennai", "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
CITY_MAP = {city: idx for idx, city in enumerate(CITY_LIST)}

# Final feature order. Both training and the API import this list directly
# so column order can never silently drift between the two.
FEATURE_COLUMNS = [
    "candidate_price",
    "candidate_popularity_score",
    "candidate_margin",
    "candidate_veg_flag",
    "cart_total_price",
    "cart_size",
    "number_of_sides",
    "number_of_drinks",
    "veg_ratio",
    "has_side",
    "has_drink",
    "meal_time",
    "city_encoded",
    "price_ratio",
    "popularity_weighted_signal",
    "candidate_category_main",
    "candidate_category_side",
    "candidate_category_drink",
    "candidate_category_dessert",
]
# NOTE on dropped columns, found during EDA on the generated data - all are
# structurally constant given how generate_orders.py / build_snapshots.py
# construct carts, not bugs in the feature computation itself:
#  - cuisine_match: always 1, candidates are drawn only from the same
#    cuisine as the order (generate_training_data.py).
#  - number_of_mains / has_main: always 1 / True. Every cart always
#    contains exactly one Main, added first by generate_orders.py.
#  - number_of_desserts / has_dessert: always 0. Dessert is always the
#    last item added when present, and build_snapshots.py never includes
#    an order's final item inside cart_items - so dessert can appear as a
#    next_item candidate but never as existing cart content.
# Worth knowing cold if asked "why isn't X in your feature set" in review -
# the honest answer is EDA caught these as constants and they were
# removed, not that they were overlooked.


def aggregate_cart(cart_list, item_dict):
    """Pure cart-composition aggregates. No knowledge of the candidate.

    Note: main and dessert counts are intentionally not tracked. Every
    cart always contains exactly one Main (generate_orders.py always adds
    it first), and dessert is always cart-terminal (see FEATURE_COLUMNS
    note above) - both would be structurally constant columns, not real
    signal, for every snapshot in this generator.
    """
    total_price = 0
    sides = drinks = veg = 0

    for item_id in cart_list:
        item = item_dict.get(item_id)
        if item is None:
            continue
        total_price += item["price"]
        cat = item["category"].lower()
        if cat == "side":
            sides += 1
        elif cat == "drink":
            drinks += 1
        veg += item["veg_flag"]

    size = len(cart_list)
    veg_ratio = veg / size if size > 0 else 0

    return {
        "cart_total_price": total_price,
        "cart_size": size,
        "number_of_sides": sides,
        "number_of_drinks": drinks,
        "veg_ratio": veg_ratio,
        "has_side": int(sides > 0),
        "has_drink": int(drinks > 0),
    }


def build_features(cart_items, candidate_item, city, meal_time, cuisine, item_dict):
    """
    Builds one feature row for a single (cart, candidate) pair.
    Used identically by training (in bulk, below) and by the live API
    (backend/api.py), so there is exactly one feature-computation path.
    """
    cart_agg = aggregate_cart(cart_items, item_dict)
    candidate_data = item_dict[candidate_item]

    candidate_price = candidate_data["price"]
    candidate_category = candidate_data["category"]
    candidate_cuisine = candidate_data["cuisine"]

    meal_encoded = MEAL_MAP.get(meal_time, 0)
    city_encoded = CITY_MAP.get(city, 0)

    # NOTE: cuisine_match is intentionally not computed - see FEATURE_COLUMNS
    # note above. Candidates are always drawn from the same cuisine as the
    # order, so candidate_cuisine == cuisine is true for 100% of rows; a
    # constant column adds nothing and was dropped after EDA.

    price_ratio = candidate_price / (cart_agg["cart_total_price"] + 1)
    popularity_weighted_signal = candidate_data["popularity_score"] * cart_agg["cart_size"]

    # One-hot the candidate's own category. This is a plain descriptor of
    # the candidate, not a precomputed match against a meal_time rule - the
    # model has to learn any meal_time x category relationship itself from
    # meal_time (numeric) and these flags, rather than being handed the
    # cross-product pre-computed.
    cat_lower = candidate_category.lower()

    row = {
        "candidate_price": candidate_price,
        "candidate_popularity_score": candidate_data["popularity_score"],
        "candidate_margin": candidate_data["margin"],
        "candidate_veg_flag": candidate_data["veg_flag"],
        "meal_time": meal_encoded,
        "city_encoded": city_encoded,
        "price_ratio": price_ratio,
        "popularity_weighted_signal": popularity_weighted_signal,
        "candidate_category_main": int(cat_lower == "main"),
        "candidate_category_side": int(cat_lower == "side"),
        "candidate_category_drink": int(cat_lower == "drink"),
        "candidate_category_dessert": int(cat_lower == "dessert"),
    }
    row.update(cart_agg)
    return row


def build_feature_matrix(training_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    df = training_df.copy()
    df["cart_items"] = df["cart_items"].apply(
        lambda v: v if isinstance(v, list) else ast.literal_eval(v)
    )

    item_dict = item_df.set_index("item_id").to_dict("index")

    feature_rows = []
    for _, row in df.iterrows():
        feature_rows.append(
            build_features(
                cart_items=row["cart_items"],
                candidate_item=row["candidate_item"],
                city=row["city"],
                meal_time=row["meal_time"],
                cuisine=row["cuisine"],
                item_dict=item_dict,
            )
        )

    feature_df = pd.DataFrame(feature_rows)
    feature_df["cart_id"] = df["cart_id"].values
    feature_df["label"] = df["label"].values
    # candidate_item is carried through as METADATA only (it is not in
    # FEATURE_COLUMNS, so the model never sees it). It's needed downstream to
    # compute the context-aware popularity baseline in train_ranker.py, which
    # ranks by per-item popularity within a (meal_time, candidate_category)
    # bucket and therefore needs to know which item each candidate row is.
    feature_df["candidate_item"] = df["candidate_item"].values

    return feature_df[["cart_id"] + FEATURE_COLUMNS + ["candidate_item", "label"]]


if __name__ == "__main__":
    training_df = pd.read_csv("training_data.csv")
    item_df = pd.read_csv("items.csv")

    feature_matrix = build_feature_matrix(training_df, item_df)
    feature_matrix.to_csv("feature_matrix.csv", index=False)

    print("Feature matrix saved. Shape:", feature_matrix.shape)
    print("Columns:", list(feature_matrix.columns))
