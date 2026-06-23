"""
City-sensitivity check (focused city-PAIR version).

WHY THIS SCRIPT EXISTS
----------------------
A reviewer pointed out that a 10-city "does city change the recommendation"
table is weak evidence: with 10 cities clustered near a decision boundary,
most picking the same top item just looks like close-call noise. The fairer
test is to take the city PAIR that *should* differ the most given the hidden
structure we actually set up, hold everything else (cart / cuisine /
meal_time) fixed, and see whether the recommendation genuinely moves.

This script:
  1. Reports the hidden parameters that drive city behaviour (so the choice
     of "most different pair" is grounded in the real values, not a guess):
       - CITY_CUISINE_PRIOR  -> shapes which carts each city generates
       - HIDDEN_CITY_WEIGHT  -> the actual label-time mechanism: it scales the
         (cuisine, candidate_cuisine) affinity in the logit.
  2. Picks the most-different pair under each, and runs the held-everything-
     else-fixed scoring probe for them, loading the real trained model.

It is a diagnostic, not part of training - run it on demand:
    python model_training/city_sensitivity_check.py
"""

from pathlib import Path
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "data_generation"))

from phase2_feature_engineering import build_features, FEATURE_COLUMNS, CITY_LIST
from generate_orders import CITY_CUISINE_PRIOR
from generate_training_data import HIDDEN_CITY_WEIGHT

MODEL_PATH = ROOT / "model_training" / "model.joblib"
ITEMS_PATH = ROOT / "data_generation" / "items.csv"

# All five cuisines, in a fixed order, so cuisine-prior vectors are comparable.
CUISINE_ORDER = ["Italian", "South Indian", "Chinese", "North Indian", "Mughlai"]


def cuisine_vector(city: str) -> np.ndarray:
    prior = CITY_CUISINE_PRIOR[city]
    return np.array([prior.get(c, 0.0) for c in CUISINE_ORDER])


def most_different_pair_by_cuisine_prior():
    best, best_d = None, -1.0
    for a, b in combinations(CITY_LIST, 2):
        d = float(np.abs(cuisine_vector(a) - cuisine_vector(b)).sum())  # L1 / TV*2
        if d > best_d:
            best, best_d = (a, b), d
    return best, best_d


def most_different_pair_by_hidden_weight():
    best, best_d = None, -1.0
    for a, b in combinations(CITY_LIST, 2):
        d = abs(HIDDEN_CITY_WEIGHT[a] - HIDDEN_CITY_WEIGHT[b])
        if d > best_d:
            best, best_d = (a, b), d
    return best, best_d


def score_candidates(model, items_dict, cart_items, candidates, city, meal_time, cuisine):
    rows = [
        build_features(cart_items, c, city, meal_time, cuisine, items_dict)
        for c in candidates
    ]
    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    scores = model.predict(X)
    return pd.DataFrame({
        "item_id": candidates,
        "name": [items_dict[c]["item_name"] for c in candidates],
        "category": [items_dict[c]["category"] for c in candidates],
        "score": scores,
    }).sort_values("score", ascending=False).reset_index(drop=True)


def run_pair_probe(model, items_df, items_dict, city_a, city_b, cuisine, meal_time, label):
    print(f"\n{'=' * 72}\nPROBE: {label}")
    print(f"  city A = {city_a}   city B = {city_b}")
    print(f"  held fixed: cuisine={cuisine}, meal_time={meal_time}")
    print('=' * 72)

    cuisine_items = items_df[items_df["cuisine"] == cuisine]
    # Fixed cart: a single Main of this cuisine (lowest item_id for determinism).
    main = cuisine_items[cuisine_items["category"] == "Main"].sort_values("item_id").iloc[0]
    cart_items = [int(main["item_id"])]
    candidates = (
        cuisine_items[~cuisine_items["item_id"].isin(cart_items)]
        .sort_values("popularity_score", ascending=False)
        .head(25)["item_id"].tolist()
    )
    print(f"  cart = [{main['item_name']}]  ({len(candidates)} candidates)\n")

    ranked_a = score_candidates(model, items_dict, cart_items, candidates, city_a, meal_time, cuisine)
    ranked_b = score_candidates(model, items_dict, cart_items, candidates, city_b, meal_time, cuisine)

    print(f"  Top-5 in {city_a}:")
    for r in ranked_a.head(5).itertuples():
        print(f"    {r.score:7.4f}  {r.name} ({r.category})")
    print(f"\n  Top-5 in {city_b}:")
    for r in ranked_b.head(5).itertuples():
        print(f"    {r.score:7.4f}  {r.name} ({r.category})")

    top_a = ranked_a.iloc[0]["item_id"]
    top_b = ranked_b.iloc[0]["item_id"]
    set_a = set(ranked_a.head(3)["item_id"])
    set_b = set(ranked_b.head(3)["item_id"])

    # Full-ordering agreement (Spearman over the shared candidate set).
    merged = ranked_a[["item_id", "score"]].merge(
        ranked_b[["item_id", "score"]], on="item_id", suffixes=("_a", "_b")
    )
    spearman = merged["score_a"].corr(merged["score_b"], method="spearman")
    mean_score_a = ranked_a["score"].mean()
    mean_score_b = ranked_b["score"].mean()

    print(f"\n  --- comparison ---")
    print(f"  top-1 same?           {top_a == top_b}")
    print(f"  top-3 overlap:        {len(set_a & set_b)}/3")
    print(f"  Spearman(rank A, B):  {spearman:.4f}   (1.0 = identical ordering)")
    print(f"  mean score A vs B:    {mean_score_a:.4f} vs {mean_score_b:.4f}"
          f"   (level shift = {mean_score_a - mean_score_b:+.4f})")
    return {
        "label": label, "city_a": city_a, "city_b": city_b,
        "top1_same": bool(top_a == top_b), "top3_overlap": len(set_a & set_b),
        "spearman": float(spearman),
        "mean_score_a": float(mean_score_a), "mean_score_b": float(mean_score_b),
    }


def main():
    model = joblib.load(MODEL_PATH)
    items_df = pd.read_csv(ITEMS_PATH)
    items_dict = items_df.set_index("item_id").to_dict("index")

    print("HIDDEN_CITY_WEIGHT (the label-time mechanism through which city acts):")
    for city in CITY_LIST:
        print(f"  {city:11s} {HIDDEN_CITY_WEIGHT[city]:.4f}")

    print("\nCITY_CUISINE_PRIOR vectors  [" + ", ".join(CUISINE_ORDER) + "]:")
    for city in CITY_LIST:
        vec = cuisine_vector(city)
        print(f"  {city:11s} {np.array2string(vec, precision=2, floatmode='fixed')}")

    (pa, pb), pd_dist = most_different_pair_by_cuisine_prior()
    (wa, wb), wd_dist = most_different_pair_by_hidden_weight()
    print(f"\nMost different by CUISINE PRIOR:  {pa} vs {pb}  (L1 distance {pd_dist:.2f})")
    print(f"Most different by HIDDEN_CITY_WEIGHT: {wa} vs {wb}  (gap {wd_dist:.4f})")

    cuisine, meal_time = "Mughlai", "Dinner"
    results = []
    results.append(run_pair_probe(model, items_df, items_dict, pa, pb, cuisine, meal_time,
                                  "most-different cuisine-prior pair"))
    if {wa, wb} != {pa, pb}:
        results.append(run_pair_probe(model, items_df, items_dict, wa, wb, cuisine, meal_time,
                                      "most-different hidden-city-weight pair"))

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for r in results:
        print(f"  {r['label']}: {r['city_a']} vs {r['city_b']} -> "
              f"top1_same={r['top1_same']}, top3_overlap={r['top3_overlap']}/3, "
              f"spearman={r['spearman']:.3f}, "
              f"mean_score {r['mean_score_a']:.4f} vs {r['mean_score_b']:.4f}")


if __name__ == "__main__":
    main()
