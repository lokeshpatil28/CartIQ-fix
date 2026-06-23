"""
Trains the CSAO ranking model.

============================================================================
WHAT CHANGED VS. THE ORIGINAL SCRIPT, AND WHY
============================================================================
1. LGBMRanker instead of LGBMClassifier.
   The original trained a binary classifier (will-this-be-added: yes/no)
   and then sorted candidates by the raw classifier score - a pointwise
   classification objective being repurposed, after the fact, as a ranking
   signal. That works, but it's not what the objective was optimized for.
   LGBMRanker with `lambdarank` is optimized directly for getting the
   *order* of candidates within a cart right, which is the actual task
   (top-3 recommendations), not "is this one item good in isolation".

2. Group-aware train/test split instead of a row-level stratified split.
   The original used sklearn's train_test_split directly on individual
   (cart, candidate) rows. Since every cart contributes ~20-30 candidate
   rows, a row-level split lets some candidates from the *same* cart end
   up in train and others in test - the model can partially "see" a cart
   during training and then get evaluated on a held-out slice of that same
   cart, which inflates reported metrics relative to genuine
   generalization to unseen carts. This script splits on cart_id via
   GroupShuffleSplit, so an entire cart - all its candidate rows - is
   either fully in train or fully in test, never split across both.

3. Metrics are persisted to disk (metrics.json), not just printed.
   The original report admitted real metrics were "printed but not
   stored" - meaning there was no record of what was actually measured.
   Every run now writes a timestamped, reproducible metrics file.

4. Saved with joblib, consistently (the original saved with pickle but
   the API loaded with joblib - inconsistent, works by luck not by
   design).
============================================================================
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "data_generation"))
from phase2_feature_engineering import FEATURE_COLUMNS  # single source of truth

DATA_PATH = Path(__file__).resolve().parent.parent / "data_generation" / "feature_matrix.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
METRICS_PATH = Path(__file__).resolve().parent / "metrics.json"

RANDOM_STATE = 42
TOP_K = 3


def ndcg_at_k(group: pd.DataFrame, k: int, score_col: str) -> float:
    ranked = group.sort_values(score_col, ascending=False).head(k)
    dcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ranked["label"]))

    ideal = group.sort_values("label", ascending=False).head(k)
    idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal["label"]))

    return dcg / idcg if idcg > 0 else 0.0


def precision_recall_at_k(group: pd.DataFrame, k: int, score_col: str):
    ranked = group.sort_values(score_col, ascending=False).head(k)
    true_positives = group["label"].sum()
    top_k_positives = ranked["label"].sum()

    precision = top_k_positives / k
    recall = top_k_positives / true_positives if true_positives > 0 else 0.0
    return precision, recall


CONTEXT_SMOOTHING = 20.0  # empirical-Bayes shrinkage strength (pseudo-counts)


def fit_context_popularity(train_df: pd.DataFrame, smoothing: float = CONTEXT_SMOOTHING):
    """Fit a *context-aware-but-not-cart-aware* popularity baseline.

    For each (meal_time, candidate_item) it estimates how often that item is
    the next-added item, smoothed (empirical-Bayes) toward the global positive
    rate so items seen only a handful of times don't get extreme scores.

    This sits deliberately between the two existing baselines:
      - the plain popularity baseline (candidate_popularity_score) has ZERO
        context - it's a single static number per item, identical regardless
        of meal_time or anything in the cart;
      - this baseline adds meal_time context (an item's popularity is allowed
        to differ between Breakfast and Dinner) but still ignores the cart
        contents entirely;
      - the full ranker uses the whole cart + city + meal_time.

    Because candidate_item determines candidate_category, grouping by
    (meal_time, candidate_item) is exactly "most popular item within this
    (meal_time, candidate_category) bucket". Fit on TRAIN only, then scored on
    TEST, so it's a fair held-out comparison just like the model.
    """
    global_rate = float(train_df["label"].mean())
    grp = train_df.groupby(["meal_time", "candidate_item"])["label"].agg(["sum", "count"])
    grp["score"] = (grp["sum"] + smoothing * global_rate) / (grp["count"] + smoothing)
    return grp["score"].to_dict(), global_rate


def apply_context_popularity(df: pd.DataFrame, lookup: dict, fallback: float) -> pd.Series:
    """Score each row by its (meal_time, candidate_item) popularity estimate.

    (meal_time, item) pairs that never appeared in train fall back to the
    global positive rate - they carry no context-specific information, which
    is the honest thing to do rather than inventing a score for them.
    """
    keys = list(zip(df["meal_time"], df["candidate_item"]))
    return pd.Series([lookup.get(k, fallback) for k in keys], index=df.index)


def evaluate_ranking(test_df: pd.DataFrame, score_col: str, k: int = TOP_K) -> dict:
    precisions, recalls, ndcgs = [], [], []
    for _, group in test_df.groupby("cart_id"):
        p, r = precision_recall_at_k(group, k, score_col)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(ndcg_at_k(group, k, score_col))

    return {
        f"precision_at_{k}": float(np.mean(precisions)),
        f"recall_at_{k}": float(np.mean(recalls)),
        f"ndcg_at_{k}": float(np.mean(ndcgs)),
    }


def main():
    print("Loading feature matrix:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)

    # ----------------------------------------------------------------
    # Group-aware split: split on cart_id, not on rows. This is the fix
    # for the leakage issue described above.
    # ----------------------------------------------------------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df["cart_id"]))

    train_df = df.iloc[train_idx].sort_values("cart_id").reset_index(drop=True)
    test_df = df.iloc[test_idx].sort_values("cart_id").reset_index(drop=True)

    # Sanity check: confirm zero cart_id overlap between train and test.
    overlap = set(train_df["cart_id"]) & set(test_df["cart_id"])
    assert not overlap, f"Leakage detected: {len(overlap)} cart_ids appear in both splits"
    print(f"Train carts: {train_df['cart_id'].nunique()} | Test carts: {test_df['cart_id'].nunique()}")
    print("Leakage check passed: zero overlapping cart_ids between train/test.")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    # LGBMRanker needs group sizes (number of candidates per cart), in the
    # same row order as X/y - hence the explicit sort_values("cart_id")
    # above on both splits, and group counts computed the same way here.
    train_group_sizes = train_df.groupby("cart_id").size().values
    test_group_sizes = test_df.groupby("cart_id").size().values

    print("\nTraining LGBMRanker (lambdarank)...")
    t0 = time.time()
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[TOP_K],
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=300,
        max_depth=-1,
        random_state=RANDOM_STATE,
    )
    ranker.fit(
        X_train, y_train,
        group=train_group_sizes,
        eval_set=[(X_test, y_test)],
        eval_group=[test_group_sizes],
        eval_at=[TOP_K],
    )
    train_time_seconds = round(time.time() - t0, 2)
    print(f"Training completed in {train_time_seconds}s")

    # ----------------------------------------------------------------
    # Evaluate: ranker score, vs. popularity baseline, vs. random baseline.
    # ----------------------------------------------------------------
    test_df = test_df.copy()
    test_df["ranker_score"] = ranker.predict(X_test)

    rng = np.random.default_rng(RANDOM_STATE)
    test_df["random_score"] = rng.random(len(test_df))

    # Context-aware baseline: fit on TRAIN, score TEST (no leakage).
    context_lookup, global_rate = fit_context_popularity(train_df)
    test_df["context_popularity_score"] = apply_context_popularity(
        test_df, context_lookup, fallback=global_rate
    )

    ranker_metrics = evaluate_ranking(test_df, "ranker_score")
    context_metrics = evaluate_ranking(test_df, "context_popularity_score")
    popularity_metrics = evaluate_ranking(test_df, "candidate_popularity_score")
    random_metrics = evaluate_ranking(test_df, "random_score")

    print("\n--- Ranker (full cart + city + meal_time) ---")
    print(ranker_metrics)
    print("--- Context-aware baseline (meal_time + item, no cart) ---")
    print(context_metrics)
    print("--- Popularity baseline (item only, zero context) ---")
    print(popularity_metrics)
    print("--- Random baseline ---")
    print(random_metrics)

    # ----------------------------------------------------------------
    # Confidence-tier calibration. LGBMRanker scores are NOT probabilities
    # and have no fixed 0-1 range (here they're ~75% negative, median
    # -0.28). A hardcoded "0.6 = high confidence" cutoff is meaningless
    # against that distribution. Instead we derive Low/Medium/High tiers
    # from the TERCILES of the held-out per-cart TOP-1 scores - i.e. "a
    # recommendation as strong as a typical #1 pick is High". These get
    # persisted and the API attaches the tier to each recommendation, so
    # the threshold lives in exactly one place (no frontend drift).
    top1_scores = (
        test_df.sort_values("ranker_score", ascending=False)
        .groupby("cart_id")
        .head(1)["ranker_score"]
    )
    score_tiers = {
        "low_max": float(round(top1_scores.quantile(1 / 3), 4)),
        "high_min": float(round(top1_scores.quantile(2 / 3), 4)),
        "basis": "terciles of held-out per-cart top-1 ranker scores",
        "n_carts": int(top1_scores.size),
    }
    print(f"\nConfidence tiers (from top-1 terciles): "
          f"Low < {score_tiers['low_max']} <= Medium < {score_tiers['high_min']} <= High")

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": ranker.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importances:")
    print(importance_df.to_string(index=False))

    # ----------------------------------------------------------------
    # Persist model + metrics (fixes "printed but not stored").
    # ----------------------------------------------------------------
    joblib.dump(ranker, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    def pct_gap(a: dict, b: dict) -> dict:
        """How much a exceeds b, as a percentage of b, per metric."""
        return {
            k: round(100 * (a[k] - b[k]) / b[k], 2) if b[k] > 0 else None
            for k in a
        }

    metrics_record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_path": str(DATA_PATH),
        "n_rows": len(df),
        "n_train_carts": int(train_df["cart_id"].nunique()),
        "n_test_carts": int(test_df["cart_id"].nunique()),
        "train_time_seconds": train_time_seconds,
        # Four-point comparison, weakest -> strongest context:
        #   random  ->  popularity (item only)  ->  context-aware (item +
        #   meal_time, no cart)  ->  ranker (full cart + city + meal_time).
        "ranker": ranker_metrics,
        "context_aware_baseline": context_metrics,
        "popularity_baseline": popularity_metrics,
        "random_baseline": random_metrics,
        # How much signal each baseline itself carries over pure chance. This
        # is reported alongside the lift below so no lift number stands alone:
        # a big "lift over popularity" means little if popularity barely beats
        # random in the first place, so both must be read together.
        "baseline_strength_vs_random_pct": {
            "popularity_over_random": pct_gap(popularity_metrics, random_metrics),
            "context_aware_over_random": pct_gap(context_metrics, random_metrics),
        },
        "relative_lift_pct": {
            "ranker_over_popularity": pct_gap(ranker_metrics, popularity_metrics),
            "ranker_over_context_aware": pct_gap(ranker_metrics, context_metrics),
        },
        "score_tiers": score_tiers,
        "feature_importances": importance_df.to_dict(orient="records"),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_record, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
