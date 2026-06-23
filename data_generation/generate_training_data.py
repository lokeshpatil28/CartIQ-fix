"""
Generates training labels (training_data.csv): for every (cart, candidate
item) pair, a binary label indicating whether the candidate would plausibly
be added next.

============================================================================
WHY THIS FILE WAS REWRITTEN (read this before touching the logic below)
============================================================================
The original version of this generator computed a label probability using
an explicit, hand-written rule like:

    prob = 0.02 + 0.32*(dinner & dessert) + 0.30*(city_cuisine_match) + ...

...and the feature-engineering step then built features such as
`is_dinner_dessert` and `city_cuisine_match` using the *exact same
predicates*. That makes the whole exercise circular: the model is handed
features that are direct lookups into the formula that produced its own
label, so a model "succeeding" here proves only that gradient boosting can
memorize a rule its own designer wrote into both sides of the experiment.
It does not demonstrate that the model learned anything about behavior.

This version breaks that circularity with three changes:

1. HIDDEN GENERATIVE STRUCTURE. The true affinity between a cart's context
   and a candidate item is driven by a hidden affinity matrix (drawn
   randomly at generation time, not chosen by hand to match feature names)
   and a per-cart latent "impulsivity" trait that is *never* exposed as a
   feature. Downstream feature engineering only sees cart contents and
   candidate metadata - it has no direct read access to the matrix or the
   latent trait.

2. NONLINEAR COMBINATION. The hidden signal enters through a nonlinear
   interaction (latent_trait squared, a sinusoidal term on price ratio) so
   that no linear combination of visible features can perfectly
   reconstruct the label, the way a simple weighted sum of binary flags
   could in the original version.

3. IRREDUCIBLE NOISE. Gaussian noise is added to the logit before the
   Bernoulli draw. This puts a ceiling on achievable AUC/NDCG (you should
   see something meaningfully below 0.99, not because the model is bad,
   but because the data-generating process itself is only partially
   predictable - exactly like real consumer behavior data. A model that
   approaches this ceiling without exceeding it is doing exactly what it
   should.

This is still synthetic data (no real users are involved), and that
limitation should be stated plainly in any write-up of this project. What
this rewrite buys you is that the *methodology* - feature engineering,
model selection, evaluation - is now exercised honestly: the model has to
discover structure it wasn't told about, the same way it would have to on
real order data.
============================================================================
"""

import pandas as pd
import numpy as np
import ast

from generate_items import CUISINES

SEED = 7  # deliberately different from the data-generation seed (42) so
          # label noise isn't accidentally correlated with cart composition
rng = np.random.default_rng(SEED)

CATEGORIES = ["Main", "Side", "Drink", "Dessert"]
MEAL_TIMES = ["Breakfast", "Lunch", "Snacks", "Dinner"]

# ----------------------------------------------------------------------
# HIDDEN GENERATIVE PARAMETERS
# These are intentionally NOT derived from, or matched to, the feature
# names used in phase2_feature_engineering.py. They are redrawn from a
# fixed seed once, then frozen - the feature engineer (a future you,
# reading this repo) should treat them as unknown when designing features,
# exactly as a real behavioral signal would be unknown.
# ----------------------------------------------------------------------

# Hidden affinity: how well does a (meal_time, candidate_category) pairing
# tend to go together, independent of anything else? Random, not hand-set.
_hidden_meal_category_affinity = rng.normal(loc=0.0, scale=0.6, size=(len(MEAL_TIMES), len(CATEGORIES)))
HIDDEN_MEAL_CATEGORY_AFFINITY = pd.DataFrame(
    _hidden_meal_category_affinity, index=MEAL_TIMES, columns=CATEGORIES
)

# Hidden affinity: how well does a (cart cuisine, candidate cuisine) pairing
# go together? Diagonal gets a same-cuisine boost baked in at random
# strength per cuisine (not a single global constant like the original).
_hidden_cuisine_affinity = rng.normal(loc=0.0, scale=0.4, size=(len(CUISINES), len(CUISINES)))
for i in range(len(CUISINES)):
    _hidden_cuisine_affinity[i, i] += rng.uniform(0.3, 0.9)  # same-cuisine bump, randomized strength
HIDDEN_CUISINE_AFFINITY = pd.DataFrame(
    _hidden_cuisine_affinity, index=CUISINES, columns=CUISINES
)

# Hidden per-city "adventurousness" - cities differ in how much a city-wide
# cuisine preference matters vs. pure in-cart context. Random, not hand-set.
HIDDEN_CITY_WEIGHT = {
    "Bangalore": rng.uniform(0.2, 0.6),
    "Delhi": rng.uniform(0.2, 0.6),
    "Mumbai": rng.uniform(0.2, 0.6),
    "Hyderabad": rng.uniform(0.2, 0.6),
    # ── 6 new cities for 10-city expansion ─────────────────────────────
    # Each gets its own random draw, producing genuinely distinct hidden
    # weights — not copy-pasted from the originals. The rng state advances
    # sequentially, so each city gets a different value deterministically.
    "Chennai": rng.uniform(0.2, 0.6),
    "Kolkata": rng.uniform(0.2, 0.6),
    "Pune": rng.uniform(0.2, 0.6),
    "Ahmedabad": rng.uniform(0.2, 0.6),
    "Jaipur": rng.uniform(0.2, 0.6),
    "Lucknow": rng.uniform(0.2, 0.6),
}

NOISE_SCALE = 0.65  # irreducible logit noise - this is what keeps AUC < 1.0
BASE_LOGIT = -2.8   # controls overall positive rate (~5-12% before signal)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_training_data(snapshots_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    snapshots_df = snapshots_df.copy()
    snapshots_df["cart_items"] = snapshots_df["cart_items"].apply(
        lambda v: v if isinstance(v, list) else ast.literal_eval(v)
    )

    cuisine_item_map = item_df.groupby("cuisine")["item_id"].apply(list).to_dict()
    item_category_map = dict(zip(item_df["item_id"], item_df["category"]))
    item_cuisine_map = dict(zip(item_df["item_id"], item_df["cuisine"]))
    item_price_map = dict(zip(item_df["item_id"], item_df["price"]))

    rows = []

    for _, row in snapshots_df.iterrows():
        cart_id = row["cart_id"]
        cart = row["cart_items"]
        cuisine = row["cuisine"]
        city = row["city"]
        meal_time = row["meal_time"]

        # Per-cart latent trait: how impulsive/exploratory this particular
        # session is. NEVER written to the output - it only acts through
        # the label. A feature engineer cannot see this column.
        latent_impulsivity = rng.beta(2, 5)

        cart_total_price = sum(item_price_map.get(i, 0) for i in cart)

        # Candidate pool: everything not already in the cart, restricted to
        # the same cuisine as the order (mirrors how a real cuisine-scoped
        # delivery menu works - you don't get cross-cuisine add-ons).
        cuisine_items = cuisine_item_map.get(cuisine, [])
        candidates = list(set(cuisine_items) - set(cart))

        for candidate in candidates:
            candidate_category = item_category_map[candidate]
            candidate_cuisine = item_cuisine_map[candidate]
            candidate_price = item_price_map[candidate]

            meal_cat_signal = HIDDEN_MEAL_CATEGORY_AFFINITY.loc[meal_time, candidate_category]
            cuisine_signal = HIDDEN_CUISINE_AFFINITY.loc[cuisine, candidate_cuisine]
            city_weight = HIDDEN_CITY_WEIGHT.get(city, 0.4)

            price_ratio = candidate_price / (cart_total_price + 1)

            # ---- nonlinear combination (no linear feature can fully invert this) ----
            logit = (
                BASE_LOGIT
                + meal_cat_signal * (1 + latent_impulsivity ** 2)       # squared latent interaction
                + cuisine_signal * (0.5 + city_weight)                  # city modulates cuisine affinity
                + 0.5 * np.sin(price_ratio * np.pi)                     # nonlinear in price ratio
                + rng.normal(0, NOISE_SCALE)                            # irreducible noise
            )

            label = rng.binomial(1, _sigmoid(logit))

            rows.append({
                "cart_id": cart_id,
                "cart_items": cart,
                "candidate_item": candidate,
                "label": int(label),
                "city": city,
                "meal_time": meal_time,
                "cuisine": cuisine,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    snapshots_df = pd.read_csv("snapshots.csv")
    item_df = pd.read_csv("items.csv")

    training_df = generate_training_data(snapshots_df, item_df)
    training_df.to_csv("training_data.csv", index=False)

    print("Training rows generated:", len(training_df))
    print("Unique cart_ids:", training_df["cart_id"].nunique())
    print("Overall label mean:", round(training_df["label"].mean(), 4))
    print("\nLabel mean by meal_time:")
    print(training_df.groupby("meal_time")["label"].mean().round(4))
