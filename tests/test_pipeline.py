"""
Automated tests for the CSAO pipeline. Run with: pytest tests/

These replace the original single broken smoke test
(backend/test_api.py, which just fired one manual POST request with no
assertions and required the server to already be running).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "data_generation"))

from generate_items import build_items, CUISINES, CATEGORIES
from generate_orders import build_orders
from build_snapshots import build_snapshots
from generate_training_data import generate_training_data
from phase2_feature_engineering import build_features, build_feature_matrix, FEATURE_COLUMNS, CITY_LIST, MEAL_MAP


@pytest.fixture(scope="module")
def item_df():
    return build_items()


@pytest.fixture(scope="module")
def orders_df(item_df):
    return build_orders(item_df)


@pytest.fixture(scope="module")
def snapshots_df(orders_df):
    return build_snapshots(orders_df)


class TestItemGeneration:
    def test_every_cuisine_category_combo_has_stock(self, item_df):
        for cuisine in CUISINES:
            for category in CATEGORIES:
                count = len(item_df[(item_df.cuisine == cuisine) & (item_df.category == category)])
                assert count > 0, f"No items for {cuisine}/{category} - candidate generation would fail"

    def test_prices_positive(self, item_df):
        assert (item_df["price"] > 0).all()

    def test_margin_in_valid_range(self, item_df):
        assert item_df["margin"].between(0, 1).all()


class TestOrderGeneration:
    def test_every_order_has_a_main(self, item_df, orders_df):
        main_ids = set(item_df[item_df.category == "Main"]["item_id"])
        for items in orders_df["items"]:
            assert any(i in main_ids for i in items), "Order missing a Main item"

    def test_cart_items_belong_to_order_cuisine(self, item_df, orders_df):
        item_cuisine = dict(zip(item_df.item_id, item_df.cuisine))
        for _, row in orders_df.head(200).iterrows():
            for i in row["items"]:
                assert item_cuisine[i] == row["cuisine"]


class TestSnapshots:
    def test_cart_id_is_unique_per_snapshot(self, snapshots_df):
        assert snapshots_df["cart_id"].is_unique

    def test_next_item_not_already_in_cart(self, snapshots_df):
        for _, row in snapshots_df.head(500).iterrows():
            assert row["next_item"] not in row["cart_items"]


class TestLabelGeneration:
    def test_label_is_binary(self, snapshots_df, item_df):
        training_df = generate_training_data(snapshots_df.head(500), item_df)
        assert set(training_df["label"].unique()) <= {0, 1}

    def test_no_candidate_equals_existing_cart_item(self, snapshots_df, item_df):
        training_df = generate_training_data(snapshots_df.head(500), item_df)
        for _, row in training_df.head(500).iterrows():
            assert row["candidate_item"] not in row["cart_items"]

    def test_positive_rate_is_plausible(self, snapshots_df, item_df):
        # Sanity bound, not a tight check: catches gross miscalibration
        # (e.g. an accidental sign flip making almost everything positive
        # or almost everything negative) without overfitting the test to
        # one exact number.
        training_df = generate_training_data(snapshots_df, item_df)
        rate = training_df["label"].mean()
        assert 0.03 < rate < 0.35, f"Positive rate {rate:.3f} looks miscalibrated"

    def test_labels_are_not_deterministic_given_features_alone(self, snapshots_df, item_df):
        """
        This is the regression test for the original circularity bug.
        It generates labels twice for the exact same (cart, candidate,
        context) inputs and confirms they are NOT always identical -
        proving the label depends on more than what a feature-engineer
        with access to cart/candidate/context alone could read off
        deterministically (i.e. the irreducible noise term is real).
        If this test ever fails, the label generator has regressed to
        being a deterministic function of visible inputs again.
        """
        run_1 = generate_training_data(snapshots_df.head(300), item_df)
        run_2 = generate_training_data(snapshots_df.head(300), item_df)
        # Same rows generated independently (latent draws differ each call
        # since the module-level rng advances) should disagree on at least
        # some labels for otherwise-identical (cart, candidate) pairs.
        merged = run_1.merge(
            run_2, on=["cart_id", "candidate_item"], suffixes=("_1", "_2")
        )
        disagreement_rate = (merged["label_1"] != merged["label_2"]).mean()
        assert disagreement_rate > 0.0, (
            "Labels are perfectly deterministic across runs - the noise "
            "term may have been removed, reintroducing circularity risk."
        )


class TestFeatureEngineering:
    def test_feature_matrix_has_no_constant_columns(self, snapshots_df, item_df):
        training_df = generate_training_data(snapshots_df.head(800), item_df)
        feature_matrix = build_feature_matrix(training_df, item_df)
        for col in FEATURE_COLUMNS:
            assert feature_matrix[col].nunique() > 1, f"{col} is constant - check generator/feature logic"

    def test_feature_matrix_has_no_nulls(self, snapshots_df, item_df):
        training_df = generate_training_data(snapshots_df.head(800), item_df)
        feature_matrix = build_feature_matrix(training_df, item_df)
        assert feature_matrix.isna().sum().sum() == 0

    def test_build_features_matches_column_order(self, item_df):
        item_dict = item_df.set_index("item_id").to_dict("index")
        cuisine = CUISINES[0]
        cart_items = item_df[item_df.cuisine == cuisine].head(1)["item_id"].tolist()
        candidate = item_df[item_df.cuisine == cuisine].iloc[1]["item_id"]

        row = build_features(cart_items, candidate, CITY_LIST[0], list(MEAL_MAP)[0], cuisine, item_dict)
        assert set(row.keys()) == set(FEATURE_COLUMNS)


class TestNoTrainServeSkew:
    def test_training_and_api_use_the_same_feature_function(self):
        """
        Both model_training (via build_feature_matrix) and backend/api.py
        import build_features from the same module rather than each
        defining their own copy. We can't always import backend.api
        directly here (it loads a real model file at import time, which
        may not exist in a fresh checkout before training has run), so
        instead this asserts on the source text: api.py must import
        build_features from phase2_feature_engineering rather than
        defining a function of that name itself. This is what actually
        guards against someone "temporarily" re-implementing the function
        in api.py in a future edit.
        """
        api_source = (Path(__file__).resolve().parent.parent / "backend" / "api.py").read_text()
        assert "from phase2_feature_engineering import build_features" in api_source
        assert "def build_features" not in api_source, (
            "api.py defines its own build_features - this reintroduces "
            "the train/serve skew risk the shared module was meant to fix."
        )
