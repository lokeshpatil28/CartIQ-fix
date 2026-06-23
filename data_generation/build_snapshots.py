"""
Slices each finished order into sequential (partial_cart -> next_item)
snapshots. E.g. an order [Main, Side, Drink] produces two snapshots:
  ([Main] -> Side), ([Main, Side] -> Drink)

This part of the original pipeline was already structurally sound and is
kept with only minor cleanup. cart_id uniquely identifies one snapshot
(one partial-cart context) and is the grouping key used everywhere
downstream for leakage-safe splitting and ranking evaluation.
"""

import pandas as pd
import ast


def build_snapshots(orders_df: pd.DataFrame) -> pd.DataFrame:
    orders_df = orders_df.copy()
    orders_df["items"] = orders_df["items"].apply(
        lambda v: v if isinstance(v, list) else ast.literal_eval(v)
    )

    snapshots = []
    cart_id_counter = 0

    for _, row in orders_df.iterrows():
        items = row["items"]
        for step in range(1, len(items)):
            snapshots.append({
                "cart_id": cart_id_counter,
                "order_id": row["order_id"],
                "city": row["city"],
                "meal_time": row["meal_time"],
                "cuisine": row["cuisine"],
                "cart_items": items[:step],
                "next_item": items[step],
            })
            cart_id_counter += 1

    return pd.DataFrame(snapshots)


if __name__ == "__main__":
    orders_df = pd.read_csv("orders.csv")
    snapshots_df = build_snapshots(orders_df)
    snapshots_df.to_csv("snapshots.csv", index=False)
    print("Snapshots generated:", len(snapshots_df))
    print("Unique cart_ids:", snapshots_df["cart_id"].nunique())
