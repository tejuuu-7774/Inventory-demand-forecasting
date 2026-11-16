"""Leak-free cohort construction for cold-start experiments."""

import numpy as np


def make_cold_start_split(data, item_fraction=0.2, random_seed=42):
    """Select unseen items and unseen combinations without target overlap."""

    series = data[["id", "item_id", "store_id"]].drop_duplicates()
    item_counts = series.groupby("item_id")["id"].nunique()
    repeated_items = set(item_counts[item_counts >= 2].index)

    combination_holdout_ids = []
    for item_id in sorted(repeated_items):
        choices = sorted(series.loc[series["item_id"].eq(item_id), "id"])
        combination_holdout_ids.append(choices[-1])

    eligible_items = sorted(set(series["item_id"]) - repeated_items)
    number_to_hold_out = max(1, int(round(item_fraction * series["item_id"].nunique())))
    number_to_hold_out = min(number_to_hold_out, len(eligible_items))
    generator = np.random.default_rng(random_seed)
    unseen_items = sorted(generator.choice(eligible_items, number_to_hold_out, replace=False))

    unseen_item_ids = sorted(series.loc[series["item_id"].isin(unseen_items), "id"])
    excluded = set(unseen_item_ids) | set(combination_holdout_ids)
    training_series_ids = sorted(set(series["id"]) - excluded)

    return {
        "training_series_ids": training_series_ids,
        "unseen_items": unseen_items,
        "unseen_item_series_ids": unseen_item_ids,
        "combination_holdout_series_ids": sorted(combination_holdout_ids),
    }


def assert_cold_start_separation(data, split):
    """Confirm that held-out item targets never enter cold-start training."""

    training = data[data["id"].isin(split["training_series_ids"])]
    held_out = data[data["id"].isin(split["unseen_item_series_ids"])]
    training_items = set(training["item_id"].unique())
    held_out_items = set(held_out["item_id"].unique())
    if training_items & held_out_items:
        raise AssertionError("An unseen item is present in cold-start training data.")

    combination_rows = data[data["id"].isin(split["combination_holdout_series_ids"])]
    for item_id in combination_rows["item_id"].unique():
        if item_id not in training_items:
            raise AssertionError("A combination-holdout item is not represented in training.")
    return True
