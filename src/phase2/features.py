"""Feature lists used by the Phase 2 experiments.

Keeping the lists here makes every ablation explicit. A model only receives the
columns named by its experiment.
"""

STATIC_COLUMNS = ["item_id", "store_id", "cat_id", "dept_id", "state_id"]

CALENDAR_FEATURES = [
    "day_of_week_scaled",
    "month_scaled",
    "is_weekend",
    "event",
    "snap",
]

PRICE_AVAILABILITY_FEATURES = [
    "log_price",
    "price_change",
    "relative_price",
    "is_available_proxy",
    "is_price_missing",
    "is_store_closed",
    "observation_mask",
]

KALMAN_FEATURES = [
    "kalman_level_scaled",
    "kalman_variance_scaled",
    "kalman_innovation_scaled",
]


def get_feature_set(experiment_name):
    """Return dynamic and static inputs for a named ablation."""

    feature_sets = {
        "target_only": {
            "dynamic": ["observation_mask"],
            "static": [],
        },
        "calendar": {
            "dynamic": CALENDAR_FEATURES + ["observation_mask"],
            "static": [],
        },
        "calendar_static": {
            "dynamic": CALENDAR_FEATURES + ["observation_mask"],
            "static": STATIC_COLUMNS,
        },
        "deepar": {
            "dynamic": CALENDAR_FEATURES + PRICE_AVAILABILITY_FEATURES,
            "static": STATIC_COLUMNS,
        },
        "kalman_only_addition": {
            "dynamic": CALENDAR_FEATURES + ["observation_mask"] + KALMAN_FEATURES,
            "static": STATIC_COLUMNS,
        },
        "hybrid": {
            "dynamic": CALENDAR_FEATURES
            + PRICE_AVAILABILITY_FEATURES
            + KALMAN_FEATURES,
            "static": STATIC_COLUMNS,
        },
    }

    if experiment_name not in feature_sets:
        available = ", ".join(sorted(feature_sets))
        raise ValueError(f"Unknown experiment '{experiment_name}'. Choose: {available}")

    selected = feature_sets[experiment_name]
    return {
        "dynamic": list(dict.fromkeys(selected["dynamic"])),
        "static": selected["static"].copy(),
    }
