import numpy as np
import pandas as pd

from src.phase2.data_preparation import (
    _add_price_features,
    apply_category_maps,
    build_category_maps,
    create_forecast_windows,
    get_training_boundaries,
)


def small_data():
    dates = pd.date_range("2020-01-01", periods=35)
    rows = []
    for series_id, item_id in [("A", "ITEM_A"), ("B", "ITEM_B")]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "id": series_id,
                    "item_id": item_id,
                    "store_id": "STORE_1",
                    "cat_id": "FOODS",
                    "dept_id": "FOODS_1",
                    "state_id": "CA",
                    "date": date,
                    "sales": index % 3,
                    "sell_price": np.nan if index < 3 else 2.0,
                    "day_of_week": date.day_of_week,
                    "month": date.month,
                    "is_weekend": int(date.day_of_week >= 5),
                    "event": 0,
                    "snap": 0,
                }
            )
    return pd.DataFrame(rows)


def test_availability_and_missing_price_masks_are_consistent():
    prepared = _add_price_features(small_data())
    assert prepared.loc[prepared["sell_price"].isna(), "is_price_missing"].eq(1).all()
    assert prepared.loc[prepared["sell_price"].isna(), "observation_mask"].eq(0).all()
    assert prepared["observation_mask"].isin([0.0, 1.0]).all()
    assert prepared[["log_price", "relative_price", "price_change"]].notna().all().all()


def test_unknown_categories_map_to_zero():
    data = small_data()
    maps = build_category_maps(data[data["id"].eq("A")])
    encoded = apply_category_maps(data, maps)
    unseen = encoded[encoded["id"].eq("B")]
    assert unseen["item_id_index"].eq(0).all()
    assert unseen["store_id_index"].gt(0).all()


def test_windows_and_boundaries_do_not_overlap():
    data = small_data()
    windows = create_forecast_windows(data, horizon=7, number_of_windows=2)
    boundaries = get_training_boundaries(data, windows, validation_days=7)
    assert len(windows) == 2
    assert (windows["forecast_end"] - windows["forecast_start"]).eq(pd.Timedelta(days=6)).all()
    assert boundaries["training_end"] < boundaries["validation_start"]
    assert boundaries["validation_end"] < boundaries["test_start"]
