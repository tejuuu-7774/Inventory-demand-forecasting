"""Prepare M5 data for the Phase 2 neural forecasting experiments."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .features import STATIC_COLUMNS


def find_project_root(start=None):
    """Find the repository root from a notebook or a Python process."""

    current = Path(start or Path.cwd()).resolve()
    while current != current.parent:
        if (current / "Datasets").exists() and (current / "Notebooks").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find the project root.")


def load_phase1_prepared(project_root):
    """Load the untouched Phase 1 long-format table."""

    path = Path(project_root) / "Datasets" / "processed" / "phase1_prepared.pkl"
    data = pd.read_pickle(path)
    data["date"] = pd.to_datetime(data["date"])
    return data


def load_selected_prices(project_root, selected_pairs, chunk_size=500_000):
    """Read only prices belonging to the 100 selected item-store pairs."""

    price_path = Path(project_root) / "Datasets" / "raw" / "sell_prices.csv"
    selected_pairs = selected_pairs[["store_id", "item_id"]].drop_duplicates()
    selected_key = set(map(tuple, selected_pairs.to_numpy()))
    kept_chunks = []

    for chunk in pd.read_csv(price_path, chunksize=chunk_size):
        keys = list(zip(chunk["store_id"], chunk["item_id"]))
        keep = np.fromiter((key in selected_key for key in keys), dtype=bool)
        if keep.any():
            kept_chunks.append(chunk.loc[keep].copy())

    if not kept_chunks:
        return pd.DataFrame(columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])

    prices = pd.concat(kept_chunks, ignore_index=True)
    prices["sell_price"] = prices["sell_price"].astype("float32")
    return prices


def _add_price_features(data):
    """Create causal price and availability features."""

    data = data.sort_values(["id", "date"]).copy()
    groups = data.groupby("id", sort=False)

    data["is_price_missing"] = data["sell_price"].isna().astype("int8")
    data["is_zero_sales"] = data["sales"].eq(0).astype("int8")
    data["is_store_closed"] = (
        data["date"].dt.month.eq(12) & data["date"].dt.day.eq(25)
    ).astype("int8")

    data["is_available_proxy"] = (
        data["sell_price"].notna() & data["is_store_closed"].eq(0)
    ).astype("int8")
    data["observation_mask"] = data["is_available_proxy"].astype("float32")

    has_evidence = data["sell_price"].notna() | data["sales"].gt(0)
    introduction_dates = data.loc[has_evidence].groupby("id")["date"].min()
    data["introduction_date"] = data["id"].map(introduction_dates)
    data["is_leading_zero"] = (
        data["date"].lt(data["introduction_date"]) & data["sales"].eq(0)
    ).astype("int8")

    data["price_change"] = groups["sell_price"].pct_change(fill_method=None)
    data["price_change"] = data["price_change"].replace([np.inf, -np.inf], np.nan)
    data["price_change"] = data["price_change"].clip(-1.0, 2.0).fillna(0.0)

    past_price_mean = groups["sell_price"].transform(
        lambda values: values.shift(1).expanding(min_periods=1).mean()
    )
    data["relative_price"] = data["sell_price"] / past_price_mean
    data["relative_price"] = data["relative_price"].clip(0.0, 3.0).fillna(0.0)
    data["log_price"] = np.log1p(data["sell_price"].fillna(0.0))

    data["day_of_week_scaled"] = data["day_of_week"].astype(float) / 6.0
    data["month_scaled"] = (data["month"].astype(float) - 1.0) / 11.0

    float_columns = [
        "price_change",
        "relative_price",
        "log_price",
        "day_of_week_scaled",
        "month_scaled",
        "observation_mask",
    ]
    data[float_columns] = data[float_columns].astype("float32")
    return data


def prepare_phase2_data(project_root, save=True):
    """Join Phase 1 data to M5 weekly prices and add safe Phase 2 features."""

    project_root = Path(project_root)
    data = load_phase1_prepared(project_root)

    calendar = pd.read_csv(
        project_root / "Datasets" / "raw" / "calendar.csv",
        usecols=["d", "wm_yr_wk"],
    )
    data = data.merge(calendar, on="d", how="left", validate="many_to_one")

    selected_pairs = data[["store_id", "item_id"]].drop_duplicates()
    prices = load_selected_prices(project_root, selected_pairs)
    data = data.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    data = _add_price_features(data)

    assert not data[["id", "date"]].duplicated().any()
    assert data["date"].notna().all()
    assert data["observation_mask"].isin([0.0, 1.0]).all()
    assert data["is_zero_sales"].eq(data["sales"].eq(0)).all()

    if save:
        output = project_root / "Datasets" / "processed" / "phase2_prepared.pkl"
        data.to_pickle(output)

    return data


def create_forecast_windows(data, horizon=7, number_of_windows=4):
    """Recreate the exact four Phase 1 rolling-origin windows."""

    dates = pd.DatetimeIndex(sorted(pd.to_datetime(data["date"]).unique()))
    test_dates = dates[-horizon * number_of_windows :]
    rows = []

    for window_index in range(number_of_windows):
        start = window_index * horizon
        window_dates = test_dates[start : start + horizon]
        rows.append(
            {
                "window": window_index + 1,
                "origin": window_dates[0] - pd.Timedelta(days=1),
                "forecast_start": window_dates[0],
                "forecast_end": window_dates[-1],
            }
        )

    return pd.DataFrame(rows)


def get_training_boundaries(data, windows, validation_days=14):
    """Create training, validation and final test boundaries."""

    test_start = pd.Timestamp(windows["forecast_start"].min())
    validation_end = test_start - pd.Timedelta(days=1)
    validation_start = validation_end - pd.Timedelta(days=validation_days - 1)
    training_end = validation_start - pd.Timedelta(days=1)

    return {
        "training_end": training_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "test_start": test_start,
        "test_end": pd.Timestamp(windows["forecast_end"].max()),
    }


def build_category_maps(data, training_series_ids=None):
    """Map known static categories to integers; zero is always unknown."""

    mapping_data = data
    if training_series_ids is not None:
        mapping_data = data[data["id"].isin(training_series_ids)]

    category_maps = {}
    for column in STATIC_COLUMNS:
        values = sorted(mapping_data[column].dropna().astype(str).unique())
        category_maps[column] = {value: index + 1 for index, value in enumerate(values)}
    return category_maps


def apply_category_maps(data, category_maps):
    """Add integer category columns and safely map unseen values to zero."""

    encoded = data.copy()
    for column, value_map in category_maps.items():
        encoded[f"{column}_index"] = (
            encoded[column].astype(str).map(value_map).fillna(0).astype("int64")
        )
    return encoded


def save_data_metadata(project_root, data, windows, boundaries, category_maps):
    """Save small, human-readable metadata without saving raw data again."""

    project_root = Path(project_root)
    output_path = project_root / "results" / "phase2" / "data_metadata.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "number_of_rows": int(len(data)),
        "number_of_series": int(data["id"].nunique()),
        "first_date": str(data["date"].min().date()),
        "last_date": str(data["date"].max().date()),
        "price_missing_fraction": float(data["is_price_missing"].mean()),
        "zero_sales_fraction": float(data["is_zero_sales"].mean()),
        "windows": [
            {key: str(value) for key, value in row.items()}
            for row in windows.to_dict(orient="records")
        ],
        "boundaries": {key: str(value.date()) for key, value in boundaries.items()},
        "category_cardinalities": {
            column: len(value_map) + 1 for column, value_map in category_maps.items()
        },
        "availability_note": (
            "is_available_proxy means a selling price exists and the date is not "
            "Christmas. It is not verified inventory-on-hand or a verified stockout."
        ),
    }
    output_path.write_text(json.dumps(metadata, indent=2))
    return output_path
