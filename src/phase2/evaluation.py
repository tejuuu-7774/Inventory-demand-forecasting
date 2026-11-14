"""Point and probabilistic metrics for Phase 2 forecasts."""

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = [
    "experiment_id",
    "Model",
    "id",
    "item_id",
    "store_id",
    "cat_id",
    "forecast_origin",
    "forecast_date",
    "horizon_step",
    "actual",
    "q10",
    "q50",
    "q90",
]


def calculate_point_metrics(actual, predicted):
    """Calculate MAE, RMSE and WAPE using the Phase 1 definitions."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    total_actual = np.abs(actual).sum()
    wape = np.nan if total_actual == 0 else 100.0 * np.abs(error).sum() / total_actual
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "WAPE_percent": float(wape),
    }


def pinball_loss(actual, prediction, quantile):
    """Calculate quantile loss; lower is better."""

    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = actual - prediction
    return float(np.maximum(quantile * error, (quantile - 1.0) * error).mean())


def evaluate_probabilistic_forecast(predictions):
    """Calculate all required Phase 2 metrics for one prediction table."""

    required = {"actual", "q10", "q50", "q90"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    if not (predictions["q10"] <= predictions["q50"]).all():
        raise ValueError("q10 must be less than or equal to q50.")
    if not (predictions["q50"] <= predictions["q90"]).all():
        raise ValueError("q50 must be less than or equal to q90.")

    metrics = calculate_point_metrics(predictions["actual"], predictions["q50"])
    metrics["pinball_q10"] = pinball_loss(predictions["actual"], predictions["q10"], 0.1)
    metrics["pinball_q50"] = pinball_loss(predictions["actual"], predictions["q50"], 0.5)
    metrics["pinball_q90"] = pinball_loss(predictions["actual"], predictions["q90"], 0.9)
    metrics["mean_pinball_loss"] = float(
        np.mean([metrics["pinball_q10"], metrics["pinball_q50"], metrics["pinball_q90"]])
    )
    covered = predictions["actual"].between(predictions["q10"], predictions["q90"])
    metrics["interval_coverage_80_percent"] = float(100.0 * covered.mean())
    metrics["empirical_q10_below_percent"] = float(
        100.0 * predictions["actual"].lt(predictions["q10"]).mean()
    )
    metrics["empirical_q90_at_or_below_percent"] = float(
        100.0 * predictions["actual"].le(predictions["q90"]).mean()
    )
    metrics["empirical_q90_above_percent"] = float(
        100.0 * predictions["actual"].gt(predictions["q90"]).mean()
    )
    metrics["mean_interval_width"] = float((predictions["q90"] - predictions["q10"]).mean())
    metrics["forecast_points"] = int(len(predictions))
    return metrics


def add_demand_segments(predictions, training_data, training_end):
    """Attach intermittency and volume labels calculated from training only."""

    history = training_data[training_data["date"] <= pd.Timestamp(training_end)]
    summary = history.groupby("id")["sales"].agg(
        zero_fraction=lambda values: values.eq(0).mean(),
        mean_demand="mean",
    )
    summary["intermittency_group"] = pd.cut(
        summary["zero_fraction"],
        bins=[-0.001, 0.50, 0.80, 0.95, 1.001],
        labels=["low", "medium", "high", "extreme"],
    )
    summary["volume_group"] = pd.qcut(
        summary["mean_demand"].rank(method="first"),
        q=3,
        labels=["low", "medium", "high"],
    )
    return predictions.merge(summary.reset_index(), on="id", how="left", validate="many_to_one")


def calibration_summary(predictions, group_columns):
    """Summarize interval behavior for one or more grouping columns."""

    rows = []
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    for group_key, part in predictions.groupby(grouper, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key))
        metrics = evaluate_probabilistic_forecast(part)
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def validate_prediction_schema(predictions):
    """Check the shared Phase 2 prediction format."""

    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing schema columns: {sorted(missing)}")
    if predictions[["id", "forecast_origin", "forecast_date"]].duplicated().any():
        raise ValueError("Duplicate series-origin-date forecast keys found.")
    if predictions[["actual", "q10", "q50", "q90"]].isna().any().any():
        raise ValueError("Predictions contain missing values.")
    if not (predictions["q10"] <= predictions["q50"]).all():
        raise ValueError("Quantiles are not ordered.")
    if not (predictions["q50"] <= predictions["q90"]).all():
        raise ValueError("Quantiles are not ordered.")
    return True
