import pandas as pd

from src.phase2.evaluation import (
    PREDICTION_COLUMNS,
    calculate_point_metrics,
    evaluate_probabilistic_forecast,
    validate_prediction_schema,
)


def prediction_frame():
    rows = []
    for step, actual in enumerate([1.0, 2.0, 3.0], start=1):
        rows.append(
            {
                "experiment_id": "test",
                "Model": "Test model",
                "id": "A",
                "item_id": "ITEM_A",
                "store_id": "STORE_1",
                "cat_id": "FOODS",
                "forecast_origin": pd.Timestamp("2020-01-01"),
                "forecast_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=step),
                "horizon_step": step,
                "actual": actual,
                "q10": actual - 1,
                "q50": actual,
                "q90": actual + 1,
            }
        )
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def test_point_metrics_known_answer():
    metrics = calculate_point_metrics([1, 2], [2, 2])
    assert metrics["MAE"] == 0.5
    assert metrics["RMSE"] == 2 ** -0.5
    assert metrics["WAPE_percent"] == 100 / 3


def test_probabilistic_metrics_and_schema():
    predictions = prediction_frame()
    assert validate_prediction_schema(predictions)
    metrics = evaluate_probabilistic_forecast(predictions)
    assert metrics["MAE"] == 0.0
    assert metrics["interval_coverage_80_percent"] == 100.0
    assert metrics["mean_interval_width"] == 2.0
