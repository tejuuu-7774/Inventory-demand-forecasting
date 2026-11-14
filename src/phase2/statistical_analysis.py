"""Paired series-level comparisons for rolling-origin forecasts."""

from pathlib import Path

import numpy as np
import pandas as pd


def series_mae(predictions, prediction_column, model_column=None, model_name=None):
    frame = predictions
    if model_column is not None:
        frame = frame[frame[model_column].eq(model_name)]
    return frame.assign(
        absolute_error=(frame["actual"] - frame[prediction_column]).abs()
    ).groupby("id")["absolute_error"].mean()


def paired_series_bootstrap(first, second, number_of_resamples=10_000, seed=42):
    """Estimate mean(first minus second) with whole-series resampling."""

    aligned = pd.concat([first.rename("first"), second.rename("second")], axis=1).dropna()
    differences = (aligned["first"] - aligned["second"]).to_numpy()
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(differences), size=(number_of_resamples, len(differences)))
    samples = differences[indices].mean(axis=1)
    return {
        "number_of_series": int(len(differences)),
        "mean_mae_difference_first_minus_second": float(differences.mean()),
        "ci_95_lower": float(np.quantile(samples, 0.025)),
        "ci_95_upper": float(np.quantile(samples, 0.975)),
        "first_wins": int((differences < 0).sum()),
        "second_wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "bootstrap_resamples": int(number_of_resamples),
        "random_seed": int(seed),
    }


def run_statistical_comparisons(project_root, number_of_resamples=10_000, seed=42):
    project_root = Path(project_root)
    phase1 = pd.read_csv(project_root / "results" / "phase1_predictions.csv.gz")
    base = pd.read_csv(project_root / "results" / "phase2" / "deepar_base_v2_predictions.csv.gz")
    hybrid = pd.read_csv(
        project_root / "results" / "phase2" / "deepar_kalman_hybrid_v2_predictions.csv.gz"
    )

    seasonal = series_mae(phase1, "prediction", "Model", "Seasonal Naive")
    gradient = series_mae(phase1, "prediction", "Model", "Gradient Boosting")
    deep = series_mae(base, "q50")
    hybrid_mae = series_mae(hybrid, "q50")

    comparisons = [
        ("Seasonal Naive", "Gradient Boosting", seasonal, gradient),
        ("Gradient Boosting", "DeepAR v2", gradient, deep),
        ("DeepAR v2", "Causal hybrid v2", deep, hybrid_mae),
    ]
    rows = []
    for first_name, second_name, first, second in comparisons:
        row = paired_series_bootstrap(first, second, number_of_resamples, seed)
        row.update({"first_model": first_name, "second_model": second_name})
        rows.append(row)

    output = pd.DataFrame(rows)
    output.to_csv(
        project_root / "results" / "phase2" / "statistical_comparisons.csv",
        index=False,
    )
    return output
