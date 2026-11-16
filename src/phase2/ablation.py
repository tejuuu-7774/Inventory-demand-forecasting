"""Train and collect the six required feature ablations."""

from pathlib import Path

import pandas as pd

from .experiments import run_training_experiment


ABLATIONS = [
    ("target_only", "ablation_target_only_v2", "DeepAR target only"),
    ("calendar", "ablation_calendar_v2", "DeepAR + calendar"),
    ("calendar_static", "ablation_calendar_static_v2", "DeepAR + calendar + static"),
    ("deepar", "deepar_base_v2", "DeepAR + price + availability"),
    ("kalman_only_addition", "ablation_kalman_v2", "DeepAR + causal Kalman features"),
    ("hybrid", "deepar_kalman_hybrid_v2", "DeepAR + price + causal Kalman hybrid"),
]


def run_ablation_suite(project_root):
    """Retrain every v2 ablation and combine compatible saved metrics.

    A full retrain is intentional: v1 artifacts predate causal Kalman scaling
    and experiment signatures, so they are never silently reused.
    """

    project_root = Path(project_root)
    output_dir = project_root / "results" / "phase2"
    rows = []

    for feature_name, experiment_id, model_name in ABLATIONS:
        metrics_path = output_dir / f"{experiment_id}_metrics.csv"
        run_training_experiment(
            project_root,
            experiment_name=feature_name,
            experiment_id=experiment_id,
            model_name=model_name,
        )
        row = pd.read_csv(metrics_path).iloc[0].to_dict()
        row["feature_configuration"] = feature_name
        rows.append(row)

    results = pd.DataFrame(rows)
    results = results.sort_values("mean_pinball_loss").reset_index(drop=True)
    results.to_csv(output_dir / "ablation_results.csv", index=False)
    return results
