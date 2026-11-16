"""Compare inventory policies for Kalman, DeepAR and the hybrid."""

from pathlib import Path

import pandas as pd

from .inventory import simulate_inventory_policies


def run_inventory_suite(project_root):
    project_root = Path(project_root)
    output_dir = project_root / "results" / "phase2"
    forecasts = []

    for experiment_id, model_name in [
        ("deepar_base_v2", "DeepAR"),
        ("deepar_kalman_hybrid_v2", "DeepAR + causal Kalman hybrid"),
    ]:
        frame = pd.read_csv(output_dir / f"{experiment_id}_predictions.csv.gz")
        forecasts.append((model_name, frame[["actual", "q10", "q50", "q90"]]))

    phase1 = pd.read_csv(project_root / "results" / "phase1_predictions.csv.gz")
    kalman = phase1[phase1["Model"].eq("Kalman Filter")].copy()
    forecasts.append(("Phase 1 Kalman Filter", kalman[["actual", "q10", "q50", "q90"]]))

    rows = []
    for stockout_cost in [1.0, 4.0, 9.0]:
        for model_name, predictions in forecasts:
            result = simulate_inventory_policies(
                predictions,
                holding_cost=1.0,
                stockout_cost=stockout_cost,
            )
            result.insert(0, "Model", model_name)
            rows.append(result)

    comparison = pd.concat(rows, ignore_index=True)
    comparison["cost_note"] = (
        "Assumed teaching costs only; not Walmart or real-company costs. "
        "The newsvendor quantity is linearly interpolated from saved quantiles."
    )
    comparison.to_csv(output_dir / "inventory_simulation_results.csv", index=False)
    return comparison
