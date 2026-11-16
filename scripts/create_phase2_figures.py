"""Create compact report figures from saved Phase 2 result tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase2"
FIGURES = ROOT / "report" / "figures" / "phase2"
FIGURES.mkdir(parents=True, exist_ok=True)


def finish(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


phase1 = pd.read_csv(ROOT / "results" / "phase1_model_comparison.csv")
base = pd.read_csv(RESULTS / "deepar_base_v2_metrics.csv")
hybrid = pd.read_csv(RESULTS / "deepar_kalman_hybrid_v2_metrics.csv")
comparison = pd.DataFrame(
    {
        "Model": ["Seasonal Naive", "Kalman", "Gradient Boosting", "DeepAR", "Hybrid"],
        "MAE": [
            phase1.set_index("Model").loc["Seasonal Naive", "MAE"],
            phase1.set_index("Model").loc["Kalman Filter", "MAE"],
            phase1.set_index("Model").loc["Gradient Boosting", "MAE"],
            base.loc[0, "MAE"],
            hybrid.loc[0, "MAE"],
        ],
    }
)
comparison.plot(x="Model", y="MAE", kind="bar", legend=False, color="#4472C4")
plt.ylabel("MAE")
plt.title("Phase 1 and Phase 2 point accuracy")
plt.xticks(rotation=30, ha="right")
finish(FIGURES / "model_mae_comparison.png")


ablation = pd.read_csv(RESULTS / "ablation_results.csv").sort_values("mean_pinball_loss")
ablation.plot(
    x="feature_configuration",
    y="mean_pinball_loss",
    kind="bar",
    legend=False,
    color="#70AD47",
)
plt.ylabel("Mean pinball loss")
plt.xlabel("Feature configuration")
plt.title("DeepAR ablation study")
plt.xticks(rotation=30, ha="right")
finish(FIGURES / "ablation_pinball_loss.png")


cold = pd.read_csv(RESULTS / "cold_start_results.csv")
cold["label"] = cold["cohort"] + " / " + cold["history_days"].astype(str)
cold.plot(x="label", y="MAE", kind="bar", legend=False, color="#ED7D31")
plt.ylabel("MAE")
plt.xlabel("Cold-start cohort / history days")
plt.title("Cold-start forecast difficulty")
plt.xticks(rotation=35, ha="right")
finish(FIGURES / "cold_start_mae.png")


calibration_rows = []
for experiment_id, label in [
    ("deepar_base_v2", "DeepAR"),
    ("deepar_kalman_hybrid_v2", "Causal hybrid"),
]:
    table = pd.read_csv(RESULTS / f"{experiment_id}_calibration.csv")
    table = table[table["group_dimension"].eq("horizon_step")].copy()
    table["Model"] = label
    calibration_rows.append(table)
calibration = pd.concat(calibration_rows)
for model_name, part in calibration.groupby("Model"):
    plt.plot(
        part["group_value"].astype(int),
        part["interval_coverage_80_percent"],
        marker="o",
        label=model_name,
    )
plt.axhline(80, color="black", linestyle="--", linewidth=1, label="Nominal 80%")
plt.xlabel("Forecast horizon step")
plt.ylabel("Empirical coverage (%)")
plt.title("Interval calibration by horizon")
plt.legend(frameon=False)
finish(FIGURES / "coverage_by_horizon.png")


for model_name, part in calibration.groupby("Model"):
    plt.plot(
        part["group_value"].astype(int),
        part["mean_interval_width"],
        marker="o",
        label=model_name,
    )
plt.xlabel("Forecast horizon step")
plt.ylabel("Mean q90 minus q10 width")
plt.title("Interval width by forecast horizon")
plt.legend(frameon=False)
finish(FIGURES / "interval_width_by_horizon.png")


miss_rates = pd.DataFrame(
    {
        "Model": ["DeepAR", "Causal hybrid"],
        "Below q10": [
            base.loc[0, "empirical_q10_below_percent"],
            hybrid.loc[0, "empirical_q10_below_percent"],
        ],
        "Above q90": [
            base.loc[0, "empirical_q90_above_percent"],
            hybrid.loc[0, "empirical_q90_above_percent"],
        ],
    }
).set_index("Model")
miss_rates.plot(kind="bar", color=["#4472C4", "#ED7D31"])
plt.axhline(10, color="black", linestyle="--", linewidth=1, label="Nominal tail rate")
plt.ylabel("Empirical miss rate (%)")
plt.title("Asymmetric interval miss rates")
plt.xticks(rotation=0)
plt.legend(frameon=False)
finish(FIGURES / "calibration_tail_miss_rates.png")


inventory = pd.read_csv(RESULTS / "inventory_simulation_results.csv")
inventory = inventory[inventory["stockout_cost_assumption"].eq(4.0)].copy()
pivot = inventory.pivot(index="policy", columns="Model", values="total_cost")
pivot.plot(kind="bar")
plt.ylabel("Total assumed cost")
plt.xlabel("Inventory policy")
plt.title("Inventory policies at holding:stockout cost 1:4")
plt.xticks(rotation=30, ha="right")
plt.legend(frameon=False)
finish(FIGURES / "inventory_cost_comparison.png")

print("Created figures in", FIGURES)
