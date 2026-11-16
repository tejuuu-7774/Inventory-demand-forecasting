"""Generate the five student-facing Phase 2 notebooks.

The notebooks are generated so their explanatory text and executable cells stay
consistent with the tested Python modules.
"""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "Notebooks"


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save_notebook(filename, cells):
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Phase 2 (Python 3.12)",
            "language": "python",
            "name": "phase2",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    nbf.write(notebook, NOTEBOOK_DIR / filename)


common_setup = """
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))
print("Project root located; Phase 2 modules are importable.")
"""


notebook_04 = [
    markdown(
        """
# 04 Global DeepAR Probabilistic Forecasting

This notebook trains the Phase 2 standalone neural forecast. It does not change any Phase 1 file.

**DeepAR idea:** one small LSTM learns from all item-store series. At each day it receives the previous demand, known features, and static product/store embeddings. It predicts a Negative Binomial distribution instead of one number.

The Negative Binomial is appropriate here because sales are non-negative counts and their variance is much larger than their mean. We extract q10, q50 and q90 from recursive sample paths.
"""
    ),
    markdown(
        """
## 1. Locate the project

The notebook may be launched from the repository root or from `Notebooks/`. This cell finds the root and makes the tested Phase 2 modules importable.
"""
    ),
    code(common_setup),
    markdown(
        """
## 2. Load configuration and prepare data

The input table has one row per series and date. The target is daily `sales`. Calendar and price features are joined without changing Phase 1. A price-present indicator is only an **availability proxy**; it is not verified inventory or a verified stockout.
"""
    ),
    code(
        """
from src.phase2.experiments import prepare_experiment_data

data, windows, boundaries, category_maps, feature_set, config = prepare_experiment_data(
    PROJECT_ROOT,
    experiment_name="deepar",
)

print("Rows:", len(data))
print("Series:", data["id"].nunique())
print("Dynamic features:", feature_set["dynamic"])
print("Static features:", feature_set["static"])
print(windows)
print(boundaries)
"""
    ),
    markdown(
        """
## 3. Inspect availability assumptions

`observation_mask = 1` means a selling price exists and the store is not marked closed for Christmas. Zero sales with a valid price remain genuine observed zeros. We never silently turn ambiguous zero sales into missing targets.
"""
    ),
    code(
        """
availability_summary = data[[
    "is_price_missing",
    "is_available_proxy",
    "is_zero_sales",
    "is_leading_zero",
    "observation_mask",
]].mean()

display(availability_summary.to_frame("fraction of rows"))
"""
    ),
    markdown(
        """
## 4 Construct one sequence

A batch sent to the LSTM has shape `(batch_size, sequence_length, number_of_features)`.

- `batch_size`: independent training examples processed together.
- `sequence_length`: 56 context days plus 7 forecast days = 63.
- `number_of_features`: the dynamic inputs for each day.

During training, actual previous sales are inputs. This is teacher forcing. The target at day `t` never enters the input for day `t`; the input contains sales from day `t-1`. The loss mask selects only the final seven target days.
"""
    ),
    code(
        """
from src.phase2.deepar_model import make_scale_map, make_training_arrays

scale_map = make_scale_map(data, boundaries["training_end"])
training_arrays = make_training_arrays(
    data=data,
    training_end=boundaries["training_end"],
    context_length=config["context_length"],
    horizon=config["forecast_horizon"],
    stride=config["training_stride"],
    dynamic_features=feature_set["dynamic"],
    static_columns=feature_set["static"],
    scale_map=scale_map,
)

print("Previous-sales tensor:", training_arrays["previous_sales"].shape)
print("Dynamic-feature tensor:", training_arrays["dynamic"].shape)
print("Static-category tensor:", training_arrays["static"].shape)
print("Target tensor:", training_arrays["target"].shape)
print("Loss-mask tensor:", training_arrays["loss_mask"].shape)
print("First example forecast mask:", training_arrays["loss_mask"][0, -10:])
"""
    ),
    markdown(
        """
## 5 Prepare a batch

The dataset converts NumPy arrays to PyTorch tensors. A data loader groups examples into a batch. Its dynamic tensor has shape `(batch size, 63 days, dynamic features)`. Static identifiers have no time dimension because one product and store identity applies to the whole sequence.
"""
    ),
    code(
        """
from torch.utils.data import DataLoader
from src.phase2.deepar_model import SequenceDataset

demo_loader = DataLoader(SequenceDataset(training_arrays), batch_size=4, shuffle=False)
demo_batch = next(iter(demo_loader))
for name, tensor in demo_batch.items():
    print(f"{name:>14}: {tuple(tensor.shape)}")
"""
    ),
    markdown(
        """
## 6 Pass the sequence through the LSTM

The model first scales and transforms previous demand with `log1p`. It concatenates that value with dynamic features and repeated static embeddings. The LSTM returns one hidden representation per day. Two linear heads turn each hidden vector into positive distribution parameters by applying `softplus`.

The first parameter is the Negative Binomial mean. The second is dispersion. In this implementation, `variance = mean + dispersion × mean²`. Positive dispersion lets the variance exceed the mean, which is common for intermittent retail counts.
"""
    ),
    code(
        """
import torch
from src.phase2.deepar_model import SimpleDeepAR

cardinalities = {name: len(category_maps[name]) + 1 for name in feature_set["static"]}
demo_model = SimpleDeepAR(
    number_of_dynamic_features=len(feature_set["dynamic"]),
    static_cardinalities=cardinalities,
    hidden_size=config["hidden_size"],
    number_of_layers=config["number_of_lstm_layers"],
    embedding_size=config["embedding_size"],
)
mean, dispersion, hidden = demo_model(
    demo_batch["previous_sales"], demo_batch["dynamic"],
    demo_batch["static"], demo_batch["scale"],
)
print("Mean shape:", tuple(mean.shape))
print("Dispersion shape:", tuple(dispersion.shape))
print("LSTM hidden-state shape:", tuple(hidden[0].shape))
print("All parameters positive:", bool((mean > 0).all() and (dispersion > 0).all()))
"""
    ),
    markdown(
        """
## 7 Calculate the Negative Binomial loss and update weights

The Negative Binomial assigns a probability to each observed count. Training minimizes negative log likelihood, so the model is penalized when it assigns low probability to the actual demand. Multiplication by the loss mask excludes the context days. `backward()` calculates gradients, gradient clipping limits unstable updates, and `optimizer.step()` changes the weights.
"""
    ),
    code(
        """
from src.phase2.deepar_model import masked_negative_binomial_loss

optimizer = torch.optim.Adam(demo_model.parameters(), lr=config["learning_rate"])
optimizer.zero_grad()
loss = masked_negative_binomial_loss(
    mean, dispersion, demo_batch["target"], demo_batch["loss_mask"]
)
loss.backward()
gradient_norm = torch.nn.utils.clip_grad_norm_(
    demo_model.parameters(), config["gradient_clip_norm"]
)
optimizer.step()
print("One teaching-step loss:", float(loss.detach()))
print("Gradient norm before clipping:", float(gradient_norm))
"""
    ),
    markdown(
        """
## 8 Train the full model

The LSTM hidden state is a learned numerical memory of recent demand and covariates. Two output heads produce the mean and dispersion of a Negative Binomial distribution. At prediction time, sampled demand is fed back into the next step, so the seven-day forecast is genuinely autoregressive.

The model trains only through `training_end`, selects its checkpoint using the following 14 validation days, and evaluates on the untouched final 28 days.
"""
    ),
    code(
        """
from src.phase2.experiments import run_training_experiment

result = run_training_experiment(
    project_root=PROJECT_ROOT,
    experiment_name="deepar",
    experiment_id="deepar_base_v2",
    model_name="DeepAR + price + availability",
)

display(result["history"])
"""
    ),
    markdown(
        """
## 9 Generate recursive sample paths and quantiles

At the forecast origin, the LSTM reads the observed context once. For forecast step one, it samples a count from the predicted Negative Binomial distribution. That sampled count becomes the previous-demand input for step two. Repeating this process creates a sample path without using future actual sales. One hundred paths form an empirical forecast distribution. The 10th, 50th, and 90th percentiles become q10, q50, and q90.
"""
    ),
    code(
        """
first_series_id = result["predictions"]["id"].iloc[0]
one_series = result["predictions"][result["predictions"]["id"].eq(first_series_id)]
display(one_series[["forecast_origin", "forecast_date", "actual", "q10", "q50", "q90"]].head(7))
print("Quantiles ordered:", bool(
    (one_series["q10"] <= one_series["q50"]).all()
    and (one_series["q50"] <= one_series["q90"]).all()
))
"""
    ),
    markdown(
        """
## 10 Evaluate point accuracy and uncertainty

q50 is median predicted demand. q10 and q90 bound an 80% central interval. Over many representative forecasts, an 80% interval should cover roughly 80% of actual values, but useful intervals must also remain narrow.
"""
    ),
    code(
        """
metrics = result["metrics"]
display(result["predictions"].head(10))
display({name: value for name, value in metrics.items() if name != "device"})
"""
    ),
    markdown(
        """
## 11 Saved outputs

The checkpoint, configuration, training history, predictions, aggregate metrics and calibration summaries are saved under `models/phase2/` and `results/phase2/`. The prediction table contains one row per series, origin and forecast day.
"""
    ),
    code(
        """
output_files = sorted((PROJECT_ROOT / "results" / "phase2").glob("deepar_base_v2*"))
for path in output_files:
    print(path.relative_to(PROJECT_ROOT))
"""
    ),
]


notebook_05 = [
    markdown(
        """
# 05 Kalman and DeepAR Hybrid

This experiment tests a meaningful hybrid: causal Kalman state estimates become historical covariates for DeepAR. Forecasts are not simply averaged.

For each day, the Kalman features are calculated **before** observing that day's sales. For future days, the filtered level is projected forward, uncertainty grows with the horizon, and no future actual sales are used.
"""
    ),
    markdown("## 1. Locate the project and import the experiment runner"),
    code(common_setup),
    markdown(
        """
## 2. Inspect the hybrid features

The three added inputs are:

- filtered local demand level;
- state variance, representing local uncertainty;
- the previous innovation, or recent Kalman forecast error.

They use an expanding past-only scale, so a target can affect only later rows. High-volume items therefore do not dominate the network, and validation or test targets cannot normalize training features.
"""
    ),
    code(
        """
from src.phase2.experiments import prepare_experiment_data

data, windows, boundaries, maps, feature_set, config = prepare_experiment_data(
    PROJECT_ROOT,
    experiment_name="hybrid",
)

kalman_columns = [name for name in feature_set["dynamic"] if name.startswith("kalman_")]
print("Kalman inputs:", kalman_columns)
display(data[["id", "date", "sales"] + kalman_columns].head(10))
"""
    ),
    markdown(
        """
## 3. Train the hybrid

Architecture and training boundaries remain the same as standalone DeepAR. Therefore, any measured difference is attributable to the added Kalman state information rather than a larger network or a different test population.
"""
    ),
    code(
        """
from src.phase2.experiments import run_training_experiment

hybrid = run_training_experiment(
    project_root=PROJECT_ROOT,
    experiment_name="hybrid",
    experiment_id="deepar_kalman_hybrid_v2",
    model_name="DeepAR + price + causal Kalman hybrid",
)

display(hybrid["history"])
"""
    ),
    markdown("## 4. Compare standalone DeepAR and the hybrid on the same 2,800 observations"),
    code(
        """
import pandas as pd

result_dir = PROJECT_ROOT / "results" / "phase2"
base_metrics = pd.read_csv(result_dir / "deepar_base_v2_metrics.csv")
hybrid_metrics = pd.read_csv(result_dir / "deepar_kalman_hybrid_v2_metrics.csv")
comparison = pd.concat([base_metrics, hybrid_metrics], ignore_index=True)

columns = [
    "Model", "MAE", "RMSE", "WAPE_percent", "mean_pinball_loss",
    "interval_coverage_80_percent", "mean_interval_width", "training_seconds",
]
display(comparison[columns])
"""
    ),
    markdown(
        """
## 5. Interpretation

A hybrid is supported only if the measured metrics improve. Coverage should be read together with interval width: extremely wide intervals can obtain high coverage without being operationally useful.
"""
    ),
]


notebook_06 = [
    markdown(
        """
# 06 Cold Start Evaluation

This notebook keeps cold-start populations separate from the main benchmark.

- **Known-item limited context:** the base model receives only 7, 14, 28 or 56 recent days, but it has learned those items during training. This is a context sensitivity check, not cold start.
- **Unseen-item limited-history transfer:** item IDs and all their targets are excluded from model training. At prediction time, only the stated last 7, 14, 28 or 56 observations and their past-only scale are supplied.
- **Unseen item:** all target history for selected item IDs is excluded from training.
- **New combination:** a store-item series is excluded while the same item remains visible in another store.
- **Zero shot:** an unseen item receives no demand history; only shared metadata and known future covariates are available.
"""
    ),
    markdown("## 1. Locate the project"),
    code(common_setup),
    markdown(
        """
## 2. Run the cold-start suite

Unknown item categories map safely to index zero. The unseen-item assertion verifies that held-out item sales never enter model training. A separate model for each history length cannot learn with the current 56-day context plus 7-day target requirement, so the notebook uses a defensible transfer design trained only on other products. These results are not directly combined with the 100-series headline result because the cohorts differ.
"""
    ),
    code(
        """
from src.phase2.cold_start_experiments import run_cold_start_suite

cold_metrics, cold_predictions, split = run_cold_start_suite(PROJECT_ROOT)
display(cold_metrics)
"""
    ),
    markdown("## 3. Inspect the cohort sizes and separation"),
    code(
        """
print("Training series:", len(split["training_series_ids"]))
print("Unseen-item series:", len(split["unseen_item_series_ids"]))
print("New-combination series:", len(split["combination_holdout_series_ids"]))
print("Unseen item IDs:", len(split["unseen_items"]))
"""
    ),
    markdown(
        """
## 4. Limitations

Only three repeated items exist in the sampled 100 series, so the new-combination cohort is small. Zero-shot scale is estimated from training items in the same broad category; it does not use held-out target history. A larger M5 sample would give more reliable cold-start subgroup estimates.
"""
    ),
]


notebook_07 = [
    markdown(
        """
# 07 Feature and Hybrid Ablation Study

An ablation changes one input group at a time while keeping the LSTM size, training dates, validation dates, forecast horizon and test keys consistent. This lets us measure which components actually help.
"""
    ),
    markdown("## 1. Locate the project"),
    code(common_setup),
    markdown(
        """
## 2. Run six controlled experiments

The suite compares target-only DeepAR, calendar inputs, static categories, price and availability, causal Kalman states, and the complete hybrid. Every v2 configuration is retrained so an incompatible v1 checkpoint cannot be reused.
"""
    ),
    code(
        """
from src.phase2.ablation import run_ablation_suite

ablation_results = run_ablation_suite(PROJECT_ROOT)
columns = [
    "feature_configuration", "MAE", "RMSE", "WAPE_percent",
    "mean_pinball_loss", "interval_coverage_80_percent",
    "mean_interval_width", "training_seconds",
]
display(ablation_results[columns])
"""
    ),
    markdown(
        """
## 3. Interpret without cherry-picking

A component is considered helpful only when the controlled experiment improves relevant metrics. It is acceptable—and scientifically useful—for calendar or embedding features to fail on this small sample. The results table, not the intended story, determines the conclusion.
"""
    ),
    code(
        """
best = ablation_results.sort_values("mean_pinball_loss").iloc[0]
print("Best probabilistic configuration:", best["feature_configuration"])
print("Mean pinball loss:", round(best["mean_pinball_loss"], 4))
"""
    ),
]


notebook_08 = [
    markdown(
        """
# 08 Inventory Decision Simulation

Forecast accuracy matters because it changes ordering decisions. This notebook applies the same actual demand observations to several forecast-based policies.

The costs below are transparent teaching assumptions—not Walmart costs. Holding cost penalizes leftover units; stockout cost penalizes unmet units.
"""
    ),
    markdown("## 1. Locate the project"),
    code(common_setup),
    markdown(
        """
## 2. Understand the newsvendor rule

For holding cost `Co` and stockout cost `Cu`, the one-period newsvendor critical quantile is `Cu / (Cu + Co)`. When stockouts are four times as expensive as holding, the target quantile is 0.8.
"""
    ),
    code(
        """
from src.phase2.inventory import newsvendor_quantile

for stockout_cost in [1.0, 4.0, 9.0]:
    quantile = newsvendor_quantile(holding_cost=1.0, stockout_cost=stockout_cost)
    print(f"Stockout cost {stockout_cost:>3}: critical quantile = {quantile:.2f}")
"""
    ),
    markdown(
        """
## 3. Simulate policies

We compare median ordering, q90 ordering, midpoint safety stock, and an approximate newsvendor quantity interpolated from the saved q10/q50/q90 values. The interpolation is explicitly approximate because full predictive samples were not saved.
"""
    ),
    code(
        """
from src.phase2.inventory_experiments import run_inventory_suite

inventory_results = run_inventory_suite(PROJECT_ROOT)
display(inventory_results)
"""
    ),
    markdown(
        """
## 4. Compare policies at the main 1:4 cost ratio

Service level measures the fraction of demanded units fulfilled. Lower total assumed cost is better, but the preferred policy changes when holding and stockout costs change.
"""
    ),
    code(
        """
main_cost_case = inventory_results[
    inventory_results["stockout_cost_assumption"].eq(4.0)
].sort_values("total_cost")

display(main_cost_case[[
    "Model", "policy", "service_level_percent", "overstock_units",
    "understock_units", "total_cost", "mean_cost_per_forecast",
]])
"""
    ),
    markdown(
        """
## 5. Limitation

This is a one-period simulation: it does not model lead time, replenishment constraints, shelf capacity, spoilage, backorders or multi-echelon inventory. Its purpose is to demonstrate how forecast distributions support decisions beyond point accuracy.
"""
    ),
]


save_notebook("04_deep_learning_forecasting.ipynb", notebook_04)
save_notebook("05_kalman_deepar_hybrid.ipynb", notebook_05)
save_notebook("06_cold_start_evaluation.ipynb", notebook_06)
save_notebook("07_ablation_study.ipynb", notebook_07)
save_notebook("08_inventory_simulation.ipynb", notebook_08)

print("Generated Phase 2 notebooks in", NOTEBOOK_DIR)
