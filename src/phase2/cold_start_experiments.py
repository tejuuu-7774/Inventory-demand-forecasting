"""Run short-history, unseen-item and unseen-combination evaluations."""

from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd

from .cold_start import assert_cold_start_separation, make_cold_start_split
from .data_preparation import apply_category_maps, create_forecast_windows, get_training_boundaries
from .deepar_model import (
    choose_device,
    forecast_windows,
    make_fallback_scales,
    make_scale_map,
)
from .evaluation import evaluate_probabilistic_forecast
from .artifacts import configuration_hash, dataframe_hash, dependency_versions
from .experiments import (
    load_or_prepare_phase2_data,
    load_phase2_config,
    load_trained_experiment,
    run_training_experiment,
)


def _metrics_row(predictions, cohort, history_days):
    metrics = evaluate_probabilistic_forecast(predictions)
    metrics["cohort"] = cohort
    metrics["history_days"] = history_days
    metrics["number_of_series"] = int(predictions["id"].nunique())
    return metrics


def run_cold_start_suite(project_root):
    """Run distinct limited-context, limited-history, and zero-shot studies.

    The limited-history cohort uses item IDs that the global model never saw
    during training. At each origin it receives only the stated last k target
    observations, and its scale is calculated from those same observations.
    This transfer design is the simplest defensible alternative to fitting a
    separate embedding with fewer than the 63 observations required by the
    current context-plus-horizon architecture.
    """

    project_root = Path(project_root)
    output_dir = project_root / "results" / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_phase2_config(project_root)
    raw_data = load_or_prepare_phase2_data(project_root)
    windows = create_forecast_windows(
        raw_data,
        config["forecast_horizon"],
        config["number_of_test_windows"],
    )
    boundaries = get_training_boundaries(raw_data, windows, config["validation_days"])

    base_model, base_metadata = load_trained_experiment(project_root, "deepar_base_v2")
    base_data = apply_category_maps(raw_data, base_metadata["category_maps"])
    all_series_ids = sorted(base_data["id"].unique())
    base_scales = make_scale_map(base_data, boundaries["training_end"], all_series_ids)
    category_scales, global_scale = make_fallback_scales(
        base_data,
        boundaries["training_end"],
        all_series_ids,
    )
    device = choose_device(config["device_preference"])
    base_model = base_model.to(device)

    metric_rows = []
    prediction_parts = []
    for history_days in [7, 14, 28, 56]:
        predictions = forecast_windows(
            model=base_model,
            data=base_data,
            windows=windows,
            experiment_id=f"known_item_limited_context_{history_days}_v2",
            model_name=f"DeepAR known-item limited context {history_days} days",
            config=config,
            dynamic_features=base_metadata["dynamic_features"],
            static_columns=base_metadata["static_columns"],
            scale_map=base_scales,
            fallback_category_scales=category_scales,
            fallback_global_scale=global_scale,
            history_limit=history_days,
            sample_seed=config["random_seed"],
            device=device,
        )
        predictions["cohort"] = "known_item_limited_context"
        predictions["history_days"] = history_days
        prediction_parts.append(predictions)
        metric_rows.append(_metrics_row(predictions, "known_item_limited_context", history_days))

    split = make_cold_start_split(
        raw_data,
        item_fraction=config["cold_start_item_fraction"],
        random_seed=config["random_seed"],
    )
    assert_cold_start_separation(raw_data, split)
    (output_dir / "cold_start_split_v2.json").write_text(json.dumps(split, indent=2))

    cold_result = run_training_experiment(
        project_root=project_root,
        experiment_name="deepar",
        experiment_id="cold_start_model_v2",
        model_name="DeepAR cold-start model",
        training_series_ids=split["training_series_ids"],
        forecast_series_ids=(
            split["unseen_item_series_ids"] + split["combination_holdout_series_ids"]
        ),
        cohort="training excludes unseen items and held-out store-item combinations",
    )
    cold_predictions = cold_result["predictions"].copy()

    combo = cold_predictions[
        cold_predictions["id"].isin(split["combination_holdout_series_ids"])
    ].copy()
    combo["cohort"] = "new_store_item_combination"
    combo["history_days"] = "full_available_history"
    prediction_parts.append(combo)
    metric_rows.append(_metrics_row(combo, "new_store_item_combination", "full"))

    cold_data = cold_result["data"]
    cold_training_ids = split["training_series_ids"]
    cold_scales = make_scale_map(cold_data, boundaries["training_end"], cold_training_ids)
    cold_category_scales, cold_global_scale = make_fallback_scales(
        cold_data,
        boundaries["training_end"],
        cold_training_ids,
    )

    # Limited-history transfer on the same unseen-item population. The model
    # has no learned item embedding or training target from these products.
    for history_days in [7, 14, 28, 56]:
        limited = forecast_windows(
            model=cold_result["model"],
            data=cold_data,
            windows=windows,
            experiment_id=f"unseen_item_limited_history_{history_days}_v2",
            model_name=f"DeepAR unseen-item transfer with {history_days} days",
            config=config,
            dynamic_features=cold_result["feature_set"]["dynamic"],
            static_columns=cold_result["feature_set"]["static"],
            scale_map={},
            fallback_category_scales=cold_category_scales,
            fallback_global_scale=cold_global_scale,
            series_ids=split["unseen_item_series_ids"],
            history_limit=history_days,
            use_history_scale=True,
            sample_seed=config["random_seed"],
            device=device,
        )
        limited["cohort"] = "unseen_item_limited_history_transfer"
        limited["history_days"] = history_days
        prediction_parts.append(limited)
        metric_rows.append(
            _metrics_row(limited, "unseen_item_limited_history_transfer", history_days)
        )

    zero_shot = forecast_windows(
        model=cold_result["model"],
        data=cold_data,
        windows=windows,
        experiment_id="unseen_item_zero_shot_v2",
        model_name="DeepAR unseen-item zero-shot",
        config=config,
        dynamic_features=cold_result["feature_set"]["dynamic"],
        static_columns=cold_result["feature_set"]["static"],
        scale_map=cold_scales,
        fallback_category_scales=cold_category_scales,
        fallback_global_scale=cold_global_scale,
        series_ids=split["unseen_item_series_ids"],
        history_limit=0,
        sample_seed=config["random_seed"],
        device=device,
    )
    zero_shot["cohort"] = "unseen_item_zero_shot"
    zero_shot["history_days"] = 0
    prediction_parts.append(zero_shot)
    metric_rows.append(_metrics_row(zero_shot, "unseen_item_zero_shot", 0))

    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    all_predictions.to_csv(
        output_dir / "cold_start_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    metrics.to_csv(output_dir / "cold_start_results.csv", index=False)

    suite_payload = {
        "experiment_id": "cold_start_suite_v2",
        "random_seed": config["random_seed"],
        "model_name": "DeepAR cold-start suite",
        "model_configuration": config,
        "feature_configuration": cold_result["feature_set"],
        "data_cohort": {
            "known_item_series": len(all_series_ids),
            "training_series": len(split["training_series_ids"]),
            "unseen_item_series": len(split["unseen_item_series_ids"]),
            "new_combination_series": len(split["combination_holdout_series_ids"]),
        },
        "forecast_origins": [str(value) for value in windows["origin"]],
        "training_cutoff": str(boundaries["training_end"].date()),
        "dependency_versions": dependency_versions(),
        "device": str(device),
        "data_hash": dataframe_hash(raw_data),
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "known_item_limited_context": "base model saw item history during training",
            "unseen_item_limited_history_transfer": (
                "held-out item targets excluded from training; only last k observations "
                "and their past-only scale are supplied at each origin"
            ),
            "unseen_item_zero_shot": "held-out item targets and target history unavailable",
        },
    }
    suite_payload["configuration_hash"] = configuration_hash(suite_payload)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "cold_start_suite_v2.json").write_text(
        json.dumps(suite_payload, indent=2)
    )
    return metrics, all_predictions, split
