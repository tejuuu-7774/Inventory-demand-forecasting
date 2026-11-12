"""Small orchestration helpers used by the Phase 2 teaching notebooks."""

from pathlib import Path
import json

import pandas as pd
import torch

from .data_preparation import (
    apply_category_maps,
    build_category_maps,
    create_forecast_windows,
    get_training_boundaries,
    prepare_phase2_data,
)
from .artifacts import (
    dataframe_hash,
    finalize_metadata,
    make_experiment_identity,
    write_experiment_metadata,
    validate_experiment_metadata,
)
from .deepar_model import (
    SimpleDeepAR,
    choose_device,
    forecast_windows,
    make_fallback_scales,
    make_origin_arrays,
    make_scale_map,
    make_training_arrays,
    save_checkpoint,
    set_random_seeds,
    train_deepar,
)
from .evaluation import (
    add_demand_segments,
    calibration_summary,
    evaluate_probabilistic_forecast,
)
from .features import get_feature_set
from .kalman_features import add_causal_kalman_features


def load_phase2_config(project_root):
    path = Path(project_root) / "configs" / "phase2_config.json"
    return json.loads(path.read_text())


def load_or_prepare_phase2_data(project_root, include_kalman=False):
    """Use cached processed data when available; raw data remains untouched."""

    project_root = Path(project_root)
    base_path = project_root / "Datasets" / "processed" / "phase2_prepared.pkl"
    if base_path.exists():
        data = pd.read_pickle(base_path)
        data["date"] = pd.to_datetime(data["date"])
    else:
        data = prepare_phase2_data(project_root, save=True)

    if include_kalman:
        kalman_path = project_root / "Datasets" / "processed" / "phase2_with_kalman_v2_causal.pkl"
        if kalman_path.exists():
            data = pd.read_pickle(kalman_path)
            data["date"] = pd.to_datetime(data["date"])
        else:
            data = add_causal_kalman_features(data)
            data.to_pickle(kalman_path)
    return data


def prepare_experiment_data(project_root, experiment_name, training_series_ids=None):
    """Prepare features, category encodings, windows and split boundaries."""

    config = load_phase2_config(project_root)
    feature_set = get_feature_set(experiment_name)
    needs_kalman = any(name.startswith("kalman_") for name in feature_set["dynamic"])
    data = load_or_prepare_phase2_data(project_root, include_kalman=needs_kalman)
    windows = create_forecast_windows(
        data,
        horizon=config["forecast_horizon"],
        number_of_windows=config["number_of_test_windows"],
    )
    boundaries = get_training_boundaries(data, windows, config["validation_days"])

    if training_series_ids is None:
        training_series_ids = sorted(data["id"].unique())
    category_maps = build_category_maps(data, training_series_ids)
    encoded = apply_category_maps(data, category_maps)
    return encoded, windows, boundaries, category_maps, feature_set, config


def run_training_experiment(
    project_root,
    experiment_name,
    experiment_id,
    model_name,
    training_series_ids=None,
    forecast_series_ids=None,
    maximum_epochs=None,
    cohort="all_selected_series",
):
    """Train one model, forecast the final windows and save all outputs."""

    project_root = Path(project_root)
    data, windows, boundaries, category_maps, feature_set, config = prepare_experiment_data(
        project_root,
        experiment_name,
        training_series_ids=training_series_ids,
    )
    if maximum_epochs is not None:
        config = config.copy()
        config["maximum_epochs"] = int(maximum_epochs)
    if training_series_ids is None:
        training_series_ids = sorted(data["id"].unique())
    if forecast_series_ids is None:
        forecast_series_ids = sorted(data["id"].unique())

    boundary_strings = {name: str(value.date()) for name, value in boundaries.items()}
    identity = make_experiment_identity(
        experiment_id=experiment_id,
        model_name=model_name,
        config=config,
        feature_set=feature_set,
        data_hash=dataframe_hash(data),
        cohort=cohort,
        boundaries=boundary_strings,
        training_series_ids=training_series_ids,
        forecast_series_ids=forecast_series_ids,
    )

    set_random_seeds(config["random_seed"])
    device = choose_device(config["device_preference"])
    scale_map = make_scale_map(data, boundaries["training_end"], training_series_ids)
    category_scales, global_scale = make_fallback_scales(
        data,
        boundaries["training_end"],
        training_series_ids,
    )

    training_arrays = make_training_arrays(
        data=data,
        training_end=boundaries["training_end"],
        context_length=config["context_length"],
        horizon=config["forecast_horizon"],
        stride=config["training_stride"],
        dynamic_features=feature_set["dynamic"],
        static_columns=feature_set["static"],
        scale_map=scale_map,
        series_ids=training_series_ids,
    )
    validation_origins = [
        boundaries["validation_start"] - pd.Timedelta(days=1),
        boundaries["validation_start"] + pd.Timedelta(days=6),
    ]
    validation_arrays = make_origin_arrays(
        data=data,
        origins=validation_origins,
        context_length=config["context_length"],
        horizon=config["forecast_horizon"],
        dynamic_features=feature_set["dynamic"],
        static_columns=feature_set["static"],
        scale_map=scale_map,
        series_ids=training_series_ids,
    )

    static_cardinalities = {
        column: len(category_maps[column]) + 1 for column in feature_set["static"]
    }
    model = SimpleDeepAR(
        number_of_dynamic_features=len(feature_set["dynamic"]),
        static_cardinalities=static_cardinalities,
        hidden_size=config["hidden_size"],
        number_of_layers=config["number_of_lstm_layers"],
        embedding_size=config["embedding_size"],
    )
    model, history, training_seconds = train_deepar(
        model,
        training_arrays,
        validation_arrays,
        config,
        device,
    )

    metadata = {
        **identity,
        "experiment_name": experiment_name,
        "dynamic_features": feature_set["dynamic"],
        "static_columns": feature_set["static"],
        "static_cardinalities": static_cardinalities,
        "category_maps": category_maps,
        "config": config,
        "training_examples": int(len(training_arrays["target"])),
        "validation_examples": int(len(validation_arrays["target"])),
        "device": str(device),
        "training_seconds": training_seconds,
        "negative_binomial_parameterization": (
            "variance = mean + dispersion * mean^2; total_count = 1/dispersion"
        ),
    }
    checkpoint_path = project_root / "models" / "phase2" / f"{experiment_id}.pt"
    save_checkpoint(model, checkpoint_path, metadata)

    experiment_metadata = finalize_metadata(
        identity,
        device=device,
        windows=windows,
        training_examples=len(training_arrays["target"]),
        validation_examples=len(validation_arrays["target"]),
        training_seconds=training_seconds,
    )
    experiment_metadata.update(
        {
            "experiment_name": experiment_name,
            "category_cardinalities": static_cardinalities,
            "availability_note": (
                "Price presence and Christmas closure form an availability proxy; "
                "the project has no verified inventory-on-hand or stockout labels."
            ),
        }
    )
    write_experiment_metadata(project_root, experiment_metadata)

    predictions = forecast_windows(
        model=model,
        data=data,
        windows=windows,
        experiment_id=experiment_id,
        model_name=model_name,
        config=config,
        dynamic_features=feature_set["dynamic"],
        static_columns=feature_set["static"],
        scale_map=scale_map,
        fallback_category_scales=category_scales,
        fallback_global_scale=global_scale,
        series_ids=forecast_series_ids,
        device=device,
        sample_seed=config["random_seed"],
    )
    metrics = evaluate_probabilistic_forecast(predictions)
    metrics.update(
        {
            "experiment_id": experiment_id,
            "Model": model_name,
            "training_seconds": training_seconds,
            "training_examples": len(training_arrays["target"]),
            "epochs_completed": len(history),
            "device": str(device),
        }
    )

    output_dir = project_root / "results" / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / f"{experiment_id}_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame([metrics]).to_csv(output_dir / f"{experiment_id}_metrics.csv", index=False)
    history.to_csv(output_dir / f"{experiment_id}_training_history.csv", index=False)

    segmented = add_demand_segments(predictions, data, boundaries["training_end"])
    calibration_parts = []
    for column in ["horizon_step", "intermittency_group", "volume_group", "cat_id", "store_id"]:
        summary = calibration_summary(segmented, [column])
        summary.insert(0, "group_dimension", column)
        summary = summary.rename(columns={column: "group_value"})
        calibration_parts.append(summary)
    calibration = pd.concat(calibration_parts, ignore_index=True)
    calibration.to_csv(output_dir / f"{experiment_id}_calibration.csv", index=False)

    return {
        "model": model,
        "data": data,
        "windows": windows,
        "boundaries": boundaries,
        "category_maps": category_maps,
        "feature_set": feature_set,
        "config": config,
        "history": history,
        "predictions": predictions,
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
    }


def load_trained_experiment(project_root, experiment_id):
    """Rebuild a trained model from its saved JSON and checkpoint."""

    project_root = Path(project_root)
    checkpoint = project_root / "models" / "phase2" / f"{experiment_id}.pt"
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    validate_experiment_metadata(metadata)
    saved_config = metadata.get("config", metadata["model_configuration"])
    model = SimpleDeepAR(
        number_of_dynamic_features=len(metadata["dynamic_features"]),
        static_cardinalities=metadata["static_cardinalities"],
        hidden_size=saved_config["hidden_size"],
        number_of_layers=saved_config["number_of_lstm_layers"],
        embedding_size=saved_config["embedding_size"],
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model, metadata
