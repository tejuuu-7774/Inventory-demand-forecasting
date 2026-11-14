"""Reproducible identifiers and compatibility checks for Phase 2 artifacts."""

from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch


ARCHITECTURE_VERSION = "simple_deepar_v2_causal_kalman"


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json(value):
    """Return stable JSON text suitable for hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def configuration_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def dataframe_hash(data, columns=None):
    """Hash values and row order in the modeling table."""

    if columns is None:
        columns = [
            column
            for column in ["id", "date", "sales", "sell_price", "observation_mask"]
            if column in data.columns
        ]
    stable = data.loc[:, columns].copy()
    if "date" in stable:
        stable["date"] = pd.to_datetime(stable["date"]).astype("int64")
    values = pd.util.hash_pandas_object(stable, index=True).to_numpy(dtype="uint64")
    return hashlib.sha256(values.tobytes()).hexdigest()


def dependency_versions():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
    }


def make_experiment_identity(
    *,
    experiment_id,
    model_name,
    config,
    feature_set,
    data_hash,
    cohort,
    boundaries,
    training_series_ids,
    forecast_series_ids,
):
    """Build immutable identity fields used to validate saved artifacts."""

    signature_payload = {
        "architecture_version": ARCHITECTURE_VERSION,
        "experiment_id": experiment_id,
        "model_name": model_name,
        "random_seed": config["random_seed"],
        "model_configuration": config,
        "feature_configuration": feature_set,
        "data_hash": data_hash,
        "cohort": cohort,
        "boundaries": boundaries,
        "training_series_ids": sorted(training_series_ids),
        "forecast_series_ids": sorted(forecast_series_ids),
        "forecast_horizon": config["forecast_horizon"],
    }
    return {
        **signature_payload,
        "configuration_hash": configuration_hash(signature_payload),
    }


def artifact_is_compatible(metadata_path, expected_configuration_hash):
    """Return False for missing, unreadable, or incompatible metadata."""

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("configuration_hash") == expected_configuration_hash


def validate_experiment_metadata(metadata):
    """Reject incomplete, stale, or internally inconsistent model metadata."""

    identity_keys = [
        "architecture_version",
        "experiment_id",
        "model_name",
        "random_seed",
        "model_configuration",
        "feature_configuration",
        "data_hash",
        "cohort",
        "boundaries",
        "training_series_ids",
        "forecast_series_ids",
        "forecast_horizon",
    ]
    missing = [key for key in identity_keys + ["configuration_hash"] if key not in metadata]
    if missing:
        raise ValueError(f"Checkpoint metadata lacks compatibility fields: {missing}")
    if metadata["architecture_version"] != ARCHITECTURE_VERSION:
        raise ValueError("Checkpoint architecture version is incompatible.")
    identity = {key: metadata[key] for key in identity_keys}
    expected = configuration_hash(identity)
    if metadata["configuration_hash"] != expected:
        raise ValueError("Checkpoint configuration hash does not match its metadata.")
    return True


def write_experiment_metadata(project_root, metadata):
    """Write one metadata file per experiment; never use a shared overwrite."""

    directory = Path(project_root) / "results" / "phase2" / "metadata"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{metadata['experiment_id']}.json"
    path.write_text(json.dumps(metadata, indent=2, default=_json_default))
    return path


def finalize_metadata(identity, *, device, windows, training_examples, validation_examples, training_seconds):
    return {
        **identity,
        "forecast_origins": [pd.Timestamp(value).isoformat() for value in windows["origin"]],
        "training_cutoff": identity["boundaries"]["training_end"],
        "device": str(device),
        "dependency_versions": dependency_versions(),
        "determinism": {
            "python_random_seeded": True,
            "numpy_seeded": True,
            "torch_cpu_seeded": True,
            "torch_accelerator_seeded_when_available": True,
            "deterministic_algorithms": True,
            "mkldnn_disabled": True,
            "cudnn_benchmark_disabled": True,
            "cpu_threads": 1,
        },
        "training_examples": int(training_examples),
        "validation_examples": int(validation_examples),
        "training_seconds": float(training_seconds),
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
