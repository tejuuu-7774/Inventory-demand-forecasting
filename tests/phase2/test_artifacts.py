import json

import pandas as pd

from src.phase2.artifacts import (
    artifact_is_compatible,
    configuration_hash,
    dataframe_hash,
    make_experiment_identity,
    validate_experiment_metadata,
)


def test_configuration_hash_is_order_independent_and_detects_change():
    assert configuration_hash({"a": 1, "b": 2}) == configuration_hash({"b": 2, "a": 1})
    assert configuration_hash({"a": 1}) != configuration_hash({"a": 2})


def test_data_hash_changes_when_target_changes():
    data = pd.DataFrame(
        {
            "id": ["A", "A"],
            "date": pd.date_range("2024-01-01", periods=2),
            "sales": [1.0, 2.0],
            "observation_mask": [1.0, 1.0],
        }
    )
    changed = data.copy()
    changed.loc[1, "sales"] = 3.0
    assert dataframe_hash(data) != dataframe_hash(changed)


def test_checkpoint_compatibility_requires_matching_hash(tmp_path):
    metadata = tmp_path / "model.json"
    metadata.write_text(json.dumps({"configuration_hash": "expected"}))
    assert artifact_is_compatible(metadata, "expected")
    assert not artifact_is_compatible(metadata, "different")
    assert not artifact_is_compatible(tmp_path / "missing.json", "expected")


def test_experiment_metadata_detects_tampering():
    identity = make_experiment_identity(
        experiment_id="x",
        model_name="m",
        config={"random_seed": 1, "forecast_horizon": 7},
        feature_set={"dynamic": ["a"], "static": []},
        data_hash="data",
        cohort="all",
        boundaries={"training_end": "2024-01-01"},
        training_series_ids=["A"],
        forecast_series_ids=["A"],
    )
    assert validate_experiment_metadata(identity)
    changed = dict(identity, data_hash="different")
    try:
        validate_experiment_metadata(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("Tampered metadata was accepted")
