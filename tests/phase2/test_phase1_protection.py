from pathlib import Path
import hashlib
import json

import pandas as pd


def test_saved_phase1_metrics_remain_at_audited_baseline():
    project_root = Path(__file__).resolve().parents[2]
    results = pd.read_csv(project_root / "results" / "phase1_model_comparison.csv")
    metrics = results.set_index("Model")
    assert metrics.loc["Gradient Boosting", "MAE"] == 1.0223863489643086
    assert metrics.loc["Kalman Filter", "MAE"] == 1.059506462336964
    assert metrics.loc["Seasonal Naive", "MAE"] == 1.2096428571428572


def test_phase1_files_match_recorded_sha256_hashes():
    project_root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (project_root / "report" / "phase1_preservation_hashes.json").read_text()
    )
    for relative_path, expected in manifest["sha256"].items():
        digest = hashlib.sha256((project_root / relative_path).read_bytes()).hexdigest()
        assert digest == expected, relative_path
