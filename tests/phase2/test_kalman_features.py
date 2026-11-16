import numpy as np
import pandas as pd

from src.phase2.kalman_features import add_causal_kalman_features, projected_kalman_features


def test_first_kalman_feature_does_not_use_first_target():
    data = pd.DataFrame(
        {
            "id": ["A", "A", "A"],
            "date": pd.date_range("2020-01-01", periods=3),
            "sales": [100.0, 0.0, 0.0],
            "observation_mask": [1.0, 1.0, 1.0],
        }
    )
    featured = add_causal_kalman_features(data)
    assert featured.loc[0, "kalman_level_scaled"] == 0.0
    assert featured.loc[1, "kalman_level_scaled"] > 0.0


def _series(values):
    return pd.DataFrame(
        {
            "id": "A",
            "date": pd.date_range("2024-01-01", periods=len(values)),
            "sales": values,
            "observation_mask": np.ones(len(values), dtype="float32"),
        }
    )


def test_future_targets_cannot_change_earlier_kalman_features():
    cutoff = 7
    original = _series([1, 0, 3, 1, 5, 2, 0, 4, 1, 2])
    changed = original.copy()
    changed.loc[cutoff:, "sales"] = [1000, 2000, 3000]

    first = add_causal_kalman_features(original)
    second = add_causal_kalman_features(changed)
    columns = [
        "kalman_scale_past_only",
        "kalman_level_scaled",
        "kalman_variance_scaled",
        "kalman_innovation_scaled",
    ]
    np.testing.assert_allclose(first.loc[:cutoff, columns], second.loc[:cutoff, columns])


def test_scaling_is_the_mean_of_strictly_past_observations():
    result = add_causal_kalman_features(_series([2.0, 4.0, 100.0]))
    np.testing.assert_allclose(result["kalman_scale_past_only"], [1.0, 2.0, 3.0])


def test_test_period_sales_do_not_change_training_period_scale():
    original = _series([0, 2, 4, 6, 8, 10])
    changed = original.copy()
    changed.loc[4:, "sales"] = [900, 1100]
    first = add_causal_kalman_features(original)
    second = add_causal_kalman_features(changed)
    np.testing.assert_allclose(
        first.loc[:4, "kalman_scale_past_only"],
        second.loc[:4, "kalman_scale_past_only"],
    )


def test_causal_features_and_projection_are_finite():
    data = _series([0.0] * 12)
    result = add_causal_kalman_features(data)
    projected = projected_kalman_features(data, 7)
    assert np.isfinite(result.filter(like="kalman_").to_numpy()).all()
    assert np.isfinite(projected.to_numpy()).all()
