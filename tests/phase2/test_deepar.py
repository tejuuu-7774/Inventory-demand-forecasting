import numpy as np
import pandas as pd
import torch

from src.phase2.deepar_model import (
    SimpleDeepAR,
    forecast_one_series,
    make_training_arrays,
    negative_binomial_distribution,
    set_random_seeds,
)


def encoded_series(number_of_days=80):
    dates = pd.date_range("2020-01-01", periods=number_of_days)
    return pd.DataFrame(
        {
            "id": "A",
            "date": dates,
            "sales": np.arange(number_of_days) % 4,
            "observation_mask": 1.0,
            "known_feature": np.arange(number_of_days) / number_of_days,
            "item_id_index": 1,
        }
    )


def test_sequence_shapes_and_forecast_mask():
    arrays = make_training_arrays(
        data=encoded_series(),
        training_end="2020-03-20",
        context_length=14,
        horizon=7,
        stride=7,
        dynamic_features=["known_feature", "observation_mask"],
        static_columns=["item_id"],
        scale_map={"A": 2.0},
    )
    assert arrays["dynamic"].shape[1:] == (21, 2)
    assert arrays["target"].shape[1] == 21
    assert np.all(arrays["loss_mask"][:, :14] == 0)
    assert np.all(arrays["loss_mask"][:, -7:] == 1)


def test_model_tensor_shapes_and_distribution_mean():
    model = SimpleDeepAR(
        number_of_dynamic_features=2,
        static_cardinalities={"item_id": 3},
        hidden_size=8,
        embedding_size=2,
    )
    previous = torch.ones(4, 21)
    dynamic = torch.zeros(4, 21, 2)
    static = torch.ones(4, 1, dtype=torch.long)
    scale = torch.ones(4)
    mean, dispersion, _ = model(previous, dynamic, static, scale)
    assert mean.shape == (4, 21)
    assert dispersion.shape == (4, 21)
    distribution = negative_binomial_distribution(mean, dispersion)
    assert torch.allclose(distribution.mean, mean, atol=1e-4)
    assert torch.isfinite(mean).all() and torch.all(mean > 0)
    assert torch.isfinite(dispersion).all() and torch.all(dispersion > 0)


def test_recursive_forecast_is_nonnegative_ordered_and_future_target_invariant():
    data = encoded_series(30)
    origin = data.loc[22, "date"]
    model = SimpleDeepAR(
        number_of_dynamic_features=2,
        static_cardinalities={"item_id": 3},
        hidden_size=8,
        embedding_size=2,
    )
    changed = data.copy()
    changed.loc[changed["date"] > origin, "sales"] = 9999

    set_random_seeds(123)
    future, quantiles = forecast_one_series(
        model, data, origin, 7, 14,
        ["known_feature", "observation_mask"], ["item_id"],
        2.0, 40, torch.device("cpu"),
    )
    set_random_seeds(123)
    _, changed_quantiles = forecast_one_series(
        model, changed, origin, 7, 14,
        ["known_feature", "observation_mask"], ["item_id"],
        2.0, 40, torch.device("cpu"),
    )
    assert len(future) == 7
    assert quantiles.shape == (3, 7)
    assert (quantiles >= 0).all()
    assert (quantiles[0] <= quantiles[1]).all()
    assert (quantiles[1] <= quantiles[2]).all()
    np.testing.assert_allclose(quantiles, changed_quantiles)
