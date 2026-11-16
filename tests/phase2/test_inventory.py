import numpy as np
import pandas as pd

from src.phase2.inventory import (
    inventory_cost,
    newsvendor_quantile,
    simulate_inventory_policies,
)


def test_newsvendor_quantile():
    assert newsvendor_quantile(holding_cost=1.0, stockout_cost=4.0) == 0.8


def test_inventory_cost_known_answer():
    overstock, understock, cost = inventory_cost(
        actual=[5, 5],
        order_quantity=[7, 3],
        holding_cost=1,
        stockout_cost=4,
    )
    assert np.array_equal(overstock, [2, 0])
    assert np.array_equal(understock, [0, 2])
    assert np.array_equal(cost, [2, 8])


def test_policy_simulation_returns_all_policies():
    predictions = pd.DataFrame(
        {"actual": [1, 3], "q10": [0, 1], "q50": [1, 2], "q90": [3, 4]}
    )
    results = simulate_inventory_policies(predictions)
    assert len(results) == 4
    assert results["total_cost"].ge(0).all()
    assert results["mean_order_quantity"].ge(0).all()
    assert results["service_level_percent"].between(0, 100).all()
    expected = results["overstock_units"] + results["understock_units"]
    assert expected.ge(0).all()
