"""Simple one-period inventory decisions using forecast quantiles."""

import numpy as np
import pandas as pd


def newsvendor_quantile(holding_cost, stockout_cost):
    """Return the critical demand quantile Cu / (Cu + Co)."""

    if holding_cost < 0 or stockout_cost < 0:
        raise ValueError("Inventory costs must be non-negative.")
    total = holding_cost + stockout_cost
    if total == 0:
        raise ValueError("At least one inventory cost must be positive.")
    return stockout_cost / total


def inventory_cost(actual, order_quantity, holding_cost=1.0, stockout_cost=4.0):
    """Calculate overstock, understock and total cost for one decision."""

    actual = np.asarray(actual, dtype=float)
    order_quantity = np.asarray(order_quantity, dtype=float)
    overstock = np.maximum(order_quantity - actual, 0.0)
    understock = np.maximum(actual - order_quantity, 0.0)
    total_cost = holding_cost * overstock + stockout_cost * understock
    return overstock, understock, total_cost


def simulate_inventory_policies(predictions, holding_cost=1.0, stockout_cost=4.0):
    """Compare median, q90 and simple safety-stock ordering policies."""

    critical_quantile = newsvendor_quantile(holding_cost, stockout_cost)
    if 0.5 <= critical_quantile <= 0.9:
        interpolation_weight = (critical_quantile - 0.5) / 0.4
        newsvendor_quantity = predictions["q50"] + interpolation_weight * (
            predictions["q90"] - predictions["q50"]
        )
    elif critical_quantile < 0.5:
        interpolation_weight = critical_quantile / 0.5
        newsvendor_quantity = predictions["q10"] + interpolation_weight * (
            predictions["q50"] - predictions["q10"]
        )
    else:
        newsvendor_quantity = predictions["q90"]

    policies = {
        "median_q50": predictions["q50"].to_numpy(),
        "high_service_q90": predictions["q90"].to_numpy(),
        "midpoint_safety_stock": (
            predictions["q50"] + 0.5 * (predictions["q90"] - predictions["q50"])
        ).to_numpy(),
        "interpolated_newsvendor_quantile": newsvendor_quantity.to_numpy(),
    }
    rows = []
    actual = predictions["actual"].to_numpy()

    for policy_name, quantity in policies.items():
        quantity = np.maximum(np.rint(quantity), 0.0)
        overstock, understock, total_cost = inventory_cost(
            actual,
            quantity,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
        )
        demand_total = actual.sum()
        fulfilled = np.minimum(actual, quantity).sum()
        rows.append(
            {
                "policy": policy_name,
                "holding_cost_assumption": holding_cost,
                "stockout_cost_assumption": stockout_cost,
                "newsvendor_critical_quantile": critical_quantile,
                "mean_order_quantity": float(quantity.mean()),
                "overstock_units": float(overstock.sum()),
                "understock_units": float(understock.sum()),
                "service_level_percent": float(100.0 * fulfilled / demand_total)
                if demand_total > 0
                else np.nan,
                "total_cost": float(total_cost.sum()),
                "mean_cost_per_forecast": float(total_cost.mean()),
            }
        )
    return pd.DataFrame(rows)
