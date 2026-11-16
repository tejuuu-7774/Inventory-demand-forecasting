"""Leak-free Kalman state features for the DeepAR hybrid."""

import numpy as np
import pandas as pd


def fit_local_level(values, observation_mask=None):
    """Fit the same simple local-level idea used in Phase 1.

    Missing/unavailable observations may be skipped with observation_mask.
    The returned state is the filtered state after the final usable observation.
    """

    values = np.asarray(values, dtype=float)
    if observation_mask is None:
        observation_mask = np.ones(len(values), dtype=bool)
    observation_mask = np.asarray(observation_mask, dtype=bool)
    usable = values[observation_mask]

    if len(usable) == 0:
        return 0.0, 1.0, 0.01, 0.25

    difference_variance = np.var(np.diff(usable), ddof=1) if len(usable) > 2 else 1.0
    q = max(0.01, 0.05 * float(difference_variance))
    r = max(0.25, 0.50 * float(difference_variance))
    level = float(usable[0])
    variance = max(r, 1.0)

    for observation, use_observation in zip(values, observation_mask):
        variance = variance + q
        if not use_observation:
            continue
        gain = variance / (variance + r)
        level = level + gain * (observation - level)
        variance = (1.0 - gain) * variance

    return level, variance, q, r


def _causal_features_for_one_series(series):
    """Calculate each day's state and scale before that day's sale is observed.

    The scale is the expanding mean of usable observations strictly before the
    current timestamp. A lower bound of one keeps intermittent and all-zero
    series numerically stable. Future targets therefore cannot alter earlier
    normalized state features.
    """

    series = series.sort_values("date").copy()
    sales = series["sales"].to_numpy(dtype=float)
    masks = series["observation_mask"].to_numpy(dtype=bool)

    # Start without seeing the first target. The first row must be a genuine
    # prior estimate, not an estimate initialized from that same day's sale.
    level = 0.0
    variance = 1.0
    q = 0.01
    r = 0.25
    previous_observation = None
    difference_count = 0
    difference_mean = 0.0
    difference_m2 = 0.0
    previous_innovation = 0.0

    scaled_levels = []
    scaled_variances = []
    scaled_innovations = []
    causal_scales = []
    observed_sum = 0.0
    observed_count = 0

    for observation, use_observation in zip(sales, masks):
        variance = variance + q
        scale = max(1.0, observed_sum / observed_count) if observed_count else 1.0
        causal_scales.append(scale)
        scaled_levels.append(level / scale)
        scaled_variances.append(variance / (scale * scale))
        scaled_innovations.append(previous_innovation / scale)

        if not use_observation:
            previous_innovation = 0.0
            continue

        innovation = observation - level
        gain = variance / (variance + r)
        level = level + gain * innovation
        variance = (1.0 - gain) * variance
        previous_innovation = innovation

        if previous_observation is not None:
            difference = observation - previous_observation
            difference_count += 1
            delta = difference - difference_mean
            difference_mean += delta / difference_count
            difference_m2 += delta * (difference - difference_mean)
            if difference_count > 1:
                difference_variance = difference_m2 / (difference_count - 1)
                q = max(0.01, 0.05 * difference_variance)
                r = max(0.25, 0.50 * difference_variance)
        previous_observation = observation
        observed_sum += float(observation)
        observed_count += 1

    series["kalman_scale_past_only"] = np.asarray(causal_scales)
    series["kalman_level_scaled"] = np.asarray(scaled_levels)
    series["kalman_variance_scaled"] = np.asarray(scaled_variances)
    series["kalman_innovation_scaled"] = np.asarray(scaled_innovations)
    return series


def add_causal_kalman_features(data):
    """Add Kalman features without allowing the current target into its inputs."""

    parts = []
    for _, series in data.groupby("id", sort=False):
        parts.append(_causal_features_for_one_series(series))
    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(["id", "date"]).reset_index(drop=True)
    return result


def projected_kalman_features(history, number_of_steps):
    """Project a filtered state forward without using future actual sales."""

    level, variance, q, _ = fit_local_level(
        history["sales"].to_numpy(),
        history["observation_mask"].to_numpy(),
    )
    observed = history.loc[history["observation_mask"].eq(1), "sales"]
    scale = max(1.0, float(observed.mean()) if len(observed) else 1.0)

    rows = []
    for step in range(1, number_of_steps + 1):
        rows.append(
            {
                "kalman_level_scaled": level / scale,
                "kalman_variance_scaled": (variance + step * q) / (scale * scale),
                "kalman_innovation_scaled": 0.0,
            }
        )
    return pd.DataFrame(rows)
