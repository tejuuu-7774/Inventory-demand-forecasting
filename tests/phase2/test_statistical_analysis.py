import pandas as pd

from src.phase2.statistical_analysis import paired_series_bootstrap


def test_paired_bootstrap_resamples_whole_series_and_is_reproducible():
    first = pd.Series({"A": 3.0, "B": 2.0, "C": 4.0})
    second = pd.Series({"A": 2.0, "B": 2.0, "C": 1.0})
    one = paired_series_bootstrap(first, second, 500, 9)
    two = paired_series_bootstrap(first, second, 500, 9)
    assert one == two
    assert one["number_of_series"] == 3
    assert one["mean_mae_difference_first_minus_second"] == 4 / 3
