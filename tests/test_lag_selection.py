"""Tests for lag order selection."""

import numpy as np
import pandas as pd
import pytest

from impulso._lag_selection import select_lag_order
from impulso.data import VARData
from impulso.results import LagOrderResult


@pytest.fixture
def var_data():
    """Simple VAR(2) DGP for testing lag selection."""
    rng = np.random.default_rng(42)
    T = 200
    n = 3
    y = np.zeros((T, n))
    for t in range(2, T):
        y[t] = 0.5 * y[t - 1] + 0.2 * y[t - 2] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2", "y3"], index=index)


class TestSelectLagOrder:
    def test_returns_lag_order_result(self, var_data):
        result = select_lag_order(var_data, max_lags=8)
        assert isinstance(result, LagOrderResult)

    def test_aic_bic_hq_are_positive_ints(self, var_data):
        result = select_lag_order(var_data, max_lags=8)
        assert result.aic >= 1
        assert result.bic >= 1
        assert result.hq >= 1

    def test_summary_has_expected_columns(self, var_data):
        result = select_lag_order(var_data, max_lags=8)
        summary = result.summary()
        assert "aic" in summary.columns
        assert "bic" in summary.columns
        assert "hq" in summary.columns

    def test_summary_rows_match_max_lags(self, var_data):
        result = select_lag_order(var_data, max_lags=6)
        assert len(result.summary()) == 6
