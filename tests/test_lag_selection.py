"""Tests for lag order selection."""

import pytest

from impulso._lag_selection import select_lag_order
from impulso.results import LagOrderResult


@pytest.fixture
def var_data(var_data_3v_dgp2):
    return var_data_3v_dgp2


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
