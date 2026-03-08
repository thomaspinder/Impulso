"""Tests for dummy observation priors on VARData."""

import numpy as np
import pandas as pd
import pytest

from impulso.data import VARData


@pytest.fixture
def var_data():
    rng = np.random.default_rng(42)
    T, n = 100, 3
    endog = rng.standard_normal((T, n))
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=endog, endog_names=["gdp", "inflation", "rate"], index=index)


class TestDummyObservationPriors:
    def test_sum_of_coefficients_appends_n_vars_rows(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, mu=5.0)
        assert augmented.endog.shape[0] == var_data.endog.shape[0] + 3

    def test_single_unit_root_appends_one_row(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, delta=1.0)
        assert augmented.endog.shape[0] == var_data.endog.shape[0] + 1

    def test_both_dummies_append_correct_rows(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, mu=5.0, delta=1.0)
        assert augmented.endog.shape[0] == var_data.endog.shape[0] + 4

    def test_sum_of_coefficients_values(self, var_data):
        mu = 5.0
        augmented = var_data.with_dummy_observations(n_lags=4, mu=mu)
        y_bar = var_data.endog.mean(axis=0)
        dummy_rows = augmented.endog[var_data.endog.shape[0] :]
        for i in range(3):
            expected = np.zeros(3)
            expected[i] = y_bar[i] / mu
            np.testing.assert_allclose(dummy_rows[i], expected)

    def test_single_unit_root_values(self, var_data):
        delta = 1.0
        augmented = var_data.with_dummy_observations(n_lags=4, delta=delta)
        y_bar = var_data.endog.mean(axis=0)
        dummy_row = augmented.endog[-1]
        np.testing.assert_allclose(dummy_row, y_bar / delta)

    def test_preserves_original_data(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, mu=5.0)
        np.testing.assert_array_equal(augmented.endog[: var_data.endog.shape[0]], var_data.endog)

    def test_returns_new_vardata(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, mu=5.0)
        assert augmented is not var_data
        assert isinstance(augmented, VARData)

    def test_index_extended(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, mu=5.0)
        assert len(augmented.index) == augmented.endog.shape[0]

    def test_endog_names_preserved(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, mu=5.0)
        assert augmented.endog_names == var_data.endog_names

    def test_raises_if_neither_mu_nor_delta(self, var_data):
        with pytest.raises(ValueError, match="At least one"):
            var_data.with_dummy_observations(n_lags=4)

    def test_raises_if_mu_not_positive(self, var_data):
        with pytest.raises(ValueError, match="mu must be"):
            var_data.with_dummy_observations(n_lags=4, mu=-1.0)

    def test_raises_if_delta_not_positive(self, var_data):
        with pytest.raises(ValueError, match="delta must be"):
            var_data.with_dummy_observations(n_lags=4, delta=0.0)

    def test_raises_if_n_lags_not_positive(self, var_data):
        with pytest.raises(ValueError, match="n_lags must be"):
            var_data.with_dummy_observations(n_lags=0, mu=5.0)

    def test_exog_zero_padded(self):
        rng = np.random.default_rng(42)
        T, n = 100, 2
        endog = rng.standard_normal((T, n))
        exog = rng.standard_normal((T, 1))
        index = pd.date_range("2000-01-01", periods=T, freq="QS")
        data = VARData(endog=endog, endog_names=["y1", "y2"], exog=exog, exog_names=["x1"], index=index)
        augmented = data.with_dummy_observations(n_lags=2, mu=5.0)
        assert augmented.exog.shape[0] == augmented.endog.shape[0]
        np.testing.assert_array_equal(augmented.exog[:T], data.exog)
        np.testing.assert_array_equal(augmented.exog[T:], 0.0)
        assert augmented.exog_names == ["x1"]
