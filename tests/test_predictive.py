"""Tests for the prior- and posterior-predictive APIs (issue #56)."""

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from impulso import VAR, VARData
from impulso._lag_selection import select_lag_order


@pytest.fixture
def var_data_2v_exog(var_data_2v):
    """`var_data_2v` plus a single deterministic exogenous column."""
    T = var_data_2v.endog.shape[0]
    exog = np.linspace(-1.0, 1.0, T).reshape(T, 1)
    return VARData(
        endog=var_data_2v.endog,
        endog_names=list(var_data_2v.endog_names),
        exog=exog,
        exog_names=["x1"],
        index=var_data_2v.index,
    )


class TestPriorPredictive:
    """`VAR.prior_predictive` draws from the same graph `fit` samples."""

    def test_groups_and_variable(self, var_data_2v):
        idata = VAR(lags=1).prior_predictive(var_data_2v, draws=50, random_seed=0)

        assert {"prior", "prior_predictive", "observed_data"} <= set(idata.groups())
        assert "obs" in idata.prior_predictive
        assert "obs" in idata.observed_data
        # The latents are simulated too, so the prior itself is inspectable.
        assert {"intercept", "B", "sigma_sd"} <= set(idata.prior.data_vars)

    def test_shape_dims_and_coords(self, var_data_2v):
        idata = VAR(lags=1).prior_predictive(var_data_2v, draws=50, random_seed=0)

        obs = idata.prior_predictive["obs"]
        # PyMC's prior predictive is a single chain.
        assert obs.shape == (1, 50, 199, 2)
        assert obs.dims == ("chain", "draw", "time", "var")
        assert list(obs.coords["var"].values) == ["y1", "y2"]
        assert np.array_equal(
            pd.to_datetime(obs.coords["time"].values),
            var_data_2v.index[1:],
        )
        assert idata.observed_data["obs"].dims == ("time", "var")
        assert np.array_equal(idata.observed_data["obs"].values, var_data_2v.endog[1:])

    def test_random_seed_is_reproducible(self, var_data_2v):
        spec = VAR(lags=1)
        a = spec.prior_predictive(var_data_2v, draws=20, random_seed=0)
        b = spec.prior_predictive(var_data_2v, draws=20, random_seed=0)
        c = spec.prior_predictive(var_data_2v, draws=20, random_seed=1)

        assert np.array_equal(a.prior_predictive["obs"].values, b.prior_predictive["obs"].values)
        assert not np.array_equal(a.prior_predictive["obs"].values, c.prior_predictive["obs"].values)

    def test_lag_order_trims_the_time_axis(self, var_data_2v):
        idata = VAR(lags=3).prior_predictive(var_data_2v, draws=10, random_seed=0)

        assert idata.prior_predictive["obs"].shape[2] == 197

    def test_criterion_lags_resolve_like_fit(self, var_data_2v):
        idata = VAR(lags="bic").prior_predictive(var_data_2v, draws=10, random_seed=0)

        expected_lags = select_lag_order(var_data_2v, max_lags=12).bic
        assert idata.prior_predictive["obs"].shape[2] == 200 - expected_lags

    def test_exogenous_regressors_are_simulated(self, var_data_2v_exog):
        idata = VAR(lags=1).prior_predictive(var_data_2v_exog, draws=10, random_seed=0)

        assert "B_exog" in idata.prior
        assert idata.prior["B_exog"].shape == (1, 10, 2, 1)

    def test_observed_series_falls_within_the_prior_band(self, var_data_2v):
        """Issue AC: the observed series must sit inside the 95% prior band.

        Quantiles, never mean +/- k*sd: the HalfCauchy scale prior has no
        finite moments, so a moment-based band is meaningless here.
        """
        idata = VAR(lags=1).prior_predictive(var_data_2v, draws=500, random_seed=0)

        draws = idata.prior_predictive["obs"].values[0]  # (draws, time, var)
        lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
        observed = var_data_2v.endog[1:]
        covered = (observed >= lower) & (observed <= upper)
        assert covered.mean() >= 0.9

    def test_plot_ppc_smoke(self, var_data_2v):
        """`az.plot_ppc` must accept the group as returned (datetime time coord)."""
        idata = VAR(lags=1).prior_predictive(var_data_2v, draws=20, random_seed=0)

        axes = az.plot_ppc(idata, group="prior", num_pp_samples=5)
        assert axes is not None
