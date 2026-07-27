"""Tests for the exogenous dynamic multiplier and its supporting primitives."""

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

import impulso
from impulso._linalg import lag_matrices
from impulso._ma import compute_ma_phi
from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.results import DynamicMultiplierResult
from impulso.samplers import NUTSSampler
from impulso.spec import VAR
from impulso.volatility import Constant


class TestLagMatrices:
    """`lag_matrices` splits the stacked B into per-lag blocks, lag 1 first."""

    def test_splits_single_draw_in_lag_order(self):
        n, p = 2, 3
        B = np.arange(n * n * p, dtype=float).reshape(n, n * p)
        A = lag_matrices(B, p)
        assert len(A) == p
        for j, A_j in enumerate(A):
            assert A_j.shape == (n, n)
            np.testing.assert_array_equal(A_j, B[:, j * n : (j + 1) * n])

    def test_preserves_leading_batch_axes(self):
        c, d, n, p = 2, 5, 3, 2
        B = np.random.default_rng(0).standard_normal((c, d, n, n * p))
        A = lag_matrices(B, p)
        assert [A_j.shape for A_j in A] == [(c, d, n, n)] * p
        np.testing.assert_array_equal(A[1], B[..., n:])

    def test_single_lag_returns_whole_matrix(self):
        B = np.random.default_rng(1).standard_normal((4, 4))
        (A_1,) = lag_matrices(B, 1)
        np.testing.assert_array_equal(A_1, B)

    @pytest.mark.parametrize("n_lags", [0, -1])
    def test_rejects_non_positive_n_lags(self, n_lags):
        with pytest.raises(ValueError, match="n_lags must be positive"):
            lag_matrices(np.zeros((2, 4)), n_lags)

    def test_rejects_indivisible_trailing_axis(self):
        with pytest.raises(ValueError, match="not divisible"):
            lag_matrices(np.zeros((2, 5)), 2)

    def test_reproduces_the_idiom_it_replaced(self):
        """Guards the call sites in identified.py and identification.py."""
        rng = np.random.default_rng(7)
        n_vars, n_lags = 3, 4
        B = rng.standard_normal((2, 6, n_vars, n_vars * n_lags))
        legacy = [B[:, :, :, j * n_vars : (j + 1) * n_vars] for j in range(n_lags)]
        for new, old in zip(lag_matrices(B, n_lags), legacy, strict=True):
            np.testing.assert_array_equal(new, old)


class TestPublicApi:
    """Both MA primitives are reachable without importing a private module."""

    @pytest.mark.parametrize("name", ["compute_ma_phi", "lag_matrices", "DynamicMultiplierResult"])
    def test_exported(self, name):
        assert name in impulso.__all__
        assert getattr(impulso, name) is not None

    def test_primitives_compose(self):
        B = np.random.default_rng(2).standard_normal((3, 6))
        Phi = impulso.compute_ma_phi(impulso.lag_matrices(B, 2), horizon=3)
        assert Phi.shape == (4, 3, 3)


@pytest.fixture
def fitted_with_exog():
    """A FittedVAR carrying a synthetic VAR(2) posterior with B_exog. No MCMC."""
    rng = np.random.default_rng(11)
    n_chains, n_draws, n_vars, n_lags, n_exog, t = 2, 25, 2, 2, 2, 40
    posterior = xr.Dataset({
        "B": xr.DataArray(
            rng.standard_normal((n_chains, n_draws, n_vars, n_vars * n_lags)) * 0.2,
            dims=["chain", "draw", "var", "coeff"],
        ),
        "B_exog": xr.DataArray(
            rng.standard_normal((n_chains, n_draws, n_vars, n_exog)),
            dims=["chain", "draw", "var", "exog"],
        ),
        "intercept": xr.DataArray(
            rng.standard_normal((n_chains, n_draws, n_vars)) * 0.01,
            dims=["chain", "draw", "var"],
        ),
    })
    data = VARData(
        endog=rng.standard_normal((t, n_vars)),
        endog_names=["ad", "brand"],
        exog=rng.standard_normal((t, n_exog)),
        exog_names=["tv", "online"],
        index=pd.date_range("2023-01-02", periods=t, freq="W-MON"),
    )
    return FittedVAR(
        idata=az.InferenceData(posterior=posterior),
        n_lags=n_lags,
        data=data,
        var_names=["ad", "brand"],
        volatility=Constant(),
    )


class TestDynamicMultiplier:
    def test_equals_phi_at_b_exog(self, fitted_with_exog):
        """Psi_h = Phi_h @ B_exog, recomputed independently from the posterior."""
        horizon = 5
        psi = fitted_with_exog.dynamic_multiplier(horizon=horizon)
        post = fitted_with_exog.idata.posterior
        Phi = compute_ma_phi(lag_matrices(post["B"].values, fitted_with_exog.n_lags), horizon)
        expected = Phi @ post["B_exog"].values[:, :, np.newaxis, :, :]
        np.testing.assert_allclose(psi.idata.posterior_predictive["dynamic_multiplier"].values, expected)

    def test_impact_multiplier_is_b_exog(self, fitted_with_exog):
        """Phi_0 = I, so the horizon-0 multiplier is B_exog itself."""
        psi = fitted_with_exog.dynamic_multiplier(horizon=3)
        np.testing.assert_allclose(
            psi.idata.posterior_predictive["dynamic_multiplier"].values[:, :, 0, :, :],
            fitted_with_exog.idata.posterior["B_exog"].values,
        )

    def test_cumulative_is_cumsum_over_horizon(self, fitted_with_exog):
        impulse = fitted_with_exog.dynamic_multiplier(horizon=6)
        step = fitted_with_exog.dynamic_multiplier(horizon=6, cumulative=True)
        np.testing.assert_allclose(
            step.idata.posterior_predictive["dynamic_multiplier"].values,
            np.cumsum(impulse.idata.posterior_predictive["dynamic_multiplier"].values, axis=2),
        )
        assert step.cumulative is True
        assert impulse.cumulative is False

    def test_cumulative_matches_forced_recursion(self, fitted_with_exog):
        """A permanent unit step in one exog, propagated through the VAR by hand."""
        horizon = 4
        post = fitted_with_exog.idata.posterior
        B = post["B"].values
        B_exog = post["B_exog"].values
        A = lag_matrices(B, fitted_with_exog.n_lags)

        # Drive exog column 0 at a constant unit level from t=0 with zero history.
        hist: list[np.ndarray] = []
        states = []
        for _ in range(horizon + 1):
            s = B_exog[..., 0].copy()
            for j, A_j in enumerate(A):
                if j < len(hist):
                    s = s + np.einsum("cdik,cdk->cdi", A_j, hist[-(j + 1)])
            hist.append(s)
            states.append(s)

        step = fitted_with_exog.dynamic_multiplier(horizon=horizon, cumulative=True)
        np.testing.assert_allclose(
            step.idata.posterior_predictive["dynamic_multiplier"].values[..., 0],
            np.stack(states, axis=2),
        )

    def test_shape_dims_and_coords(self, fitted_with_exog):
        psi = fitted_with_exog.dynamic_multiplier(horizon=4)
        da = psi.idata.posterior_predictive["dynamic_multiplier"]
        assert da.dims == ("chain", "draw", "horizon", "response", "exog")
        assert da.shape == (2, 25, 5, 2, 2)
        assert da.coords["response"].values.tolist() == ["ad", "brand"]
        assert da.coords["exog"].values.tolist() == ["tv", "online"]
        assert psi.exog_names == ["tv", "online"]

    def test_horizon_zero_is_allowed(self, fitted_with_exog):
        psi = fitted_with_exog.dynamic_multiplier(horizon=0)
        assert psi.idata.posterior_predictive["dynamic_multiplier"].sizes["horizon"] == 1

    def test_rejects_negative_horizon(self, fitted_with_exog):
        with pytest.raises(ValueError, match="horizon must be non-negative"):
            fitted_with_exog.dynamic_multiplier(horizon=-1)

    def test_rejects_posterior_without_b_exog(self, synthetic_idata_2v, var_data_2v):
        """Guards on the posterior, not on has_exog."""
        fitted = FittedVAR(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        with pytest.raises(ValueError, match="no B_exog"):
            fitted.dynamic_multiplier(horizon=3)

    def test_realigns_scrambled_dim_order(self, fitted_with_exog):
        """A hand-built posterior with non-canonical dim order gives identical draws."""
        post = fitted_with_exog.idata.posterior
        scrambled = xr.Dataset({
            "B": post["B"].transpose("chain", "draw", "coeff", "var"),
            "B_exog": post["B_exog"].transpose("exog", "var", "chain", "draw"),
            "intercept": post["intercept"],
        })
        fitted = FittedVAR(
            idata=az.InferenceData(posterior=scrambled),
            n_lags=fitted_with_exog.n_lags,
            data=fitted_with_exog.data,
            var_names=fitted_with_exog.var_names,
            volatility=Constant(),
        )
        np.testing.assert_allclose(
            fitted.dynamic_multiplier(horizon=4).idata.posterior_predictive["dynamic_multiplier"].values,
            fitted_with_exog.dynamic_multiplier(horizon=4).idata.posterior_predictive["dynamic_multiplier"].values,
        )

    def test_rejects_exog_name_count_mismatch(self, fitted_with_exog):
        """A hand-built FittedVAR whose data disagrees with its posterior fails loudly."""
        rng = np.random.default_rng(3)
        t = fitted_with_exog.data.endog.shape[0]
        data = VARData(
            endog=fitted_with_exog.data.endog,
            endog_names=fitted_with_exog.data.endog_names,
            exog=rng.standard_normal((t, 3)),
            exog_names=["tv", "online", "print"],
            index=fitted_with_exog.data.index,
        )
        fitted = FittedVAR(
            idata=fitted_with_exog.idata,
            n_lags=fitted_with_exog.n_lags,
            data=data,
            var_names=fitted_with_exog.var_names,
            volatility=Constant(),
        )
        with pytest.raises(ValueError, match="carries 3 names"):
            fitted.dynamic_multiplier(horizon=2)


class TestDynamicMultiplierResult:
    def test_median_frame_labels(self, fitted_with_exog):
        med = fitted_with_exog.dynamic_multiplier(horizon=4).median()
        assert isinstance(med, pd.DataFrame)
        assert med.index.name == "horizon"
        assert med.columns.names == ["response", "exog"]
        assert ("brand", "online") in med.columns
        assert len(med) == 5

    def test_hdi_mirrors_median_shape(self, fitted_with_exog):
        result = fitted_with_exog.dynamic_multiplier(horizon=4)
        hdi = result.hdi(prob=0.8)
        assert hdi.prob == 0.8
        assert hdi.lower.shape == result.median().shape
        assert list(hdi.upper.columns) == list(result.median().columns)
        assert (hdi.lower <= hdi.upper).all().all()

    def test_to_dataframe_matches_median(self, fitted_with_exog):
        result = fitted_with_exog.dynamic_multiplier(horizon=3)
        pd.testing.assert_frame_equal(result.to_dataframe(), result.median())

    def test_is_result_type(self, fitted_with_exog):
        assert isinstance(fitted_with_exog.dynamic_multiplier(horizon=2), DynamicMultiplierResult)

    def test_plot_returns_figure(self, fitted_with_exog):
        from matplotlib.figure import Figure

        assert isinstance(fitted_with_exog.dynamic_multiplier(horizon=3).plot(), Figure)


class TestFittedModelRetention:
    """`VAR.fit` keeps the PyMC model it built."""

    @pytest.mark.slow
    def test_fit_retains_pymc_model(self, var_data_2v):
        import pymc as pm

        fitted = VAR(lags=1).fit(
            var_data_2v,
            sampler=NUTSSampler(draws=40, tune=40, chains=1, cores=1, random_seed=3, progressbar=False),
        )
        assert isinstance(fitted.pymc_model, pm.Model)
        assert {"intercept", "B"} <= {v.name for v in fitted.pymc_model.unobserved_RVs}

    def test_defaults_to_none_when_not_supplied(self, synthetic_idata_2v, var_data_2v):
        """ConjugateVAR builds no PyMC graph, so the field must stay optional."""
        fitted = FittedVAR(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        assert fitted.pymc_model is None


class TestPosteriorCoords:
    """The fitted posterior labels itself instead of using B_dim_0 / B_dim_1."""

    @pytest.mark.slow
    def test_named_dims_and_coords(self, var_data_2v):
        fitted = VAR(lags=2).fit(
            var_data_2v,
            sampler=NUTSSampler(draws=40, tune=40, chains=1, cores=1, random_seed=4, progressbar=False),
        )
        post = fitted.idata.posterior
        assert post["B"].dims == ("chain", "draw", "var", "coeff")
        assert post["intercept"].dims == ("chain", "draw", "var")
        assert post["Sigma"].dims == ("chain", "draw", "var1", "var2")
        assert post["B"].coords["var"].values.tolist() == var_data_2v.endog_names
        # coeff is lag-major, mirroring the X_lag hstack in VAR.fit
        assert post["B"].coords["coeff"].values.tolist() == [
            f"L{lag}.{name}" for lag in (1, 2) for name in var_data_2v.endog_names
        ]

    @pytest.mark.slow
    def test_label_selection_matches_positional(self, var_data_2v):
        fitted = VAR(lags=2).fit(
            var_data_2v,
            sampler=NUTSSampler(draws=40, tune=40, chains=1, cores=1, random_seed=5, progressbar=False),
        )
        post = fitted.idata.posterior
        name_1 = var_data_2v.endog_names[1]
        selected = post["B"].sel(var=name_1, coeff=f"L2.{var_data_2v.endog_names[0]}")
        np.testing.assert_allclose(selected.values, post["B"].values[:, :, 1, 2])

    @pytest.mark.slow
    def test_exog_names_reach_the_posterior(self, var_data_2v):
        rng = np.random.default_rng(6)
        t = var_data_2v.endog.shape[0]
        data = VARData(
            endog=var_data_2v.endog,
            endog_names=var_data_2v.endog_names,
            exog=rng.standard_normal((t, 2)),
            exog_names=["tv", "online"],
            index=var_data_2v.index,
        )
        fitted = VAR(lags=1).fit(
            data,
            sampler=NUTSSampler(draws=40, tune=40, chains=1, cores=1, random_seed=7, progressbar=False),
        )
        b_exog = fitted.idata.posterior["B_exog"]
        assert b_exog.dims == ("chain", "draw", "var", "exog")
        assert b_exog.coords["exog"].values.tolist() == ["tv", "online"]
