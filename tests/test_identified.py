"""Tests for IdentifiedVAR."""

import arviz as az
import numpy as np
import pytest
import xarray as xr

from impulso.identification import Cholesky, SignRestriction
from impulso.identified import IdentifiedVAR
from impulso.results import FEVDResult, HistoricalDecompositionResult, IRFResult
from impulso.samplers import NUTSSampler
from impulso.spec import VAR


@pytest.fixture
def fitted_var(var_data_2v):
    """Fit a small VAR for testing."""
    spec = VAR(lags=1, prior="minnesota")
    sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
    return spec.fit(var_data_2v, sampler=sampler)


class TestIdentifiedVAR:
    @pytest.mark.slow
    def test_set_identification_returns_identified(self, fitted_var):
        identified = fitted_var.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        assert isinstance(identified, IdentifiedVAR)

    @pytest.mark.slow
    def test_impulse_response(self, fitted_var):
        identified = fitted_var.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        irf = identified.impulse_response(horizon=10)
        assert isinstance(irf, IRFResult)
        assert irf.horizon == 10

    @pytest.mark.slow
    def test_fevd(self, fitted_var):
        identified = fitted_var.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        fevd = identified.fevd(horizon=10)
        assert isinstance(fevd, FEVDResult)

    @pytest.mark.slow
    def test_historical_decomposition(self, fitted_var):
        identified = fitted_var.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        hd = identified.historical_decomposition()
        assert isinstance(hd, HistoricalDecompositionResult)


class TestIdentifiedVARFast:
    """Fast tests using synthetic InferenceData (no MCMC)."""

    def test_impulse_response_shape(self, synthetic_identified_idata_2v, var_data_2v):
        from impulso.volatility import Constant

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        irf = identified.impulse_response(horizon=10)
        assert isinstance(irf, IRFResult)
        assert irf.horizon == 10
        med = irf.median()
        assert med.shape == (11, 4)  # (horizon+1, n_vars*n_vars)

    def test_fevd_shape(self, synthetic_identified_idata_2v, var_data_2v):
        from impulso.volatility import Constant

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        fevd = identified.fevd(horizon=10)
        assert isinstance(fevd, FEVDResult)
        med = fevd.median()
        assert med.shape == (11, 4)

    def test_fevd_sums_to_one(self, synthetic_identified_idata_2v, var_data_2v):
        """FEVD shares should sum to ~1 for each response at each horizon."""
        from impulso.volatility import Constant

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        fevd = identified.fevd(horizon=10)
        fevd_da = fevd.idata.posterior_predictive["fevd"]
        med = fevd_da.median(dim=("chain", "draw"))
        # For each response, shares across shocks should sum to 1
        for resp in ["y1", "y2"]:
            sums = med.sel(response=resp).values.sum(axis=1)
            np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_historical_decomposition_shape(self, synthetic_identified_idata_2v, var_data_2v):
        from impulso.volatility import Constant

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        hd = identified.historical_decomposition()
        assert isinstance(hd, HistoricalDecompositionResult)

    def test_irf_deterministic_values(self, synthetic_identified_idata_2v, var_data_2v):
        """IRF at horizon 0 should equal the structural shock matrix."""
        from impulso.volatility import Constant

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        irf = identified.impulse_response(horizon=5)
        irf_draws = irf.idata.posterior_predictive["irf"].values  # (C, D, H+1, n, n)

        P = synthetic_identified_idata_2v.posterior["structural_shock_matrix"].values
        # At h=0, IRF = Phi_0 @ P = I @ P = P
        np.testing.assert_allclose(irf_draws[:, :, 0, :, :], P, atol=1e-12)

    def test_irf_propagates_custom_shock_coords(self, synthetic_idata_2v, var_data_2v):
        """IRF shock coordinates should match structural_shock_matrix, not var_names."""
        from impulso.volatility import Constant

        sigma = synthetic_idata_2v.posterior["Sigma"].values
        P = np.linalg.cholesky(sigma)
        P_da = xr.DataArray(
            P,
            dims=["chain", "draw", "response", "shock"],
            coords={"response": ["y1", "y2"], "shock": ["my_shock", "unidentified_1"]},
        )
        idata = az.InferenceData(posterior=synthetic_idata_2v.posterior.assign(structural_shock_matrix=P_da))
        identified = IdentifiedVAR.model_construct(
            idata=idata,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        irf = identified.impulse_response(horizon=5)
        irf_shocks = list(irf.idata.posterior_predictive["irf"].coords["shock"].values)
        assert irf_shocks == ["my_shock", "unidentified_1"]

    def test_repr(self, synthetic_identified_idata_2v, var_data_2v):
        from impulso.volatility import Constant

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        r = repr(identified)
        assert "IdentifiedVAR" in r


class TestP2PosteriorEquivalence:
    """P2 must not change identified posteriors. Fits VAR(lags=1) +
    Cholesky + SignRestriction under the new pipeline and asserts the
    structural_shock_matrix posterior matches the mathematically expected
    value (Cholesky) or is well-formed (SignRestriction smoke test).
    """

    @pytest.mark.slow
    def test_cholesky_identified_posterior_unchanged(self, var_data_2v):
        """P2 invariant: Cholesky-identified posterior equals np.linalg.cholesky(Sigma) at 1e-10."""
        sampler = NUTSSampler(cores=1, chains=2, draws=200, tune=200, random_seed=42, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        P = identified.idata.posterior["structural_shock_matrix"].values
        assert P.shape == (2, 200, 2, 2)

        # Cholesky with identity ordering: P should equal np.linalg.cholesky(Sigma).
        sigma = fitted.idata.posterior["Sigma"].values
        np.testing.assert_allclose(P, np.linalg.cholesky(sigma), rtol=1e-10)

    @pytest.mark.slow
    def test_sign_restriction_identified_posterior_runs(self, var_data_2v):
        """Smoke test: SignRestriction with restriction_horizon=2 produces
        a valid structural_shock_matrix posterior."""
        from impulso.samplers import NUTSSampler
        from impulso.spec import VAR

        sampler = NUTSSampler(cores=1, chains=2, draws=100, tune=100, random_seed=42, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)
        scheme = SignRestriction(
            restrictions={fitted.var_names[0]: {"shock_a": "+"}},
            n_rotations=50,
            restriction_horizon=2,
            random_seed=42,
        )
        identified = fitted.set_identification_strategy(scheme)

        P = identified.idata.posterior["structural_shock_matrix"].values
        assert P.shape == (2, 100, 2, 2)
        assert not np.isnan(P).any(), "Some draws produced NaN structural shock matrix"


def _fitted_from_synthetic(synthetic_idata_2v, var_data_2v):
    """Build a FittedVAR from synthetic InferenceData (no MCMC)."""
    from impulso.fitted import FittedVAR
    from impulso.volatility import Constant

    return FittedVAR.model_construct(
        idata=synthetic_idata_2v,
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )


class TestIdentifiedVarCarriesVolatilityAndScheme:
    def test_carries_volatility_and_scheme(self, synthetic_idata_2v, var_data_2v):
        from impulso.protocols import IdentificationScheme, VolatilityProcess

        fitted = _fitted_from_synthetic(synthetic_idata_2v, var_data_2v)
        scheme = Cholesky(ordering=fitted.var_names)
        identified = fitted.set_identification_strategy(scheme)

        assert isinstance(identified.volatility, VolatilityProcess)
        assert isinstance(identified.scheme, IdentificationScheme)
        assert identified.scheme is scheme


class TestImpulseResponseAt:
    def test_at_ignored_for_constant(self, synthetic_idata_2v, var_data_2v):
        """For Constant volatility, at= must be a no-op."""
        fitted = _fitted_from_synthetic(synthetic_idata_2v, var_data_2v)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        irf_default = identified.impulse_response(horizon=5)
        irf_at_last = identified.impulse_response(horizon=5, at="last")
        irf_at_int = identified.impulse_response(horizon=5, at=10)
        irf_at_none = identified.impulse_response(horizon=5, at=None)

        np.testing.assert_array_equal(
            irf_default.idata.posterior_predictive["irf"].values,
            irf_at_last.idata.posterior_predictive["irf"].values,
        )
        np.testing.assert_array_equal(
            irf_default.idata.posterior_predictive["irf"].values,
            irf_at_int.idata.posterior_predictive["irf"].values,
        )
        np.testing.assert_array_equal(
            irf_default.idata.posterior_predictive["irf"].values,
            irf_at_none.idata.posterior_predictive["irf"].values,
        )

    def test_at_all_returns_time_axis(self, synthetic_idata_2v, var_data_2v):
        """at='all' adds a time dim of correct length to the result."""
        fitted = _fitted_from_synthetic(synthetic_idata_2v, var_data_2v)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        horizon = 5
        irf_all = identified.impulse_response(horizon=horizon, at="all")
        irf = irf_all.idata.posterior_predictive["irf"]
        assert "time" in irf.dims
        T_eff = var_data_2v.endog.shape[0] - 1  # lags=1
        n_chains = fitted.idata.posterior.sizes["chain"]
        n_draws = fitted.idata.posterior.sizes["draw"]
        n_vars = len(fitted.var_names)
        assert irf.sizes["time"] == T_eff
        assert irf.shape == (n_chains, n_draws, T_eff, horizon + 1, n_vars, n_vars)
        # For Constant, every time slice must be identical.
        np.testing.assert_array_equal(irf.values[:, :, 0, :, :, :], irf.values[:, :, -1, :, :, :])

    def test_at_all_with_named_index_does_not_collide(self, synthetic_idata_2v):
        """Regression: VARData built from a DataFrame whose index has a name
        (e.g. 'date') used to crash at='all' because xarray inferred the
        coord's dim from the DatetimeIndex name, conflicting with the
        declared 'time' dim."""
        import pandas as pd

        from impulso.data import VARData

        T, n = 30, 2
        rng = np.random.default_rng(0)
        y = rng.standard_normal((T, n)) * 0.1
        named_index = pd.date_range("2000-01-01", periods=T, freq="QS", name="date")
        var_data_named = VARData(endog=y, endog_names=["y1", "y2"], index=named_index)

        fitted = _fitted_from_synthetic(synthetic_idata_2v, var_data_named)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        irf_all = identified.impulse_response(horizon=3, at="all")
        irf = irf_all.idata.posterior_predictive["irf"]
        assert "time" in irf.dims
        assert "date" not in irf.dims


class TestFEVDAt:
    def test_at_ignored_for_constant(self, synthetic_idata_2v, var_data_2v):
        fitted = _fitted_from_synthetic(synthetic_idata_2v, var_data_2v)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        fevd_default = identified.fevd(horizon=5)
        fevd_at_int = identified.fevd(horizon=5, at=3)
        np.testing.assert_array_equal(
            fevd_default.idata.posterior_predictive["fevd"].values,
            fevd_at_int.idata.posterior_predictive["fevd"].values,
        )

    def test_at_all_returns_time_axis(self, synthetic_idata_2v, var_data_2v):
        """at='all' adds a time dim of correct length to the result."""
        fitted = _fitted_from_synthetic(synthetic_idata_2v, var_data_2v)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        horizon = 5
        fevd_all = identified.fevd(horizon=horizon, at="all")
        fevd = fevd_all.idata.posterior_predictive["fevd"]
        assert "time" in fevd.dims
        T_eff = var_data_2v.endog.shape[0] - 1  # lags=1
        n_chains = fitted.idata.posterior.sizes["chain"]
        n_draws = fitted.idata.posterior.sizes["draw"]
        n_vars = len(fitted.var_names)
        assert fevd.sizes["time"] == T_eff
        assert fevd.shape == (n_chains, n_draws, T_eff, horizon + 1, n_vars, n_vars)
        # For Constant, every time slice must be identical.
        np.testing.assert_array_equal(fevd.values[:, :, 0, :, :, :], fevd.values[:, :, -1, :, :, :])

    def test_at_all_with_named_index_does_not_collide(self, synthetic_idata_2v):
        """Regression: same as the IRF test above — named DatetimeIndex used to
        crash at='all' for FEVD as well."""
        import pandas as pd

        from impulso.data import VARData

        T, n = 30, 2
        rng = np.random.default_rng(0)
        y = rng.standard_normal((T, n)) * 0.1
        named_index = pd.date_range("2000-01-01", periods=T, freq="QS", name="date")
        var_data_named = VARData(endog=y, endog_names=["y1", "y2"], index=named_index)

        fitted = _fitted_from_synthetic(synthetic_idata_2v, var_data_named)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        fevd_all = identified.fevd(horizon=3, at="all")
        fevd = fevd_all.idata.posterior_predictive["fevd"]
        assert "time" in fevd.dims
        assert "date" not in fevd.dims

    def test_median_raises_for_at_all(self, synthetic_identified_idata_2v, var_data_2v):
        """median()/hdi()/to_dataframe() must refuse FEVDs with a time dim."""
        from impulso.volatility import Constant

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        fevd_all = identified.fevd(horizon=5, at="all")
        with pytest.raises(NotImplementedError, match="time-varying FEVDs"):
            fevd_all.median()
        with pytest.raises(NotImplementedError, match="time-varying FEVDs"):
            fevd_all.hdi()
        with pytest.raises(NotImplementedError, match="time-varying FEVDs"):
            fevd_all.to_dataframe()


class TestHistoricalDecompositionAt:
    """Task 11: ``at=`` parameter on historical_decomposition.

    HD is intrinsically time-indexed, so ``at=None`` / ``at="all"`` both
    use the per-t identification path. For Constant volatility every
    L_t is identical, so all ``at=`` modes produce identical results.
    """

    def test_at_default_matches_at_all_for_constant(self, synthetic_identified_idata_2v, var_data_2v):
        """For Constant volatility, ``at=None`` and ``at='all'`` are identical."""
        from impulso.volatility import Constant

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        hd_default = identified.historical_decomposition()
        hd_all = identified.historical_decomposition(at="all")
        np.testing.assert_array_equal(
            hd_default.idata.posterior_predictive["hd"].values,
            hd_all.idata.posterior_predictive["hd"].values,
        )

    def test_at_int_matches_default_for_constant(self, synthetic_identified_idata_2v, var_data_2v):
        """For Constant volatility, ``at=int`` and ``at='last'`` match the default."""
        from impulso.volatility import Constant

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        hd_default = identified.historical_decomposition()
        hd_at_int = identified.historical_decomposition(at=5)
        hd_at_last = identified.historical_decomposition(at="last")
        np.testing.assert_array_equal(
            hd_default.idata.posterior_predictive["hd"].values,
            hd_at_int.idata.posterior_predictive["hd"].values,
        )
        np.testing.assert_array_equal(
            hd_default.idata.posterior_predictive["hd"].values,
            hd_at_last.idata.posterior_predictive["hd"].values,
        )

    def test_at_default_matches_legacy_structural_shock_matrix(self, synthetic_identified_idata_2v, var_data_2v):
        """For Constant, per-t HD must equal the legacy single-P decomposition.

        Recomputes HD by hand using the stored ``structural_shock_matrix`` (the
        P2 single-L identification) and checks the byte-for-byte equivalence
        gate that protects the Constant-volatility code path.
        """
        from impulso.volatility import Constant

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        hd = identified.historical_decomposition()
        hd_vals = hd.idata.posterior_predictive["hd"].values

        # Recompute the legacy way.
        B = synthetic_identified_idata_2v.posterior["B"].values
        P = synthetic_identified_idata_2v.posterior["structural_shock_matrix"].values
        intercept = synthetic_identified_idata_2v.posterior["intercept"].values
        y = var_data_2v.endog
        T = y.shape[0]
        n_lags = 1
        x_lag = np.concatenate([y[n_lags - lag : T - lag] for lag in range(1, n_lags + 1)], axis=1)
        y_hat = intercept[:, :, np.newaxis, :] + np.einsum("cdij,tj->cdti", B, x_lag)
        resid = y[n_lags:][np.newaxis, np.newaxis, :, :] - y_hat
        P_inv = np.linalg.inv(P)
        s = np.einsum("cdij,cdtj->cdti", P_inv, resid)
        hd_legacy = P[:, :, np.newaxis, :, :] * s[:, :, :, np.newaxis, :]
        np.testing.assert_allclose(hd_vals, hd_legacy, atol=1e-12)

    def test_at_preserves_time_dim(self, synthetic_identified_idata_2v, var_data_2v):
        """HD always carries a time dim; ``at=`` must not change its length."""
        from impulso.volatility import Constant

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        hd = identified.historical_decomposition(at="all")
        hd_da = hd.idata.posterior_predictive["hd"]
        assert "time" in hd_da.dims
        T_eff = var_data_2v.endog.shape[0] - 1
        assert hd_da.sizes["time"] == T_eff

    def test_at_int_warns_for_sv(self, var_data_2v, synthetic_idata_2v):
        """Under SV, ``at=int`` is a non-standard hypothetical and warns."""
        import xarray as xr_

        from impulso.volatility import Constant

        # Build a fake SV-flavoured IdentifiedVAR by overriding ``name`` via a
        # minimal stub that satisfies the VolatilityProcess Protocol and
        # delegates to Constant for the actual computation. This is the
        # cheapest way to exercise the warning branch without fitting an SV
        # model (sv/ adapters require their own MCMC fixtures).
        constant = Constant()

        class _FakeSV:
            name = "sv"

            def build_pymc_latent(self, n_vars, T):  # pragma: no cover
                raise NotImplementedError

            def cholesky_at(self, posterior, t):
                return constant.cholesky_at(posterior, t=t)

            def forecast_cholesky_path(self, posterior, steps, rng):  # pragma: no cover
                return constant.forecast_cholesky_path(posterior, steps=steps, rng=rng)

            def cholesky_path(self, posterior, T):
                return constant.cholesky_path(posterior, T=T)

        # Wire P into the synthetic idata as the structural_shock_matrix.
        sigma = synthetic_idata_2v.posterior["Sigma"].values
        P = np.linalg.cholesky(sigma)
        P_da = xr_.DataArray(
            P,
            dims=["chain", "draw", "response", "shock"],
            coords={"response": ["y1", "y2"], "shock": ["y1", "y2"]},
        )
        idata = az.InferenceData(posterior=synthetic_idata_2v.posterior.assign(structural_shock_matrix=P_da))

        identified = IdentifiedVAR.model_construct(
            idata=idata,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=_FakeSV(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        with pytest.warns(UserWarning, match="non-standard hypothetical"):
            identified.historical_decomposition(at=3)
        with pytest.warns(UserWarning, match="non-standard hypothetical"):
            identified.historical_decomposition(at="last")

    def test_at_default_does_not_warn_for_sv(self, var_data_2v, synthetic_idata_2v):
        """Default (per-t) HD is the standard SV decomposition — no warning."""
        import warnings as _warnings

        import xarray as xr_

        from impulso.volatility import Constant

        constant = Constant()

        class _FakeSV:
            name = "sv"

            def build_pymc_latent(self, n_vars, T):  # pragma: no cover
                raise NotImplementedError

            def cholesky_at(self, posterior, t):
                return constant.cholesky_at(posterior, t=t)

            def forecast_cholesky_path(self, posterior, steps, rng):  # pragma: no cover
                return constant.forecast_cholesky_path(posterior, steps=steps, rng=rng)

            def cholesky_path(self, posterior, T):
                return constant.cholesky_path(posterior, T=T)

        sigma = synthetic_idata_2v.posterior["Sigma"].values
        P = np.linalg.cholesky(sigma)
        P_da = xr_.DataArray(
            P,
            dims=["chain", "draw", "response", "shock"],
            coords={"response": ["y1", "y2"], "shock": ["y1", "y2"]},
        )
        idata = az.InferenceData(posterior=synthetic_idata_2v.posterior.assign(structural_shock_matrix=P_da))

        identified = IdentifiedVAR.model_construct(
            idata=idata,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=_FakeSV(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")  # any warning becomes an error
            identified.historical_decomposition()
            identified.historical_decomposition(at=None)
            identified.historical_decomposition(at="all")
