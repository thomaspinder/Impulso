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


class TestIdentifiedVarCarriesVolatilityAndScheme:
    @pytest.mark.slow
    def test_carries_volatility_and_scheme(self, var_data_2v):
        from impulso.identification import Cholesky
        from impulso.protocols import IdentificationScheme, VolatilityProcess
        from impulso.samplers import NUTSSampler
        from impulso.spec import VAR

        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)
        scheme = Cholesky(ordering=fitted.var_names)
        identified = fitted.set_identification_strategy(scheme)

        assert isinstance(identified.volatility, VolatilityProcess)
        assert isinstance(identified.scheme, IdentificationScheme)
        assert identified.scheme is scheme


class TestImpulseResponseAt:
    @pytest.mark.slow
    def test_at_ignored_for_constant(self, var_data_2v):
        """For Constant volatility, at= must be a no-op."""
        from impulso.identification import Cholesky
        from impulso.samplers import NUTSSampler
        from impulso.spec import VAR

        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)
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

    @pytest.mark.slow
    def test_at_all_returns_time_axis(self, var_data_2v):
        """at='all' adds a time dim of correct length to the result."""
        from impulso.identification import Cholesky
        from impulso.samplers import NUTSSampler
        from impulso.spec import VAR

        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        horizon = 5
        irf_all = identified.impulse_response(horizon=horizon, at="all")
        irf = irf_all.idata.posterior_predictive["irf"]
        assert "time" in irf.dims
        T_eff = var_data_2v.endog.shape[0] - 1  # lags=1
        n_vars = len(fitted.var_names)
        assert irf.sizes["time"] == T_eff
        assert irf.shape == (1, 20, T_eff, horizon + 1, n_vars, n_vars)
        # For Constant, every time slice must be identical.
        np.testing.assert_array_equal(irf.values[:, :, 0, :, :, :], irf.values[:, :, -1, :, :, :])


class TestFEVDAt:
    @pytest.mark.slow
    def test_at_ignored_for_constant(self, var_data_2v):
        from impulso.identification import Cholesky
        from impulso.samplers import NUTSSampler
        from impulso.spec import VAR

        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        fevd_default = identified.fevd(horizon=5)
        fevd_at_int = identified.fevd(horizon=5, at=3)
        np.testing.assert_array_equal(
            fevd_default.idata.posterior_predictive["fevd"].values,
            fevd_at_int.idata.posterior_predictive["fevd"].values,
        )

    @pytest.mark.slow
    def test_at_all_returns_time_axis(self, var_data_2v):
        """at='all' adds a time dim of correct length to the result."""
        from impulso.identification import Cholesky
        from impulso.samplers import NUTSSampler
        from impulso.spec import VAR

        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        horizon = 5
        fevd_all = identified.fevd(horizon=horizon, at="all")
        fevd = fevd_all.idata.posterior_predictive["fevd"]
        assert "time" in fevd.dims
        T_eff = var_data_2v.endog.shape[0] - 1  # lags=1
        n_vars = len(fitted.var_names)
        assert fevd.sizes["time"] == T_eff
        assert fevd.shape == (1, 20, T_eff, horizon + 1, n_vars, n_vars)
        # For Constant, every time slice must be identical.
        np.testing.assert_array_equal(fevd.values[:, :, 0, :, :, :], fevd.values[:, :, -1, :, :, :])

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
