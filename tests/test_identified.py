"""Tests for IdentifiedVAR."""

import numpy as np
import pytest

from impulso.identification import Cholesky
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

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )
        irf = identified.impulse_response(horizon=10)
        assert isinstance(irf, IRFResult)
        assert irf.horizon == 10
        med = irf.median()
        assert med.shape == (11, 4)  # (horizon+1, n_vars*n_vars)

    def test_fevd_shape(self, synthetic_identified_idata_2v, var_data_2v):
        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )
        fevd = identified.fevd(horizon=10)
        assert isinstance(fevd, FEVDResult)
        med = fevd.median()
        assert med.shape == (11, 4)

    def test_fevd_sums_to_one(self, synthetic_identified_idata_2v, var_data_2v):
        """FEVD shares should sum to ~1 for each response at each horizon."""

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )
        fevd = identified.fevd(horizon=10)
        fevd_da = fevd.idata.posterior_predictive["fevd"]
        med = fevd_da.median(dim=("chain", "draw"))
        # For each response, shares across shocks should sum to 1
        for resp in ["y1", "y2"]:
            sums = med.sel(response=resp).values.sum(axis=1)
            np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_historical_decomposition_shape(self, synthetic_identified_idata_2v, var_data_2v):
        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )
        hd = identified.historical_decomposition()
        assert isinstance(hd, HistoricalDecompositionResult)

    def test_irf_deterministic_values(self, synthetic_identified_idata_2v, var_data_2v):
        """IRF at horizon 0 should equal the structural shock matrix."""
        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )
        irf = identified.impulse_response(horizon=5)
        irf_draws = irf.idata.posterior_predictive["irf"].values  # (C, D, H+1, n, n)

        P = synthetic_identified_idata_2v.posterior["structural_shock_matrix"].values
        # At h=0, IRF = Phi_0 @ P = I @ P = P
        np.testing.assert_allclose(irf_draws[:, :, 0, :, :], P, atol=1e-12)

    def test_repr(self, synthetic_identified_idata_2v, var_data_2v):
        identified = IdentifiedVAR.model_construct(
            idata=synthetic_identified_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )
        r = repr(identified)
        assert "IdentifiedVAR" in r
