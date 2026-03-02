"""Tests for IdentifiedVAR."""

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
