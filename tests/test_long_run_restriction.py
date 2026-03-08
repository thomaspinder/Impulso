"""Tests for Blanchard-Quah long-run identification."""

import arviz as az
import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from impulso.protocols import IdentificationScheme


@pytest.fixture
def stationary_idata_2v():
    """Synthetic InferenceData with stationary VAR(1) coefficients."""
    rng = np.random.default_rng(42)
    n_chains, n_draws, n_vars = 2, 50, 2

    # Stationary coefficients: eigenvalues inside unit circle
    B = np.zeros((n_chains, n_draws, n_vars, n_vars))
    for c in range(n_chains):
        for d in range(n_draws):
            # Diagonal with small values ensures stationarity
            B[c, d] = np.diag(rng.uniform(0.1, 0.4, n_vars))

    intercept = rng.standard_normal((n_chains, n_draws, n_vars)) * 0.01
    sigma = np.zeros((n_chains, n_draws, n_vars, n_vars))
    for c in range(n_chains):
        for d in range(n_draws):
            A = rng.standard_normal((n_vars, n_vars)) * 0.5
            sigma[c, d] = A @ A.T + np.eye(n_vars)

    posterior = xr.Dataset({
        "B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"]),
        "intercept": xr.DataArray(intercept, dims=["chain", "draw", "var"]),
        "Sigma": xr.DataArray(sigma, dims=["chain", "draw", "var1", "var2"]),
    })
    return az.InferenceData(posterior=posterior)


class TestLongRunRestrictionConstruction:
    def test_basic_construction(self):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["output", "prices"])
        assert lr.ordering == ["output", "prices"]

    def test_frozen(self):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["a", "b"])
        with pytest.raises(ValidationError):
            lr.ordering = ["b", "a"]

    def test_satisfies_protocol(self):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["a", "b"])
        assert isinstance(lr, IdentificationScheme)


class TestLongRunRestrictionIdentify:
    def test_produces_structural_shock_matrix(self, stationary_idata_2v):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y1", "y2"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        assert "structural_shock_matrix" in result.posterior

    def test_output_shape(self, stationary_idata_2v):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y1", "y2"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        P = result.posterior["structural_shock_matrix"].values
        assert P.shape == (2, 50, 2, 2)

    def test_no_nan_values(self, stationary_idata_2v):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y1", "y2"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        P = result.posterior["structural_shock_matrix"].values
        assert not np.any(np.isnan(P))

    def test_long_run_impact_is_lower_triangular(self, stationary_idata_2v):
        """The long-run cumulative impact C(1) @ P should be lower triangular."""
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y1", "y2"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        P = result.posterior["structural_shock_matrix"].values
        B = stationary_idata_2v.posterior["B"].values
        n_vars = 2

        for c in range(2):
            for d in range(50):
                lag_coefficient_sum = B[c, d, :, :n_vars]
                long_run_multiplier = np.linalg.inv(np.eye(n_vars) - lag_coefficient_sum)
                long_run_impact = long_run_multiplier @ P[c, d]
                # Upper triangle (excluding diagonal) should be ~zero
                np.testing.assert_allclose(
                    np.triu(long_run_impact, k=1),
                    0.0,
                    atol=1e-10,
                )

    def test_reordering_works(self, stationary_idata_2v):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y2", "y1"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        assert result.posterior["structural_shock_matrix"].coords["shock"].values.tolist() == ["y2", "y1"]

    def test_coordinates_match_ordering(self, stationary_idata_2v):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y1", "y2"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        assert result.posterior["structural_shock_matrix"].coords["shock"].values.tolist() == ["y1", "y2"]
        assert result.posterior["structural_shock_matrix"].coords["response"].values.tolist() == ["y1", "y2"]

    def test_preserves_other_posterior_variables(self, stationary_idata_2v):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y1", "y2"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        assert "B" in result.posterior
        assert "Sigma" in result.posterior
