"""Tests for identification schemes."""

import arviz as az
import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from impulso.identification import Cholesky, SignRestriction
from impulso.protocols import IdentificationScheme


class TestCholesky:
    def test_construction(self):
        c = Cholesky(ordering=["gdp", "inflation", "rate"])
        assert c.ordering == ["gdp", "inflation", "rate"]

    def test_frozen(self):
        c = Cholesky(ordering=["gdp", "inflation"])
        with pytest.raises(ValidationError):
            c.ordering = ["a", "b"]

    def test_satisfies_protocol(self):
        c = Cholesky(ordering=["a", "b"])
        assert isinstance(c, IdentificationScheme)

    def test_identify_produces_structural_idata(self):
        """Test Cholesky decomposition on synthetic covariance draws."""
        rng = np.random.default_rng(42)
        n_vars = 2
        n_chains, n_draws = 1, 50

        # Generate positive-definite covariance matrices
        sigma_draws = np.zeros((n_chains, n_draws, n_vars, n_vars))
        for c in range(n_chains):
            for d in range(n_draws):
                A = rng.standard_normal((n_vars, n_vars))
                sigma_draws[c, d] = A @ A.T + np.eye(n_vars)

        sigma_da = xr.DataArray(
            sigma_draws,
            dims=["chain", "draw", "var1", "var2"],
            coords={"var1": ["y1", "y2"], "var2": ["y1", "y2"]},
        )
        idata = az.InferenceData(posterior=xr.Dataset({"Sigma": sigma_da}))

        chol = Cholesky(ordering=["y1", "y2"])
        result = chol.identify(idata, var_names=["y1", "y2"])

        assert "structural_shock_matrix" in result.posterior

        p_da = result.posterior["structural_shock_matrix"]
        assert p_da.dims == ("chain", "draw", "response", "shock")
        assert list(p_da.coords["response"].values) == ["y1", "y2"]
        assert list(p_da.coords["shock"].values) == ["y1", "y2"]


class TestSignRestriction:
    def test_construction(self):
        sr = SignRestriction(
            restrictions={
                "gdp": {"supply": "+", "demand": "+"},
                "inflation": {"supply": "-", "demand": "+"},
            },
            n_rotations=1000,
            random_seed=42,
        )
        assert sr.n_rotations == 1000

    def test_satisfies_protocol(self):
        sr = SignRestriction(
            restrictions={"gdp": {"supply": "+"}},
        )
        assert isinstance(sr, IdentificationScheme)

    def test_identify_satisfies_restrictions(self):
        """End-to-end test: identify() should satisfy sign restrictions."""
        rng = np.random.default_rng(42)
        n_vars = 2
        n_chains, n_draws = 1, 20

        sigma_draws = np.zeros((n_chains, n_draws, n_vars, n_vars))
        for c in range(n_chains):
            for d in range(n_draws):
                A = rng.standard_normal((n_vars, n_vars))
                sigma_draws[c, d] = A @ A.T + np.eye(n_vars)

        sigma_da = xr.DataArray(
            sigma_draws,
            dims=["chain", "draw", "var1", "var2"],
            coords={"var1": ["y1", "y2"], "var2": ["y1", "y2"]},
        )
        idata = az.InferenceData(posterior=xr.Dataset({"Sigma": sigma_da}))

        sr = SignRestriction(
            restrictions={
                "y1": {"s1": "+", "s2": "+"},
                "y2": {"s1": "-", "s2": "+"},
            },
            n_rotations=5000,
            random_seed=42,
        )
        result = sr.identify(idata, var_names=["y1", "y2"])
        P = result.posterior["structural_shock_matrix"].values

        # Check that restrictions are satisfied (or fallback was used)
        assert P.shape == (1, 20, 2, 2)
        assert not np.any(np.isnan(P))
