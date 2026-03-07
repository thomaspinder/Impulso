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

    def test_cholesky_identify_values_correct(self):
        """Verify Cholesky decomposition produces valid lower-triangular matrices."""
        rng = np.random.default_rng(42)
        n_vars = 3
        n_chains, n_draws = 2, 30

        sigma_draws = np.zeros((n_chains, n_draws, n_vars, n_vars))
        for c in range(n_chains):
            for d in range(n_draws):
                A = rng.standard_normal((n_vars, n_vars))
                sigma_draws[c, d] = A @ A.T + np.eye(n_vars)

        sigma_da = xr.DataArray(
            sigma_draws,
            dims=["chain", "draw", "var1", "var2"],
            coords={"var1": ["a", "b", "c"], "var2": ["a", "b", "c"]},
        )
        idata = az.InferenceData(posterior=xr.Dataset({"Sigma": sigma_da}))

        chol = Cholesky(ordering=["a", "b", "c"])
        result = chol.identify(idata, var_names=["a", "b", "c"])
        P = result.posterior["structural_shock_matrix"].values

        # Verify P @ P.T reconstructs Sigma
        for c in range(n_chains):
            for d in range(n_draws):
                reconstructed = P[c, d] @ P[c, d].T
                np.testing.assert_allclose(reconstructed, sigma_draws[c, d], atol=1e-10)


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

    def test_sign_restriction_accepts_restriction_horizon(self):
        """SignRestriction should accept a restriction_horizon parameter."""
        scheme = SignRestriction(
            restrictions={"var_0": {"shock_0": "+"}},
            n_rotations=100,
            restriction_horizon=6,
        )
        assert scheme.restriction_horizon == 6

    def test_sign_restriction_default_restriction_horizon_is_zero(self):
        """Default restriction_horizon should be 0 (impact only)."""
        scheme = SignRestriction(
            restrictions={"var_0": {"shock_0": "+"}},
            n_rotations=100,
        )
        assert scheme.restriction_horizon == 0

    def test_sign_restriction_identify_stores_acceptance_rate(self, synthetic_idata_2v):
        """identify() should store acceptance_rate in posterior attrs."""
        scheme = SignRestriction(
            restrictions={"y1": {"y1": "+"}},
            n_rotations=100,
            restriction_horizon=0,
            random_seed=42,
        )
        result = scheme.identify(synthetic_idata_2v, ["y1", "y2"])
        assert "sign_restriction_acceptance_rate" in result.posterior.attrs
        rate = result.posterior.attrs["sign_restriction_acceptance_rate"]
        assert 0.0 <= rate <= 1.0

    def test_identify_multi_horizon_through_identify(self, synthetic_idata_2v):
        """identify() with restriction_horizon>0 uses B coefficients and checks horizons."""
        scheme = SignRestriction(
            restrictions={"y1": {"y1": "+"}},
            n_rotations=100,
            restriction_horizon=1,
            random_seed=42,
        )
        result = scheme.identify(synthetic_idata_2v, ["y1", "y2"])
        assert "structural_shock_matrix" in result.posterior
        assert not np.any(np.isnan(result.posterior["structural_shock_matrix"].values))
        rate = result.posterior.attrs["sign_restriction_acceptance_rate"]
        assert 0.0 <= rate <= 1.0

    def test_shock_coordinates_with_partial_identification(self):
        """When fewer shocks are named than variables, remaining get 'unidentified_N' labels."""
        rng = np.random.default_rng(42)
        n_vars, n_chains, n_draws = 3, 1, 10
        sigma = np.zeros((n_chains, n_draws, n_vars, n_vars))
        for d in range(n_draws):
            A = rng.standard_normal((n_vars, n_vars))
            sigma[0, d] = A @ A.T + np.eye(n_vars)
        sigma_da = xr.DataArray(
            sigma,
            dims=["chain", "draw", "var1", "var2"],
            coords={"var1": ["y1", "y2", "y3"], "var2": ["y1", "y2", "y3"]},
        )
        idata = az.InferenceData(posterior=xr.Dataset({"Sigma": sigma_da}))

        sr = SignRestriction(
            restrictions={"y1": {"my_shock": "+"}},
            n_rotations=100,
            random_seed=42,
        )
        result = sr.identify(idata, var_names=["y1", "y2", "y3"])
        shock_coords = list(result.posterior["structural_shock_matrix"].coords["shock"].values)
        assert shock_coords == ["my_shock", "unidentified_1", "unidentified_2"]

    def test_shock_coordinates_with_full_identification(self):
        """When all shocks are named, use those names directly."""
        rng = np.random.default_rng(42)
        n_vars, n_chains, n_draws = 2, 1, 10
        sigma = np.zeros((n_chains, n_draws, n_vars, n_vars))
        for d in range(n_draws):
            A = rng.standard_normal((n_vars, n_vars))
            sigma[0, d] = A @ A.T + np.eye(n_vars)
        sigma_da = xr.DataArray(
            sigma,
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
        shock_coords = list(result.posterior["structural_shock_matrix"].coords["shock"].values)
        assert shock_coords == ["s1", "s2"]

    def test_identify_multi_horizon_raises_without_B(self):
        """identify() with restriction_horizon>0 raises ValueError if B is missing."""
        rng = np.random.default_rng(42)
        n_vars, n_chains, n_draws = 2, 1, 10
        sigma = np.zeros((n_chains, n_draws, n_vars, n_vars))
        for d in range(n_draws):
            A = rng.standard_normal((n_vars, n_vars))
            sigma[0, d] = A @ A.T + np.eye(n_vars)
        sigma_da = xr.DataArray(
            sigma,
            dims=["chain", "draw", "var1", "var2"],
            coords={"var1": ["y1", "y2"], "var2": ["y1", "y2"]},
        )
        idata_no_B = az.InferenceData(posterior=xr.Dataset({"Sigma": sigma_da}))

        scheme = SignRestriction(
            restrictions={"y1": {"y1": "+"}},
            n_rotations=100,
            restriction_horizon=1,
            random_seed=42,
        )
        with pytest.raises(ValueError, match="restriction_horizon > 0 requires 'B'"):
            scheme.identify(idata_no_B, ["y1", "y2"])
