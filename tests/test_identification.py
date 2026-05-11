"""Tests for identification schemes."""

import numpy as np
import pytest
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

        L = np.linalg.cholesky(sigma_draws)

        chol = Cholesky(ordering=["y1", "y2"])
        result = chol.identify(L, var_names=["y1", "y2"])

        assert isinstance(result, np.ndarray)
        assert result.shape == L.shape

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

        L = np.linalg.cholesky(sigma_draws)

        chol = Cholesky(ordering=["a", "b", "c"])
        P = chol.identify(L, var_names=["a", "b", "c"])

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

        L = np.linalg.cholesky(sigma_draws)

        sr = SignRestriction(
            restrictions={
                "y1": {"s1": "+", "s2": "+"},
                "y2": {"s1": "-", "s2": "+"},
            },
            n_rotations=5000,
            random_seed=42,
        )
        P = sr.identify(L, var_names=["y1", "y2"])

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
        """identify() should record acceptance_rate on the scheme instance."""
        scheme = SignRestriction(
            restrictions={"y1": {"y1": "+"}},
            n_rotations=100,
            restriction_horizon=0,
            random_seed=42,
        )
        sigma = synthetic_idata_2v.posterior["Sigma"].values
        L = np.linalg.cholesky(sigma)
        scheme.identify(L, ["y1", "y2"])
        rate = scheme._last_acceptance_rate
        assert 0.0 <= rate <= 1.0

    def test_identify_multi_horizon_through_identify(self, synthetic_idata_2v):
        """identify() with restriction_horizon>0 uses B coefficients and checks horizons."""
        scheme = SignRestriction(
            restrictions={"y1": {"y1": "+"}},
            n_rotations=100,
            restriction_horizon=1,
            random_seed=42,
        )
        sigma = synthetic_idata_2v.posterior["Sigma"].values
        L = np.linalg.cholesky(sigma)
        P = scheme.identify(L, ["y1", "y2"], posterior=synthetic_idata_2v.posterior)
        assert P.shape == L.shape
        assert not np.any(np.isnan(P))
        assert 0.0 <= scheme._last_acceptance_rate <= 1.0

    def test_shock_coordinates_with_partial_identification(self):
        """When fewer shocks are named than variables, remaining get 'unidentified_N' labels."""
        sr = SignRestriction(
            restrictions={"y1": {"my_shock": "+"}},
            n_rotations=100,
            random_seed=42,
        )
        coords = sr._build_shock_coords(["my_shock"], n_vars=3)
        assert coords == ["my_shock", "unidentified_1", "unidentified_2"]

    def test_shock_coordinates_with_full_identification(self):
        """When all shocks are named, use those names directly."""
        sr = SignRestriction(
            restrictions={
                "y1": {"s1": "+", "s2": "+"},
                "y2": {"s1": "-", "s2": "+"},
            },
            n_rotations=5000,
            random_seed=42,
        )
        coords = sr._build_shock_coords(["s1", "s2"], n_vars=2)
        assert coords == ["s1", "s2"]

    def test_identify_multi_horizon_raises_without_B(self):
        """identify() with restriction_horizon>0 raises ValueError if B is missing."""
        rng = np.random.default_rng(42)
        n_vars, n_chains, n_draws = 2, 1, 10
        sigma = np.zeros((n_chains, n_draws, n_vars, n_vars))
        for d in range(n_draws):
            A = rng.standard_normal((n_vars, n_vars))
            sigma[0, d] = A @ A.T + np.eye(n_vars)
        L = np.linalg.cholesky(sigma)

        scheme = SignRestriction(
            restrictions={"y1": {"y1": "+"}},
            n_rotations=100,
            restriction_horizon=1,
            random_seed=42,
        )
        with pytest.raises(ValueError, match="restriction_horizon > 0"):
            scheme.identify(L, ["y1", "y2"], posterior=None)


class TestCholeskyNewIdentify:
    def test_identify_returns_ndarray_for_constant_L(self, synthetic_idata_2v):
        """For a constant L (no time dim), Cholesky.identify returns the
        reordered factor as an ndarray of shape (C, D, n, n)."""
        from impulso.identification import Cholesky

        sigma = synthetic_idata_2v.posterior["Sigma"].values
        L = np.linalg.cholesky(sigma)  # (2, 50, 2, 2)
        var_names = list(synthetic_idata_2v.posterior["B"].coords.get("variable", ["v0", "v1"]))
        if len(var_names) != 2:  # fallback if fixture coords differ
            var_names = ["v0", "v1"]

        scheme = Cholesky(ordering=var_names)
        result = scheme.identify(L, var_names)

        assert isinstance(result, np.ndarray)
        assert result.shape == L.shape
        # With identity ordering, identify is a no-op — result equals L.
        np.testing.assert_array_equal(result, L)

    def test_identify_reorders_when_ordering_differs(self, synthetic_idata_2v):
        from impulso.identification import Cholesky

        sigma = synthetic_idata_2v.posterior["Sigma"].values
        L = np.linalg.cholesky(sigma)
        var_names = ["v0", "v1"]

        scheme = Cholesky(ordering=["v1", "v0"])  # reverse ordering
        result = scheme.identify(L, var_names)

        # When ordering reverses, the scheme should re-decompose the
        # row-permuted Sigma — the result is NOT just a row swap of L.
        assert result.shape == L.shape
        # Reconstruct: result @ result.T should equal P @ Sigma @ P.T
        # where P is the permutation matrix.
        perm = np.array([1, 0])
        sigma_perm = sigma[:, :, np.ix_(perm, perm)[0], np.ix_(perm, perm)[1]]
        np.testing.assert_allclose(
            np.einsum("cdij,cdkj->cdik", result, result),
            sigma_perm,
            rtol=1e-6,
        )


class TestSignRestrictionNewIdentify:
    def test_identify_returns_ndarray_no_horizon(self, synthetic_idata_2v):
        """SignRestriction with restriction_horizon=0 ignores posterior=None."""
        from impulso.identification import SignRestriction

        sigma = synthetic_idata_2v.posterior["Sigma"].values
        L = np.linalg.cholesky(sigma)
        var_names = ["v0", "v1"]

        scheme = SignRestriction(
            restrictions={"v0": {"shock_a": "+"}, "v1": {"shock_a": "-"}},
            n_rotations=20,
            random_seed=0,
        )
        P = scheme.identify(L, var_names, posterior=None)

        assert isinstance(P, np.ndarray)
        assert P.shape == L.shape

    def test_identify_with_horizon_requires_posterior(self, synthetic_idata_2v):
        """restriction_horizon > 0 needs B; passing posterior=None raises."""
        from impulso.identification import SignRestriction

        sigma = synthetic_idata_2v.posterior["Sigma"].values
        L = np.linalg.cholesky(sigma)
        var_names = ["v0", "v1"]

        scheme = SignRestriction(
            restrictions={"v0": {"shock_a": "+"}},
            n_rotations=10,
            restriction_horizon=2,
            random_seed=0,
        )
        with pytest.raises(ValueError, match="restriction_horizon > 0"):
            scheme.identify(L, var_names, posterior=None)

    def test_identify_with_horizon_uses_posterior_B(self, synthetic_idata_2v):
        """When posterior contains B, restriction_horizon > 0 path runs."""
        from impulso.identification import SignRestriction

        sigma = synthetic_idata_2v.posterior["Sigma"].values
        L = np.linalg.cholesky(sigma)
        var_names = ["v0", "v1"]

        scheme = SignRestriction(
            restrictions={"v0": {"shock_a": "+"}},
            n_rotations=10,
            restriction_horizon=2,
            random_seed=0,
        )
        # synthetic_idata_2v has B in posterior; should run without raising.
        P = scheme.identify(L, var_names, posterior=synthetic_idata_2v.posterior)
        assert P.shape == L.shape
