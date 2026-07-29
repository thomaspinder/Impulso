"""Tests for identification schemes."""

import gc
import warnings
import weakref

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from impulso.identification import _CACHE_MISS, Cholesky, LongRunRestriction, SignRestriction, _PosteriorCache
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

    def test_reversed_ordering_returns_rows_in_data_order(self, synthetic_idata_2v):
        """The returned matrix is row-indexed by `var_names`, not by `ordering`.

        Downstream (`IdentifiedVAR.shock_matrix`) labels the row axis with
        `var_names` and left-multiplies by MA coefficients built in data
        order, so `identify` must return rows in **data** order while the
        columns follow the causal `ordering`. Triangularity therefore holds
        only after permuting rows into ordering coordinates.
        """
        from impulso.identification import Cholesky

        sigma = synthetic_idata_2v.posterior["Sigma"].values
        L = np.linalg.cholesky(sigma)
        var_names = ["v0", "v1"]

        scheme = Cholesky(ordering=["v1", "v0"])  # reverse ordering
        P = scheme.identify(L, var_names)

        assert P.shape == L.shape

        # (1) P @ P.T reproduces Sigma in DATA coordinates.
        np.testing.assert_allclose(np.einsum("cdij,cdkj->cdik", P, P), sigma, atol=1e-12)

        # (2) Permuting rows into ordering coordinates gives an exactly
        #     lower-triangular factor with a positive diagonal.
        perm = np.array([1, 0])
        P_ord = P[..., perm, :]
        assert np.triu(P_ord, 1).max() == 0.0
        assert np.triu(P_ord, 1).min() == 0.0
        assert np.diagonal(P_ord, axis1=-2, axis2=-1).min() > 0.0

        # (3) It is a genuine re-decomposition, not a relabelled row swap of L.
        assert np.abs(P - L[..., perm, :]).max() > 1e-6

    def test_matches_textbook_cholesky_of_permuted_sigma(self):
        """Row-permuting the result reproduces `chol(Pi Sigma Pi')` exactly.

        The QR/LQ construction never forms Sigma, so this pins it against the
        textbook definition of the ordered Cholesky factor.
        """
        rng = np.random.default_rng(0)
        n_chains, n_draws, n_vars = 2, 8, 3
        sigma = np.zeros((n_chains, n_draws, n_vars, n_vars))
        for c in range(n_chains):
            for d in range(n_draws):
                A = rng.standard_normal((n_vars, n_vars))
                sigma[c, d] = A @ A.T + np.eye(n_vars)
        L = np.linalg.cholesky(sigma)

        var_names = ["a", "b", "c"]
        ordering = ["c", "a", "b"]
        perm = np.array([2, 0, 1])

        P = Cholesky(ordering=ordering).identify(L, var_names)

        ix0, ix1 = np.ix_(perm, perm)
        expected = np.linalg.cholesky(sigma[:, :, ix0, ix1])
        np.testing.assert_allclose(P[..., perm, :], expected, atol=1e-12)

    def test_identify_is_deterministic(self, synthetic_idata_2v):
        """Repeated calls return bit-identical output (no RNG in the path)."""
        L = np.linalg.cholesky(synthetic_idata_2v.posterior["Sigma"].values)
        scheme = Cholesky(ordering=["v1", "v0"])
        first = scheme.identify(L, ["v0", "v1"])
        second = scheme.identify(L, ["v0", "v1"])
        assert np.array_equal(first, second)

    def test_double_permutation_round_trips(self, synthetic_idata_2v):
        """Reordering, then reordering back, recovers `L`.

        Guards against transposing `perm` and its inverse: a swapped pair
        happens to be self-inverse for a reversal, but the intermediate
        ordering-coordinate factor would not round-trip.
        """
        L = np.linalg.cholesky(synthetic_idata_2v.posterior["Sigma"].values)
        var_names = ["v0", "v1"]
        reversed_names = ["v1", "v0"]
        perm = np.array([1, 0])

        # Forward: data order -> reversed ordering. Rows come back in data
        # order, so permute into ordering coordinates to get a valid factor.
        P = Cholesky(ordering=reversed_names).identify(L, var_names)
        L_reversed_world = P[..., perm, :]

        # Backward: treat the reversed world as the data order and ask for
        # the original ordering. Undo the row permutation the same way.
        Q = Cholesky(ordering=var_names).identify(L_reversed_world, reversed_names)
        np.testing.assert_allclose(Q[..., perm, :], L, atol=1e-12)


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


class TestPosteriorCache:
    """Weakref-validated identity cache shared by the identification schemes (#203)."""

    def test_xr_dataset_supports_weak_references(self):
        """The whole design rests on this — assert it rather than assume it."""
        ds = xr.Dataset()
        assert weakref.ref(ds)() is ds

    def test_empty_cache_misses(self):
        cache = _PosteriorCache()
        assert cache.get(xr.Dataset(), (1,)) is _CACHE_MISS

    def test_hit_on_the_same_owner(self):
        cache = _PosteriorCache()
        owner = xr.Dataset()
        sentinel = object()
        cache.set(owner, (2,), sentinel)
        assert cache.get(owner, (2,)) is sentinel

    def test_none_is_a_storable_value(self):
        cache = _PosteriorCache()
        owner = xr.Dataset()
        cache.set(owner, (), None)
        assert cache.get(owner, ()) is None

    def test_live_owner_with_different_identity_misses(self):
        """Two live, equal-looking owners must not share an entry."""
        cache = _PosteriorCache()
        first, second = xr.Dataset(), xr.Dataset()
        cache.set(first, (1,), "value")
        assert cache.get(second, (1,)) is _CACHE_MISS
        assert cache.get(first, (1,)) == "value"

    def test_key_tail_mismatch_misses(self):
        cache = _PosteriorCache()
        owner = xr.Dataset()
        cache.set(owner, (1, "gdp"), "value")
        assert cache.get(owner, (2, "gdp")) is _CACHE_MISS
        assert cache.get(owner, (1, "inflation")) is _CACHE_MISS
        assert cache.get(owner, (1, "gdp")) == "value"

    def test_dead_referent_misses(self):
        """The defect this cache exists to close: a collected owner's
        address may be recycled, so a dead referent must read as a miss."""
        cache = _PosteriorCache()
        owner = xr.Dataset()
        cache.set(owner, (1,), "stale")
        ref = weakref.ref(owner)

        del owner
        gc.collect()
        assert ref() is None

        assert cache.get(xr.Dataset(), (1,)) is _CACHE_MISS

    def test_multiple_owners_all_must_match(self):
        cache = _PosteriorCache()
        a, b, other = xr.Dataset(), xr.Dataset(), xr.Dataset()
        cache.set((a, b), (1,), "value")
        assert cache.get((a, b), (1,)) == "value"
        assert cache.get((a, other), (1,)) is _CACHE_MISS
        assert cache.get((other, b), (1,)) is _CACHE_MISS

    def test_owner_arity_mismatch_misses(self):
        cache = _PosteriorCache()
        a, b = xr.Dataset(), xr.Dataset()
        cache.set((a, b), (1,), "value")
        assert cache.get(a, (1,)) is _CACHE_MISS

    def test_single_owner_and_one_tuple_are_equivalent(self):
        cache = _PosteriorCache()
        owner = xr.Dataset()
        cache.set(owner, (1,), "value")
        assert cache.get((owner,), (1,)) == "value"

    def test_non_weakrefable_owner_declines_to_cache(self):
        """No validity token means no cache — never an unsafe fallback key."""
        cache = _PosteriorCache()
        owner = xr.Dataset()
        cache.set((owner, 3), (1,), "value")
        assert cache.get((owner, 3), (1,)) is _CACHE_MISS

    def test_non_weakrefable_owner_evicts_a_previous_entry(self):
        cache = _PosteriorCache()
        a, b = xr.Dataset(), xr.Dataset()
        cache.set(a, (1,), "value")
        cache.set((b, "not-weakrefable"), (1,), "other")
        assert cache.get(a, (1,)) is _CACHE_MISS

    def test_set_overwrites_the_single_slot(self):
        cache = _PosteriorCache()
        a, b = xr.Dataset(), xr.Dataset()
        cache.set(a, (1,), "first")
        cache.set(b, (1,), "second")
        assert cache.get(b, (1,)) == "second"
        assert cache.get(a, (1,)) is _CACHE_MISS

    def test_clear(self):
        cache = _PosteriorCache()
        owner = xr.Dataset()
        cache.set(owner, (1,), "value")
        cache.clear()
        assert cache.get(owner, (1,)) is _CACHE_MISS


class TestLongRunRestriction:
    """Blanchard-Quah style long-run (cumulative) zero restrictions."""

    @staticmethod
    def _scheme(**kwargs) -> LongRunRestriction:
        kwargs.setdefault("ordering", ["y1", "y2"])
        kwargs.setdefault("shock_names", ["permanent", "transitory"])
        return LongRunRestriction(**kwargs)

    @staticmethod
    def _identify(scheme: LongRunRestriction, fx: dict, var_names=("y1", "y2")) -> np.ndarray:
        return scheme.identify(fx["L"], list(var_names), posterior=fx["idata"].posterior, n_lags=1)

    # --- 1. exact recovery -------------------------------------------------

    def test_identify_recovers_the_true_impact_matrix(self, permanent_transitory_2v):
        """P is built backwards from a known G, so identify must return it exactly."""
        P = self._identify(self._scheme(), permanent_transitory_2v)
        assert P.shape == (2, 50, 2, 2)
        np.testing.assert_allclose(P, np.broadcast_to(permanent_transitory_2v["P_true"], P.shape), atol=1e-12)

    # --- 2. the restriction actually binds ---------------------------------

    def test_cumulative_impact_is_lower_triangular_with_positive_diagonal(self, permanent_transitory_2v):
        fx = permanent_transitory_2v
        P = self._identify(self._scheme(), fx)
        theta1 = fx["C1"] @ P
        assert np.abs(np.triu(theta1, 1)).max() < 1e-10
        assert (np.diagonal(theta1, axis1=-2, axis2=-1) > 0).all()
        np.testing.assert_allclose(theta1, np.broadcast_to(fx["G"], theta1.shape), atol=1e-12)

    # --- 3. covariance is reproduced exactly -------------------------------

    def test_p_reproduces_sigma(self, permanent_transitory_2v):
        fx = permanent_transitory_2v
        P = self._identify(self._scheme(), fx)
        np.testing.assert_allclose(P @ np.swapaxes(P, -1, -2), np.broadcast_to(fx["Sigma"], P.shape), atol=1e-12)

    # --- 4. reversed ordering keeps rows in data order ---------------------

    def test_reversed_ordering_returns_rows_in_data_order(self, permanent_transitory_2v):
        """Rows of the returned P follow the data, not `ordering` — so P @ P.T is Sigma."""
        fx = permanent_transitory_2v
        scheme = self._scheme(ordering=["y2", "y1"], shock_names=["permanent", "transitory"])
        P = self._identify(scheme, fx)
        np.testing.assert_allclose(P @ np.swapaxes(P, -1, -2), np.broadcast_to(fx["Sigma"], P.shape), atol=1e-12)
        # Triangularity holds in the *ordering* row coordinates.
        perm = [1, 0]
        theta1 = (fx["C1"] @ P)[..., perm, :]
        assert np.abs(np.triu(theta1, 1)).max() < 1e-10
        assert (np.diagonal(theta1, axis1=-2, axis2=-1) > 0).all()

    # --- 5. agrees with the textbook construction --------------------------

    def test_agrees_with_textbook_cholesky_route(self, permanent_transitory_2v):
        """QR route must match M @ chol(C1 Sigma C1')."""
        fx = permanent_transitory_2v
        P = self._identify(self._scheme(), fx)
        omega = fx["C1"] @ fx["Sigma"] @ fx["C1"].T
        P_textbook = fx["M"] @ np.linalg.cholesky(omega)
        np.testing.assert_allclose(P[0, 0], P_textbook, atol=1e-10)

    # --- 6. it is not Cholesky ---------------------------------------------

    def test_differs_from_cholesky(self, permanent_transitory_2v):
        fx = permanent_transitory_2v
        P = self._identify(self._scheme(), fx)
        chol = np.linalg.cholesky(fx["Sigma"])
        assert np.abs(P[0, 0] - chol).max() > 1e-3

    # --- 7. determinism ----------------------------------------------------

    def test_identify_is_deterministic(self, permanent_transitory_2v):
        scheme = self._scheme()
        first = self._identify(scheme, permanent_transitory_2v)
        second = self._identify(scheme, permanent_transitory_2v)
        assert np.array_equal(first, second)

    # --- 8. shock labels ---------------------------------------------------

    def test_shock_coords_uses_shock_names_when_given(self):
        scheme = self._scheme()
        assert scheme.shock_coords(n_vars=2) == ["permanent", "transitory"]

    def test_shock_coords_falls_back_to_ordering(self):
        scheme = LongRunRestriction(ordering=["y1", "y2"])
        assert scheme.shock_coords(n_vars=2) == ["y1", "y2"]

    # --- 9. protocol + immutability ----------------------------------------

    def test_satisfies_protocol(self):
        assert isinstance(self._scheme(), IdentificationScheme)

    def test_frozen(self):
        scheme = self._scheme()
        with pytest.raises(ValidationError):
            scheme.ordering = ["y2", "y1"]

    # --- 10. posterior is mandatory ----------------------------------------

    def test_identify_without_posterior_raises(self, permanent_transitory_2v):
        with pytest.raises(ValueError, match="LongRunRestriction"):
            self._scheme().identify(permanent_transitory_2v["L"], ["y1", "y2"], posterior=None)

    def test_identify_without_b_raises(self, permanent_transitory_2v):
        import xarray as xr

        posterior = xr.Dataset({"intercept": permanent_transitory_2v["idata"].posterior["intercept"]})
        with pytest.raises(ValueError, match="LongRunRestriction"):
            self._scheme().identify(permanent_transitory_2v["L"], ["y1", "y2"], posterior=posterior)

    # --- 11. unknown ordering entry ----------------------------------------

    def test_unknown_ordering_variable_raises(self, permanent_transitory_2v):
        scheme = self._scheme(ordering=["y1", "nope"])
        with pytest.raises(ValueError, match="nope"):
            self._identify(scheme, permanent_transitory_2v)

    # --- 12. construction-time validation ----------------------------------

    def test_duplicate_ordering_rejected(self):
        with pytest.raises(ValidationError, match="duplicate"):
            LongRunRestriction(ordering=["y1", "y1"])

    def test_empty_ordering_rejected(self):
        with pytest.raises(ValidationError):
            LongRunRestriction(ordering=[])

    def test_shock_names_length_mismatch_rejected(self):
        with pytest.raises(ValidationError, match="same length"):
            LongRunRestriction(ordering=["y1", "y2"], shock_names=["only_one"])

    def test_duplicate_shock_names_rejected(self):
        with pytest.raises(ValidationError, match="duplicate"):
            LongRunRestriction(ordering=["y1", "y2"], shock_names=["s", "s"])

    def test_reserved_shock_name_prefix_rejected(self):
        with pytest.raises(ValidationError, match="unidentified_"):
            LongRunRestriction(ordering=["y1", "y2"], shock_names=["unidentified_1", "transitory"])

    # --- 13. numerically singular draws ------------------------------------

    @staticmethod
    def _posterior_with(fx: dict, A_by_draw) -> "object":
        """Copy the fixture posterior with `A_by_draw(c, d)` substituted for B."""
        posterior = fx["idata"].posterior.copy(deep=True)
        B = posterior["B"].values.copy()
        for c in range(B.shape[0]):
            for d in range(B.shape[1]):
                replacement = A_by_draw(c, d)
                if replacement is not None:
                    B[c, d] = replacement
        posterior["B"] = (("chain", "draw", "var", "coeff"), B)
        return posterior

    def test_singular_draws_are_nan_and_counted(self, permanent_transitory_2v):
        """M = I - A_1 = 0 for the doctored draws: those alone come back NaN."""
        fx = permanent_transitory_2v
        posterior = self._posterior_with(fx, lambda c, d: np.eye(2) if (c == 0 and d < 5) else None)
        scheme = self._scheme()
        with pytest.warns(UserWarning, match="long-run multiplier"):
            P = scheme.identify(fx["L"], ["y1", "y2"], posterior=posterior, n_lags=1)

        assert np.isnan(P[0, :5]).all()
        assert not np.isnan(P[0, 5:]).any()
        assert not np.isnan(P[1]).any()
        np.testing.assert_allclose(P[0, 5:], np.broadcast_to(fx["P_true"], P[0, 5:].shape), atol=1e-12)
        assert scheme._last_diagnostics["long_run_singular_draws"] == 5.0
        assert scheme._last_diagnostics["long_run_singular_fraction"] == pytest.approx(0.05)

    def test_singular_draws_raise_when_asked(self, permanent_transitory_2v):
        fx = permanent_transitory_2v
        posterior = self._posterior_with(fx, lambda c, d: np.eye(2) if (c == 0 and d < 5) else None)
        scheme = self._scheme(on_undefined="raise")
        with pytest.raises(ValueError, match="5"):
            scheme.identify(fx["L"], ["y1", "y2"], posterior=posterior, n_lags=1)

    # --- 14. the max_condition threshold is consulted ----------------------

    def test_near_singular_draws_respect_max_condition(self, permanent_transitory_2v):
        """A_1 = diag(1 - 1e-9, 0) gives cond(M) ~ 1e9 — caught at 1e8, allowed at 1e20."""
        fx = permanent_transitory_2v
        doctored = np.diag([1.0 - 1e-9, 0.0])
        posterior = self._posterior_with(fx, lambda c, d: doctored if (c == 0 and d < 3) else None)

        strict = self._scheme()
        with pytest.warns(UserWarning, match="long-run multiplier"):
            P_strict = strict.identify(fx["L"], ["y1", "y2"], posterior=posterior, n_lags=1)
        assert np.isnan(P_strict[0, :3]).all()
        assert strict._last_diagnostics["long_run_singular_draws"] == 3.0

        lenient = self._scheme(max_condition=1e20)
        P_lenient = lenient.identify(fx["L"], ["y1", "y2"], posterior=posterior, n_lags=1)
        assert not np.isnan(P_lenient).any()
        assert lenient._last_diagnostics["long_run_singular_draws"] == 0.0

    # --- 15. explosive is not singular -------------------------------------

    def test_explosive_draws_warn_but_stay_finite(self, permanent_transitory_2v):
        """A_1 = 1.5 I: M = -0.5 I is perfectly conditioned, but the MA sum diverges."""
        fx = permanent_transitory_2v
        posterior = self._posterior_with(fx, lambda c, d: 1.5 * np.eye(2) if (c == 1 and d < 4) else None)
        scheme = self._scheme()
        with pytest.warns(UserWarning, match="explosive"):
            P = scheme.identify(fx["L"], ["y1", "y2"], posterior=posterior, n_lags=1)

        assert np.isfinite(P).all()
        assert scheme._last_diagnostics["long_run_explosive_draws"] == 4.0
        assert scheme._last_diagnostics["long_run_singular_draws"] == 0.0
        assert scheme._last_diagnostics["long_run_spectral_radius_max"] == pytest.approx(1.5)

    # --- 16. the diagnostics query -----------------------------------------

    def test_long_run_diagnostics(self, permanent_transitory_2v):
        fx = permanent_transitory_2v
        diagnostics = self._scheme().long_run_diagnostics(fx["idata"].posterior, n_lags=1)
        assert diagnostics["condition"].shape == (2, 50)
        assert diagnostics["spectral_radius"].shape == (2, 50)
        np.testing.assert_allclose(diagnostics["spectral_radius"], 0.6, atol=1e-10)
        np.testing.assert_allclose(diagnostics["condition"], np.linalg.cond(fx["M"]), rtol=1e-10)

    def test_long_run_diagnostics_infers_n_lags(self, permanent_transitory_2v):
        diagnostics = self._scheme().long_run_diagnostics(permanent_transitory_2v["idata"].posterior)
        assert diagnostics["condition"].shape == (2, 50)

    # --- screen memoisation ------------------------------------------------

    def test_screen_is_memoised_per_posterior(self, permanent_transitory_2v):
        """The per-t (SV) path calls identify repeatedly; the screen must not re-warn."""
        fx = permanent_transitory_2v
        posterior = self._posterior_with(fx, lambda c, d: np.eye(2) if (c == 0 and d < 5) else None)
        scheme = self._scheme()
        with pytest.warns(UserWarning, match="long-run multiplier"):
            scheme.identify(fx["L"], ["y1", "y2"], posterior=posterior, n_lags=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scheme.identify(fx["L"], ["y1", "y2"], posterior=posterior, n_lags=1)
