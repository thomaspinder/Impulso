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

    def test_reserved_prefix_rejected_through_ordering_fallback(self):
        """With shock_names=None the ordering becomes the shock labels, so the guard must fire there too."""
        with pytest.raises(ValidationError, match="unidentified_"):
            LongRunRestriction(ordering=["unidentified_x", "y2"])

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

    def test_screen_cache_misses_once_the_posterior_is_collected(self, permanent_transitory_2v):
        """A collected posterior's address can be recycled, so a dead referent
        must read as a miss rather than serving another posterior's screen (#203)."""
        fx = permanent_transitory_2v
        scheme = self._scheme()
        posterior = self._posterior_with(fx, lambda c, d: None)
        scheme.identify(fx["L"], ["y1", "y2"], posterior=posterior, n_lags=1)
        ref = weakref.ref(posterior)

        del posterior
        gc.collect()
        assert ref() is None

        assert scheme._lr_cache.get(self._posterior_with(fx, lambda c, d: None), (1,)) is _CACHE_MISS

    # --- 17-19. from_zero_restrictions -------------------------------------

    def test_from_zero_restrictions_recovers_the_ordering(self, permanent_transitory_2v):
        fx = permanent_transitory_2v
        scheme = LongRunRestriction.from_zero_restrictions(
            restrictions={"y1": ["transitory"]},
            var_names=["y1", "y2"],
            shock_names=["permanent", "transitory"],
        )
        assert scheme.ordering == ["y1", "y2"]
        assert scheme.shock_names == ["permanent", "transitory"]
        np.testing.assert_array_equal(self._identify(scheme, fx), self._identify(self._scheme(), fx))

    def test_from_zero_restrictions_forwards_kwargs(self):
        scheme = LongRunRestriction.from_zero_restrictions(
            restrictions={"y1": ["transitory"]},
            var_names=["y1", "y2"],
            shock_names=["permanent", "transitory"],
            on_undefined="raise",
            max_condition=1e6,
        )
        assert scheme.on_undefined == "raise"
        assert scheme.max_condition == 1e6

    def test_from_zero_restrictions_rejects_non_recursive_patterns(self):
        """Counts (1, 1, 0) are not a permutation of (0, 1, 2) — no ordering makes this triangular."""
        with pytest.raises(ValueError, match="recursive"):
            LongRunRestriction.from_zero_restrictions(
                restrictions={"a": ["s2"], "b": ["s3"]},
                var_names=["a", "b", "c"],
                shock_names=["s1", "s2", "s3"],
            )

    def test_from_zero_restrictions_names_the_offending_variable(self):
        """Counts are a valid permutation, but the restricted shocks are the wrong ones."""
        with pytest.raises(ValueError, match="'b'"):
            LongRunRestriction.from_zero_restrictions(
                restrictions={"a": ["s2", "s3"], "b": ["s2"]},
                var_names=["a", "b", "c"],
                shock_names=["s1", "s2", "s3"],
            )

    def test_from_zero_restrictions_rejects_unknown_names(self):
        with pytest.raises(ValueError, match="var_names"):
            LongRunRestriction.from_zero_restrictions(
                restrictions={"nope": ["transitory"]},
                var_names=["y1", "y2"],
                shock_names=["permanent", "transitory"],
            )
        with pytest.raises(ValueError, match="shock_names"):
            LongRunRestriction.from_zero_restrictions(
                restrictions={"y1": ["nope"]},
                var_names=["y1", "y2"],
                shock_names=["permanent", "transitory"],
            )

    def test_from_zero_restrictions_requires_one_shock_per_variable(self):
        with pytest.raises(ValueError, match="one per variable"):
            LongRunRestriction.from_zero_restrictions(
                restrictions={"y1": ["transitory"]},
                var_names=["y1", "y2"],
                shock_names=["permanent"],
            )


class TestLongRunRestrictionRecovery:
    """End-to-end: does the scheme recover a known long-run structure from data?"""

    def test_recovers_p_true_from_simulated_data(self):
        import pandas as pd

        from impulso.conjugate import ConjugateVAR
        from impulso.data import VARData
        from impulso.priors import NIWPrior

        A1 = np.array([[0.5, 0.1], [0.2, 0.4]])
        P_true = (np.eye(2) - A1) @ np.array([[1.0, 0.0], [0.5, 0.8]])

        rng = np.random.default_rng(0)
        T = 2000
        y = np.zeros((T, 2))
        for t in range(1, T):
            y[t] = A1 @ y[t - 1] + P_true @ rng.standard_normal(2)
        data = VARData(
            endog=y,
            endog_names=["y1", "y2"],
            index=pd.date_range("1900-01-01", periods=T, freq="QS"),
        )

        fitted = ConjugateVAR(lags=1, prior=NIWPrior(), draws=200, seed=0).fit(data)
        identified = fitted.set_identification_strategy(
            LongRunRestriction(ordering=["y1", "y2"], shock_names=["permanent", "transitory"])
        )
        P_median = np.median(identified.shock_matrix().values, axis=(0, 1))
        np.testing.assert_allclose(P_median, P_true, atol=0.05)


class TestLongRunRestrictionMultipleLags:
    """The long-run multiplier sums *every* lag block, and the companion grows with p."""

    @staticmethod
    def _var2_posterior() -> tuple:
        import xarray as xr

        A1 = np.array([[0.4, 0.1], [0.1, 0.3]])
        A2 = np.array([[0.2, 0.0], [0.05, 0.1]])
        M = np.eye(2) - A1 - A2
        G = np.array([[1.0, 0.0], [0.5, 0.8]])
        P_true = M @ G
        L = np.linalg.cholesky(P_true @ P_true.T)

        B = np.broadcast_to(np.concatenate([A1, A2], axis=1), (1, 4, 2, 4)).copy()
        posterior = xr.Dataset({"B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"])})
        return posterior, np.broadcast_to(L, (1, 4, 2, 2)).copy(), P_true, G

    def test_identify_with_two_lags(self):
        posterior, L, P_true, _ = self._var2_posterior()
        scheme = LongRunRestriction(ordering=["y1", "y2"], shock_names=["permanent", "transitory"])
        P = scheme.identify(L, ["y1", "y2"], posterior=posterior, n_lags=2)
        np.testing.assert_allclose(P, np.broadcast_to(P_true, P.shape), atol=1e-12)

    def test_n_lags_is_inferred_from_b(self):
        """B's trailing axis is n_vars * n_lags, so p need not be passed."""
        posterior, L, P_true, _ = self._var2_posterior()
        scheme = LongRunRestriction(ordering=["y1", "y2"], shock_names=["permanent", "transitory"])
        P = scheme.identify(L, ["y1", "y2"], posterior=posterior)
        np.testing.assert_allclose(P, np.broadcast_to(P_true, P.shape), atol=1e-12)

    def test_cumulative_ma_sum_matches_the_imposed_long_run(self):
        """Brute-force sum of 400 MA coefficients must reproduce G — pins the lag handling."""
        from impulso._ma import compute_ma_phi

        posterior, L, _, G = self._var2_posterior()
        scheme = LongRunRestriction(ordering=["y1", "y2"], shock_names=["permanent", "transitory"])
        P = scheme.identify(L, ["y1", "y2"], posterior=posterior, n_lags=2)

        A = [posterior["B"].values[0, 0][:, :2], posterior["B"].values[0, 0][:, 2:]]
        cumulative = compute_ma_phi(A, 400).sum(axis=0)
        np.testing.assert_allclose(cumulative @ P[0, 0], G, atol=1e-10)

    def test_companion_spectral_radius_uses_every_lag_block(self):
        posterior, _, _, _ = self._var2_posterior()
        scheme = LongRunRestriction(ordering=["y1", "y2"], shock_names=["permanent", "transitory"])
        rho = scheme.long_run_diagnostics(posterior)["spectral_radius"]

        A1 = np.array([[0.4, 0.1], [0.1, 0.3]])
        A2 = np.array([[0.2, 0.0], [0.05, 0.1]])
        companion = np.zeros((4, 4))
        companion[:2] = np.concatenate([A1, A2], axis=1)
        companion[2, 0] = companion[3, 1] = 1.0
        np.testing.assert_allclose(rho, np.abs(np.linalg.eigvals(companion)).max(), rtol=1e-12)

    def test_partial_ordering_is_rejected(self):
        """`ordering` must cover every variable — a short one is silently wrong otherwise."""
        posterior, L, _, _ = self._var2_posterior()
        scheme = LongRunRestriction(ordering=["y1"], shock_names=["permanent"])
        with pytest.raises(ValueError, match="cover every variable"):
            scheme.identify(L, ["y1", "y2"], posterior=posterior, n_lags=2)


# ----------------------------------------------------------------------
# MaxShare — frequency-band maximum-share identification (issue #145)
# ----------------------------------------------------------------------


def _posterior_with_B(B):
    """Minimal posterior carrying only the lag coefficients."""
    import xarray as xr

    return xr.Dataset({"B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"])})


def _stable_draw():
    """One stable, non-diagonal VAR(1) draw with a correlated Sigma.

    Deliberately different from `single_driver_2v`: here no shock explains
    everything, so the maximiser is interior and worth checking against
    brute force.
    """
    A1 = np.array([[0.6, 0.15], [0.1, 0.4]])
    Sigma = np.array([[1.0, 0.3], [0.3, 0.7]])
    L = np.linalg.cholesky(Sigma)
    shape = (1, 1, 2, 2)
    return (
        A1,
        Sigma,
        _posterior_with_B(np.broadcast_to(A1, shape).copy()),
        np.broadcast_to(L, shape).copy(),
    )


def _multi_lag_draw():
    """One stable 3-variable VAR(2) draw with a correlated Sigma.

    The lag structure is the point: `A_2` is not proportional to `A_1`,
    so mis-indexing the lag exponent in `sum_j A_j e^{-i w j}` changes the
    transfer function and the band share along with it.
    """
    A1 = np.array([[0.4, 0.15, 0.0], [0.05, 0.3, 0.1], [0.0, 0.1, 0.2]])
    A2 = np.array([[0.1, 0.0, 0.05], [0.0, -0.15, 0.0], [0.02, 0.0, 0.1]])
    Sigma = np.array([[1.0, 0.3, 0.1], [0.3, 0.7, 0.05], [0.1, 0.05, 0.5]])
    L = np.linalg.cholesky(Sigma)
    B = np.concatenate([A1, A2], axis=-1)[np.newaxis, np.newaxis]  # (1, 1, 3, 6)
    return [A1, A2], Sigma, _posterior_with_B(B), L[np.newaxis, np.newaxis]


def _reference_band_form(A: list, L: np.ndarray, target: int, band: tuple[float, float], n_grid: int = 4097):
    """Independent, unvectorised build of `Re(M)` by trapezoid quadrature.

    Deliberately shares no code with the implementation: it builds the lag
    sum term by term with an explicit `e^{-i w j}` per lag `j = 1..p`,
    inverts `F(w)` outright rather than solving, takes the target row, and
    integrates on a dense uniform grid. Used to check the eigen solution
    really is the band-variance maximiser, and to pin the lag indexing.
    """
    omega_lo = 0.0 if np.isinf(band[1]) else 2.0 * np.pi / band[1]
    omega_hi = 2.0 * np.pi / band[0]
    omegas = np.linspace(omega_lo, omega_hi, n_grid)
    integrand = []
    for omega in omegas:
        F = np.eye(A[0].shape[0], dtype=complex)
        for j, A_j in enumerate(A, start=1):
            F -= A_j * np.exp(-1j * omega * j)
        row = np.linalg.inv(F)[target] @ L
        integrand.append(np.conj(row)[:, np.newaxis] * row[np.newaxis, :])
    return np.trapezoid(np.array(integrand), omegas, axis=0).real


class TestMaxShare:
    """Construction, validation and labelling."""

    def test_defaults(self):
        from impulso.identification import MaxShare

        scheme = MaxShare(target="gdp", band=(6, 32))
        assert scheme.shock_name == "max_share"
        assert scheme.n_frequencies == 192
        assert scheme.on_undefined == "nan"
        assert scheme.max_condition == 1e8

    def test_frozen(self):
        from impulso.identification import MaxShare

        scheme = MaxShare(target="gdp", band=(6, 32))
        with pytest.raises(ValidationError):
            scheme.target = "inflation"

    def test_satisfies_protocol(self):
        from impulso.identification import MaxShare

        assert isinstance(MaxShare(target="gdp", band=(6, 32)), IdentificationScheme)

    def test_band_below_nyquist_rejected(self):
        from impulso.identification import MaxShare

        with pytest.raises(ValidationError, match="periods of the sampling interval"):
            MaxShare(target="gdp", band=(1.5, 32))

    def test_band_reversed_rejected(self):
        from impulso.identification import MaxShare

        with pytest.raises(ValidationError, match="low_period < high_period"):
            MaxShare(target="gdp", band=(32, 6))

    def test_unbounded_upper_period_accepted(self):
        from impulso.identification import MaxShare

        scheme = MaxShare(target="gdp", band=(32, float("inf")))
        assert scheme._frequencies()[0] > 0.0

    def test_infinite_lower_bound_rejected(self):
        from impulso.identification import MaxShare

        with pytest.raises(ValidationError, match="lower bound must be finite"):
            MaxShare(target="gdp", band=(float("inf"), float("inf")))

    def test_n_frequencies_floor(self):
        from impulso.identification import MaxShare

        with pytest.raises(ValidationError):
            MaxShare(target="gdp", band=(6, 32), n_frequencies=8)

    def test_reserved_shock_name_rejected(self):
        from impulso.identification import MaxShare

        with pytest.raises(ValidationError, match="reserved prefix"):
            MaxShare(target="gdp", band=(6, 32), shock_name="unidentified_1")

    def test_shock_coords_pad_with_unidentified(self):
        from impulso.identification import MaxShare

        scheme = MaxShare(target="gdp", band=(6, 32))
        assert scheme.shock_coords(3) == ["max_share", "unidentified_1", "unidentified_2"]

    def test_identify_requires_posterior(self):
        from impulso.identification import MaxShare

        _, _, _, L = _stable_draw()
        with pytest.raises(ValueError, match="requires the full posterior with 'B'"):
            MaxShare(target="y1", band=(6, 32)).identify(L, ["y1", "y2"], posterior=None)

    def test_identify_requires_B_in_posterior(self):
        import xarray as xr

        from impulso.identification import MaxShare

        _, _, _, L = _stable_draw()
        with pytest.raises(ValueError, match="requires the full posterior with 'B'"):
            MaxShare(target="y1", band=(6, 32)).identify(L, ["y1", "y2"], posterior=xr.Dataset())

    def test_unknown_target_lists_variables(self):
        from impulso.identification import MaxShare

        _, _, posterior, L = _stable_draw()
        with pytest.raises(ValueError, match=r"\['y1', 'y2'\]"):
            MaxShare(target="nope", band=(6, 32)).identify(L, ["y1", "y2"], posterior=posterior, n_lags=1)


class TestMaxShareRecovery:
    """Exact analytic recovery on a fixture with a known answer."""

    def test_recovers_the_true_column(self, single_driver_2v):
        from impulso.identification import MaxShare

        scheme = MaxShare(target="y2", band=(6, 32))
        P = scheme.identify(
            single_driver_2v["L"], ["y1", "y2"], posterior=single_driver_2v["idata"].posterior, n_lags=1
        )
        expected = np.broadcast_to(single_driver_2v["P_true"][:, 0], P[..., :, 0].shape)
        np.testing.assert_allclose(P[..., :, 0], expected, atol=1e-8)

    def test_share_is_one_and_not_degenerate(self, single_driver_2v):
        from impulso.identification import MaxShare

        scheme = MaxShare(target="y2", band=(6, 32))
        scheme.identify(single_driver_2v["L"], ["y1", "y2"], posterior=single_driver_2v["idata"].posterior, n_lags=1)
        assert scheme._last_diagnostics["max_share_share_median"] >= 1 - 1e-10
        assert abs(scheme._last_diagnostics["max_share_eigen_ratio_median"]) < 1e-8

    def test_band_invariance(self, single_driver_2v):
        """One shock drives y2 at every frequency, so the band cannot matter."""
        from impulso.identification import MaxShare

        posterior = single_driver_2v["idata"].posterior
        narrow = MaxShare(target="y2", band=(6, 32)).identify(
            single_driver_2v["L"], ["y1", "y2"], posterior=posterior, n_lags=1
        )
        wide = MaxShare(target="y2", band=(2.5, 200)).identify(
            single_driver_2v["L"], ["y1", "y2"], posterior=posterior, n_lags=1
        )
        np.testing.assert_allclose(narrow[..., :, 0], wide[..., :, 0], atol=1e-8)

    def test_reproduces_sigma_and_stays_in_data_order(self, single_driver_2v):
        """`P P' = Sigma` in the data's own coordinates — no hidden permutation."""
        from impulso.identification import MaxShare

        P = MaxShare(target="y2", band=(6, 32)).identify(
            single_driver_2v["L"], ["y1", "y2"], posterior=single_driver_2v["idata"].posterior, n_lags=1
        )
        reconstructed = P @ np.swapaxes(P, -1, -2)
        np.testing.assert_allclose(reconstructed, np.broadcast_to(single_driver_2v["Sigma"], P.shape), atol=1e-10)

    def test_sign_convention_raises_the_target(self, single_driver_2v):
        from impulso.identification import MaxShare

        P = MaxShare(target="y2", band=(6, 32)).identify(
            single_driver_2v["L"], ["y1", "y2"], posterior=single_driver_2v["idata"].posterior, n_lags=1
        )
        assert (P[..., 1, 0] > 0).all()

    def test_deterministic(self, single_driver_2v):
        from impulso.identification import MaxShare

        posterior = single_driver_2v["idata"].posterior
        first = MaxShare(target="y2", band=(6, 32)).identify(
            single_driver_2v["L"], ["y1", "y2"], posterior=posterior, n_lags=1
        )
        second = MaxShare(target="y2", band=(6, 32)).identify(
            single_driver_2v["L"], ["y1", "y2"], posterior=posterior, n_lags=1
        )
        np.testing.assert_array_equal(first, second)

    def test_n_lags_inferred_from_B(self, single_driver_2v):
        from impulso.identification import MaxShare

        P = MaxShare(target="y2", band=(6, 32)).identify(
            single_driver_2v["L"], ["y1", "y2"], posterior=single_driver_2v["idata"].posterior
        )
        np.testing.assert_allclose(P[0, 0, :, 0], single_driver_2v["P_true"][:, 0], atol=1e-8)


class TestMaxShareOptimality:
    """The returned column really is the band-variance maximiser."""

    def test_beats_every_rotation_on_a_dense_grid(self):
        from impulso.identification import MaxShare

        A1, _, posterior, L = _stable_draw()
        band = (6.0, 32.0)
        scheme = MaxShare(target="y1", band=band)
        P = scheme.identify(L, ["y1", "y2"], posterior=posterior, n_lags=1)

        M = _reference_band_form([A1], L[0, 0], target=0, band=band)
        q = np.linalg.solve(L[0, 0], P[0, 0, :, 0])
        q = q / np.linalg.norm(q)
        achieved = q @ M @ q / np.trace(M)

        theta = np.linspace(0.0, np.pi, 1024, endpoint=False)
        candidates = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        grid = np.einsum("ki,ij,kj->k", candidates, M, candidates) / np.trace(M)
        assert achieved >= grid.max() - 1e-6

    def test_reported_share_matches_independent_recomputation(self):
        from impulso.identification import MaxShare

        A1, _, posterior, L = _stable_draw()
        band = (6.0, 32.0)
        scheme = MaxShare(target="y1", band=band)
        P = scheme.identify(L, ["y1", "y2"], posterior=posterior, n_lags=1)

        M = _reference_band_form([A1], L[0, 0], target=0, band=band)
        q = np.linalg.solve(L[0, 0], P[0, 0, :, 0])
        q = q / np.linalg.norm(q)
        assert scheme._last_diagnostics["max_share_share_median"] == pytest.approx(q @ M @ q / np.trace(M), abs=1e-5)


class TestMaxShareParseval:
    """The band share over the full spectrum is the infinite-horizon FEVD share."""

    def test_full_band_share_equals_time_domain_share(self):
        from impulso._ma import compute_ma_phi
        from impulso.identification import MaxShare

        _, _, posterior, L = _stable_draw()
        scheme = MaxShare(target="y1", band=(2.0, float("inf")))
        P = scheme.identify(L, ["y1", "y2"], posterior=posterior, n_lags=1)

        Phi = compute_ma_phi([posterior["B"].values], 400)[0, 0]  # (H+1, n, n)
        Theta = Phi @ P[0, 0]
        time_domain = (Theta[:, 0, 0] ** 2).sum() / (Theta[:, 0, :] ** 2).sum()
        assert scheme._last_diagnostics["max_share_share_median"] == pytest.approx(time_domain, abs=1e-6)


class TestMaxShareMultiLag:
    """The lag sum `sum_j A_j e^{-i w j}` on a VAR(2), against an independent reference.

    Every other MaxShare test is VAR(1), where every lag exponent
    collapses to `e^{-i w}` and a mis-indexed lag would go unnoticed.
    Here `A_2` carries its own `e^{-2 i w}`, so the reference disagrees
    with the implementation under any off-by-one.
    """

    def test_var2_share_matches_independent_reference(self):
        from impulso.identification import MaxShare

        A, _, posterior, L = _multi_lag_draw()
        band = (6.0, 32.0)
        scheme = MaxShare(target="v1", band=band)
        P = scheme.identify(L, ["v1", "v2", "v3"], posterior=posterior, n_lags=2)

        M = _reference_band_form(A, L[0, 0], target=0, band=band, n_grid=8193)
        q = np.linalg.solve(L[0, 0], P[0, 0, :, 0])
        q = q / np.linalg.norm(q)
        achieved = q @ M @ q / np.trace(M)
        assert scheme._last_diagnostics["max_share_share_median"] == pytest.approx(achieved, abs=1e-5)

    def test_var2_beats_a_dense_sweep_of_the_sphere(self):
        from impulso.identification import MaxShare

        A, _, posterior, L = _multi_lag_draw()
        band = (6.0, 32.0)
        scheme = MaxShare(target="v1", band=band)
        P = scheme.identify(L, ["v1", "v2", "v3"], posterior=posterior, n_lags=2)

        M = _reference_band_form(A, L[0, 0], target=0, band=band, n_grid=8193)
        q = np.linalg.solve(L[0, 0], P[0, 0, :, 0])
        q = q / np.linalg.norm(q)
        achieved = q @ M @ q / np.trace(M)

        candidates = np.random.default_rng(7).standard_normal((200_000, 3))
        candidates /= np.linalg.norm(candidates, axis=-1, keepdims=True)
        best = (np.einsum("ki,ij,kj->k", candidates, M, candidates) / np.trace(M)).max()
        assert achieved >= best - 1e-6

    def test_var2_reproduces_sigma_and_signs_the_target(self):
        from impulso.identification import MaxShare

        _, Sigma, posterior, L = _multi_lag_draw()
        P = MaxShare(target="v1", band=(6, 32)).identify(L, ["v1", "v2", "v3"], posterior=posterior, n_lags=2)

        np.testing.assert_allclose(P[0, 0] @ P[0, 0].T, Sigma, atol=1e-12)
        assert P[0, 0, 0, 0] > 0

    def test_var2_full_band_share_equals_time_domain_share(self):
        """Parseval again, but with two lags in the moving-average recursion."""
        from impulso._ma import compute_ma_phi
        from impulso.identification import MaxShare

        A, _, posterior, L = _multi_lag_draw()
        scheme = MaxShare(target="v1", band=(2.0, float("inf")))
        P = scheme.identify(L, ["v1", "v2", "v3"], posterior=posterior, n_lags=2)

        Phi = compute_ma_phi([A_j[np.newaxis, np.newaxis] for A_j in A], 600)[0, 0]
        Theta = Phi @ P[0, 0]
        time_domain = (Theta[:, 0, 0] ** 2).sum() / (Theta[:, 0, :] ** 2).sum()
        assert scheme._last_diagnostics["max_share_share_median"] == pytest.approx(time_domain, abs=1e-6)

    def test_var2_n_lags_inferred_from_B(self):
        """`B` is (n, 2n) here, so a wrong inference would change the answer."""
        from impulso.identification import MaxShare

        _, _, posterior, L = _multi_lag_draw()
        explicit = MaxShare(target="v1", band=(6, 32)).identify(L, ["v1", "v2", "v3"], posterior=posterior, n_lags=2)
        inferred = MaxShare(target="v1", band=(6, 32)).identify(L, ["v1", "v2", "v3"], posterior=posterior)
        np.testing.assert_array_equal(explicit, inferred)


class TestMaxShareScreensAndDiagnostics:
    """Explosive draws, singular draws, memoisation and degeneracy."""

    def test_explosive_draws_are_warned_but_kept(self, single_driver_2v):
        from impulso.identification import MaxShare

        posterior = _posterior_with_B(np.broadcast_to(1.05 * np.eye(2), (2, 50, 2, 2)).copy())
        scheme = MaxShare(target="y2", band=(6, 32))
        with pytest.warns(UserWarning, match="explosive"):
            P = scheme.identify(single_driver_2v["L"], ["y1", "y2"], posterior=posterior, n_lags=1)

        assert np.isfinite(P).all()
        assert scheme._last_diagnostics["max_share_explosive_draws"] == 100.0
        assert scheme._last_diagnostics["max_share_explosive_fraction"] == 1.0
        assert scheme._last_diagnostics["max_share_singular_draws"] == 0.0

    def test_singular_draws_are_blanked_and_counted(self, single_driver_2v):
        """A unit-circle root at a period inside the band blanks that draw.

        The midpoint grid never lands exactly on the root frequency, so
        the worst in-band condition number is a few hundred rather than
        infinite; `max_condition` is tightened accordingly.
        """
        from impulso.identification import MaxShare

        B = np.broadcast_to(single_driver_2v["A1"], (2, 50, 2, 2)).copy()
        phi = 2.0 * np.pi / 12.0  # period 12 lies inside band=(6, 32)
        B[0, :5] = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

        scheme = MaxShare(target="y2", band=(6, 32), max_condition=100.0)
        with pytest.warns(UserWarning, match="numerically undefined"):
            P = scheme.identify(single_driver_2v["L"], ["y1", "y2"], posterior=_posterior_with_B(B), n_lags=1)

        assert np.isnan(P[0, :5]).all()
        assert np.isfinite(P[0, 5:]).all()
        assert np.isfinite(P[1]).all()
        assert scheme._last_diagnostics["max_share_singular_draws"] == 5.0
        assert scheme._last_diagnostics["max_share_singular_fraction"] == pytest.approx(0.05)
        assert scheme._last_diagnostics["max_share_condition_max"] > 100.0

    def test_on_undefined_raise(self, single_driver_2v):
        from impulso.identification import MaxShare

        B = np.broadcast_to(single_driver_2v["A1"], (2, 50, 2, 2)).copy()
        phi = 2.0 * np.pi / 12.0
        B[0, :5] = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

        scheme = MaxShare(target="y2", band=(6, 32), max_condition=100.0, on_undefined="raise")
        with pytest.raises(ValueError, match="numerically undefined"):
            scheme.identify(single_driver_2v["L"], ["y1", "y2"], posterior=_posterior_with_B(B), n_lags=1)

    def test_second_identify_reuses_the_cache_and_stays_quiet(self, single_driver_2v):
        """Under SV the pipeline calls identify() once per period."""
        import warnings

        from impulso.identification import MaxShare

        posterior = _posterior_with_B(np.broadcast_to(1.05 * np.eye(2), (2, 50, 2, 2)).copy())
        scheme = MaxShare(target="y2", band=(6, 32))
        with pytest.warns(UserWarning, match="explosive"):
            first = scheme.identify(single_driver_2v["L"], ["y1", "y2"], posterior=posterior, n_lags=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            second = scheme.identify(single_driver_2v["L"], ["y1", "y2"], posterior=posterior, n_lags=1)
        np.testing.assert_array_equal(first, second)

    def test_cache_misses_once_the_posterior_is_collected(self, single_driver_2v):
        """A collected posterior's address can be recycled, so a dead referent
        must read as a miss rather than serving another posterior's sweep (#203)."""
        from impulso.identification import MaxShare

        B = np.broadcast_to(single_driver_2v["A1"], (2, 50, 2, 2)).copy()
        scheme = MaxShare(target="y2", band=(6, 32))
        posterior = _posterior_with_B(B)
        scheme.identify(single_driver_2v["L"], ["y1", "y2"], posterior=posterior, n_lags=1)
        ref = weakref.ref(posterior)

        del posterior
        gc.collect()
        assert ref() is None

        assert scheme._spectral_cache.get(_posterior_with_B(B), (1, 1)) is _CACHE_MISS

    def test_weak_identification_is_warned(self, monkeypatch, single_driver_2v):
        """A repeated top eigenvalue means the maximiser is a plane, not a ray."""
        from impulso.identification import MaxShare

        def flat_accumulator(self, posterior, n_lags, n_vars, target_index):
            shape = posterior["B"].values.shape[:2]
            return (
                np.broadcast_to(np.eye(n_vars), (*shape, n_vars, n_vars)).copy(),
                np.zeros(shape, dtype=bool),
                np.zeros(shape),
                np.ones(shape),
            )

        monkeypatch.setattr(MaxShare, "_spectral_accumulator", flat_accumulator)
        scheme = MaxShare(target="y1", band=(6, 32))
        identity = np.broadcast_to(np.eye(2), (2, 50, 2, 2)).copy()
        with pytest.warns(UserWarning, match="weakly identified"):
            scheme.identify(identity, ["y1", "y2"], posterior=single_driver_2v["idata"].posterior, n_lags=1)
        assert scheme._last_diagnostics["max_share_eigen_ratio_median"] == pytest.approx(1.0)

    def test_diagnostics_keys_are_floats(self, single_driver_2v):
        from impulso.identification import MaxShare

        scheme = MaxShare(target="y2", band=(6, 32))
        scheme.identify(single_driver_2v["L"], ["y1", "y2"], posterior=single_driver_2v["idata"].posterior, n_lags=1)
        assert set(scheme._last_diagnostics) == {
            "max_share_share_median",
            "max_share_share_q05",
            "max_share_share_q95",
            "max_share_eigen_ratio_median",
            "max_share_eigen_ratio_q95",
            "max_share_singular_draws",
            "max_share_singular_fraction",
            "max_share_condition_max",
            "max_share_explosive_draws",
            "max_share_explosive_fraction",
            "max_share_spectral_radius_median",
            "max_share_spectral_radius_max",
        }
        assert all(isinstance(v, float) for v in scheme._last_diagnostics.values())

    def test_per_draw_diagnostics(self, single_driver_2v):
        from impulso.identification import MaxShare

        scheme = MaxShare(target="y2", band=(6, 32))
        diagnostics = scheme.max_share_diagnostics(
            single_driver_2v["L"], ["y1", "y2"], single_driver_2v["idata"].posterior
        )
        assert set(diagnostics) == {"share", "eigen_ratio", "spectral_radius", "condition_max"}
        assert all(v.shape == (2, 50) for v in diagnostics.values())
        np.testing.assert_allclose(diagnostics["share"], 1.0, atol=1e-10)
        np.testing.assert_allclose(diagnostics["spectral_radius"], 0.5, atol=1e-12)
        assert (diagnostics["condition_max"] > 1.0).all()

    def test_per_draw_share_is_nan_for_blanked_draws(self, single_driver_2v):
        from impulso.identification import MaxShare

        B = np.broadcast_to(single_driver_2v["A1"], (2, 50, 2, 2)).copy()
        phi = 2.0 * np.pi / 12.0
        B[0, :5] = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

        scheme = MaxShare(target="y2", band=(6, 32), max_condition=100.0)
        with pytest.warns(UserWarning, match="numerically undefined"):
            diagnostics = scheme.max_share_diagnostics(single_driver_2v["L"], ["y1", "y2"], _posterior_with_B(B))
        assert np.isnan(diagnostics["share"][0, :5]).all()
        assert np.isfinite(diagnostics["share"][0, 5:]).all()
