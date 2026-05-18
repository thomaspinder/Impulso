"""Tests for the MA coefficient recursion helper."""

import numpy as np
import pytest

from impulso._ma import compute_ma_phi


class TestComputeMaPhiSingleDraw:
    """A_j arrays have shape (n, n) — the per-rotation case used by
    `SignRestriction._check_restrictions_at_horizons`.
    """

    def test_horizon_zero_returns_identity(self):
        """Phi_0 is the identity, regardless of A. Returned shape is (1, n, n)."""
        n = 3
        rng = np.random.default_rng(0)
        A = [rng.standard_normal((n, n))]
        Phi = compute_ma_phi(A, horizon=0)
        assert Phi.shape == (1, n, n)
        np.testing.assert_array_equal(Phi[0], np.eye(n))

    def test_phi_one_equals_a_one(self):
        """Phi_1 = sum_{j=1}^{min(1, p)} A_j @ Phi_0 = A_1 @ I = A_1."""
        rng = np.random.default_rng(1)
        n = 3
        A_1 = rng.standard_normal((n, n))
        Phi = compute_ma_phi([A_1, rng.standard_normal((n, n))], horizon=1)
        assert Phi.shape == (2, n, n)
        np.testing.assert_allclose(Phi[1], A_1)

    def test_recursion_matches_manual_two_lags_three_horizons(self):
        """Closed-form check: Phi_2 = A_1 @ Phi_1 + A_2 @ Phi_0 = A_1 @ A_1 + A_2."""
        rng = np.random.default_rng(2)
        n = 2
        A_1 = rng.standard_normal((n, n))
        A_2 = rng.standard_normal((n, n))
        Phi = compute_ma_phi([A_1, A_2], horizon=3)
        assert Phi.shape == (4, n, n)
        np.testing.assert_allclose(Phi[0], np.eye(n))
        np.testing.assert_allclose(Phi[1], A_1)
        np.testing.assert_allclose(Phi[2], A_1 @ A_1 + A_2)
        np.testing.assert_allclose(Phi[3], A_1 @ (A_1 @ A_1 + A_2) + A_2 @ A_1)

    def test_horizon_less_than_n_lags_truncates_sum(self):
        """At h=1 with p=3, only A_1 contributes (j=1..min(1, 3)=1)."""
        rng = np.random.default_rng(3)
        n = 2
        A_1 = rng.standard_normal((n, n))
        A_2 = rng.standard_normal((n, n))
        A_3 = rng.standard_normal((n, n))
        Phi = compute_ma_phi([A_1, A_2, A_3], horizon=1)
        # A_2 and A_3 must NOT contribute to Phi_1; perturbing them is invisible.
        Phi_perturbed = compute_ma_phi([A_1, A_2 * 100, A_3 * -50], horizon=1)
        np.testing.assert_allclose(Phi[1], Phi_perturbed[1])
        np.testing.assert_allclose(Phi[1], A_1)


class TestComputeMaPhiBatched:
    """A_j arrays have shape (C, D, n, n) — the posterior case used by
    `IdentifiedVAR._ma_coefficients`.
    """

    def test_returns_expected_shape(self):
        rng = np.random.default_rng(10)
        n, n_chains, n_draws, n_lags, horizon = 3, 2, 5, 2, 4
        A = [rng.standard_normal((n_chains, n_draws, n, n)) for _ in range(n_lags)]
        Phi = compute_ma_phi(A, horizon=horizon)
        assert Phi.shape == (n_chains, n_draws, horizon + 1, n, n)

    def test_batched_matches_per_draw_loop(self):
        """Each (c, d) slice of the batched result must equal the single-draw
        recursion on the same coefficients — guarantees `@` broadcasting is
        semantically equivalent to scalar matmul.
        """
        rng = np.random.default_rng(11)
        n, n_chains, n_draws, n_lags, horizon = 3, 2, 4, 2, 3
        A_batched = [rng.standard_normal((n_chains, n_draws, n, n)) for _ in range(n_lags)]
        Phi_batched = compute_ma_phi(A_batched, horizon=horizon)

        for c in range(n_chains):
            for d in range(n_draws):
                A_single = [A_j[c, d] for A_j in A_batched]
                Phi_single = compute_ma_phi(A_single, horizon=horizon)
                np.testing.assert_allclose(Phi_batched[c, d], Phi_single)

    def test_horizon_zero_returns_identity_broadcast(self):
        rng = np.random.default_rng(12)
        n, n_chains, n_draws = 2, 2, 3
        A = [rng.standard_normal((n_chains, n_draws, n, n))]
        Phi = compute_ma_phi(A, horizon=0)
        assert Phi.shape == (n_chains, n_draws, 1, n, n)
        for c in range(n_chains):
            for d in range(n_draws):
                np.testing.assert_array_equal(Phi[c, d, 0], np.eye(n))


class TestComputeMaPhiAgreesWithLegacyRecursion:
    """Numerical agreement gate: the helper output must match the original
    pre-refactor recursion in `identified.py` on a fixed seed.
    """

    @staticmethod
    def _legacy_recursion(B_draws: np.ndarray, n_vars: int, n_lags: int, horizon: int) -> np.ndarray:
        """Verbatim copy of the pre-refactor `_ma_coefficients` body."""
        A = [B_draws[:, :, :, j * n_vars : (j + 1) * n_vars] for j in range(n_lags)]
        n_chains, n_draws = B_draws.shape[:2]
        Phi = [np.broadcast_to(np.eye(n_vars), (n_chains, n_draws, n_vars, n_vars)).copy()]
        for h in range(1, horizon + 1):
            phi_h = np.zeros((n_chains, n_draws, n_vars, n_vars))
            for j in range(min(h, n_lags)):
                phi_h += np.einsum("cdij,cdjk->cdik", A[j], Phi[h - j - 1])
            Phi.append(phi_h)
        return np.stack(Phi, axis=2)

    def test_helper_matches_legacy_on_fixed_seed(self):
        rng = np.random.default_rng(42)
        n_chains, n_draws, n_vars, n_lags, horizon = 2, 30, 3, 2, 8
        B_draws = rng.standard_normal((n_chains, n_draws, n_vars, n_vars * n_lags)) * 0.3

        legacy = self._legacy_recursion(B_draws, n_vars, n_lags, horizon)
        A = [B_draws[:, :, :, j * n_vars : (j + 1) * n_vars] for j in range(n_lags)]
        helper = compute_ma_phi(A, horizon)

        np.testing.assert_allclose(helper, legacy, atol=1e-12)


class TestComputeMaPhiValidation:
    def test_empty_A_raises(self):
        with pytest.raises(ValueError, match="at least one lag"):
            compute_ma_phi([], horizon=1)

    def test_negative_horizon_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            compute_ma_phi([np.eye(2)], horizon=-1)
