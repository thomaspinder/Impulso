"""Direct unit tests for the shared lag-recursion engine (`impulso._propagate`).

Covers the observable contracts of `propagate` and `propagate_contributions`:
closed-form VAR(1) decay, hand-rolled p=2 recursion cross-checks, y_init
broadcasting semantics, the reproduce-the-sample identity, linearity and
additivity of shock contributions, and the ValueError modes.
"""

import numpy as np
import pytest

from impulso._propagate import propagate, propagate_contributions


def _diag_lag(a: np.ndarray) -> np.ndarray:
    """Build a (1, 1, n, n) diagonal lag matrix from diagonal entries `a`."""
    return np.diag(a)[None, None, :, :]


class TestPropagateClosedForm:
    def test_var1_diagonal_decays_geometrically(self):
        """With A_1 = diag(a) and a t=0 impulse only, y_t = a**t * impulse."""
        a = np.array([0.5, -0.8])
        A = [_diag_lag(a)]
        T, n = 6, 2
        impulse = np.array([1.0, 2.0])
        forcing = np.zeros((1, 1, T, n))
        forcing[0, 0, 0, :] = impulse
        y_init = np.zeros((1, n))

        path = propagate(A, forcing, y_init)

        assert path.shape == (1, 1, T, n)
        for t in range(T):
            np.testing.assert_allclose(path[0, 0, t, :], a**t * impulse, atol=1e-12)


class TestPropagateRecursionCrossCheck:
    def test_matches_hand_rolled_p2_recursion(self):
        """propagate agrees with an explicit Python loop for p=2, (C=1, D=2)."""
        rng = np.random.default_rng(42)
        C, D, T, n, p = 1, 2, 5, 2, 2
        A = [rng.normal(scale=0.3, size=(C, D, n, n)) for _ in range(p)]
        forcing = rng.normal(size=(C, D, T, n))
        y_init = rng.normal(size=(C, D, p, n))

        expected = np.empty((C, D, T, n))
        for c in range(C):
            for d in range(D):
                # Chronological history: hist[-1] immediately precedes t=0.
                hist = [y_init[c, d, k, :] for k in range(p)]
                for t in range(T):
                    y_t = forcing[c, d, t, :].copy()
                    for i in range(p):
                        y_t = y_t + A[i][c, d] @ hist[-1 - i]
                    hist.append(y_t)
                    expected[c, d, t, :] = y_t

        np.testing.assert_allclose(propagate(A, forcing, y_init), expected, atol=1e-12)


class TestPropagateYInit:
    def test_broadcast_and_explicit_y_init_agree(self):
        """(p, n) y_init tiled to (C, D, p, n) yields identical output."""
        rng = np.random.default_rng(42)
        C, D, T, n, p = 2, 3, 4, 2, 1
        A = [rng.normal(scale=0.3, size=(C, D, n, n))]
        forcing = rng.normal(size=(C, D, T, n))
        y_init = rng.normal(size=(p, n))
        y_init_explicit = np.broadcast_to(y_init, (C, D, p, n)).copy()

        np.testing.assert_allclose(
            propagate(A, forcing, y_init),
            propagate(A, forcing, y_init_explicit),
            atol=1e-12,
        )

    def test_nonzero_y_init_enters_recursion(self):
        """A nonzero y_init changes the path from the very first step."""
        a = np.array([0.9, 0.9])
        A = [_diag_lag(a)]
        T, n = 3, 2
        forcing = np.zeros((1, 1, T, n))
        y0 = np.array([[1.0, -2.0]])

        path = propagate(A, forcing, y0)
        zero_path = propagate(A, forcing, np.zeros((1, n)))

        # y_1 = A_1 @ y_0, then geometric continuation.
        np.testing.assert_allclose(path[0, 0, 0, :], a * y0[0], atol=1e-12)
        np.testing.assert_allclose(path[0, 0, 2, :], a**3 * y0[0], atol=1e-12)
        np.testing.assert_allclose(zero_path, np.zeros((1, 1, T, n)), atol=0)
        assert np.abs(path - zero_path).max() > 0.1


class TestPropagateReproducesSample:
    def test_reproduces_synthetic_var1_path(self):
        """propagate returns exactly a path built as y_t = f_t + A_1 y_{t-1}."""
        rng = np.random.default_rng(42)
        C, D, T, n = 1, 1, 8, 2
        A1 = rng.normal(scale=0.4, size=(C, D, n, n))
        forcing = rng.normal(size=(C, D, T, n))
        y0 = rng.normal(size=(1, n))

        y = np.empty((C, D, T, n))
        prev = y0[0]
        for t in range(T):
            y[0, 0, t, :] = forcing[0, 0, t, :] + A1[0, 0] @ prev
            prev = y[0, 0, t, :]

        np.testing.assert_allclose(propagate([A1], forcing, y0), y, atol=1e-12)


class TestPropagateErrors:
    def test_empty_lag_list_raises(self):
        with pytest.raises(ValueError, match="at least one lag coefficient matrix"):
            propagate([], np.zeros((1, 1, 3, 2)), np.zeros((1, 2)))

    def test_non_4d_forcing_raises(self):
        A = [np.zeros((1, 1, 2, 2))]
        with pytest.raises(ValueError, match="forcing must be 4-D"):
            propagate(A, np.zeros((1, 3, 2)), np.zeros((1, 2)))

    def test_wrong_y_init_trailing_shape_raises(self):
        A = [np.zeros((1, 1, 2, 2))]
        with pytest.raises(ValueError, match="y_init trailing shape"):
            propagate(A, np.zeros((1, 1, 3, 2)), np.zeros((2, 2)))


class TestPropagateContributions:
    def test_zero_impact_returns_zeros(self):
        """Zero impact propagates to exactly zero from zero initial conditions."""
        rng = np.random.default_rng(42)
        A = [rng.normal(scale=0.3, size=(1, 1, 2, 2))]
        impact = np.zeros((1, 1, 4, 2, 3))

        out = propagate_contributions(A, impact)

        assert out.shape == impact.shape
        np.testing.assert_allclose(out, np.zeros_like(impact), atol=0)

    def test_linearity_in_impact(self):
        """Contributions of impact1 + impact2 equal the sum of separate calls."""
        rng = np.random.default_rng(42)
        C, D, T, n, S = 1, 2, 5, 2, 2
        A = [rng.normal(scale=0.3, size=(C, D, n, n)) for _ in range(2)]
        impact1 = rng.normal(size=(C, D, T, n, S))
        impact2 = rng.normal(size=(C, D, T, n, S))

        np.testing.assert_allclose(
            propagate_contributions(A, impact1 + impact2),
            propagate_contributions(A, impact1) + propagate_contributions(A, impact2),
            atol=1e-12,
        )

    def test_shock_sum_matches_propagate_from_zero_init(self):
        """Summing contributions over shocks equals propagate of the summed
        impact from zero initial conditions."""
        rng = np.random.default_rng(42)
        C, D, T, n, S, p = 1, 2, 6, 2, 3, 2
        A = [rng.normal(scale=0.3, size=(C, D, n, n)) for _ in range(p)]
        impact = rng.normal(size=(C, D, T, n, S))

        summed = propagate_contributions(A, impact).sum(axis=-1)
        direct = propagate(A, impact.sum(axis=-1), np.zeros((p, n)))

        np.testing.assert_allclose(summed, direct, atol=1e-12)


class TestPropagateContributionsErrors:
    def test_empty_lag_list_raises(self):
        with pytest.raises(ValueError, match="at least one lag coefficient matrix"):
            propagate_contributions([], np.zeros((1, 1, 3, 2, 1)))

    def test_non_5d_impact_raises(self):
        A = [np.zeros((1, 1, 2, 2))]
        with pytest.raises(ValueError, match="impact must be 5-D"):
            propagate_contributions(A, np.zeros((1, 1, 3, 2)))
