"""Tests for the companion-form stability primitives."""

import numpy as np
import pytest

from impulso._stability import companion_eigenvalues, companion_matrix, spectral_radius


class TestCompanionMatrix:
    def test_shape_single_draw(self):
        B = np.zeros((3, 6))
        assert companion_matrix(B, n_lags=2).shape == (6, 6)

    def test_shape_batched(self):
        B = np.zeros((4, 50, 3, 9))
        assert companion_matrix(B, n_lags=3).shape == (4, 50, 9, 9)

    def test_top_block_is_B_verbatim(self):
        rng = np.random.default_rng(0)
        B = rng.standard_normal((2, 6))
        F = companion_matrix(B, n_lags=3)
        np.testing.assert_array_equal(F[:2, :], B)

    def test_subdiagonal_is_identity(self):
        B = np.zeros((2, 6))
        F = companion_matrix(B, n_lags=3)
        np.testing.assert_array_equal(F[2:, :4], np.eye(4))

    def test_trailing_block_column_is_zero_below_top(self):
        B = np.ones((2, 6))
        F = companion_matrix(B, n_lags=3)
        np.testing.assert_array_equal(F[2:, 4:], np.zeros((4, 2)))

    def test_single_lag_is_B_itself(self):
        rng = np.random.default_rng(1)
        B = rng.standard_normal((3, 3))
        np.testing.assert_array_equal(companion_matrix(B, n_lags=1), B)

    def test_batch_agnostic(self):
        rng = np.random.default_rng(2)
        B = rng.standard_normal((2, 5, 3, 6))
        batched = companion_matrix(B, n_lags=2)
        for c in range(2):
            for d in range(5):
                np.testing.assert_array_equal(batched[c, d], companion_matrix(B[c, d], n_lags=2))

    def test_rejects_zero_lags(self):
        with pytest.raises(ValueError, match="n_lags must be positive"):
            companion_matrix(np.zeros((2, 4)), n_lags=0)

    def test_rejects_indivisible_trailing_axis(self):
        with pytest.raises(ValueError, match="not divisible"):
            companion_matrix(np.zeros((2, 5)), n_lags=2)

    def test_rejects_non_matching_row_count(self):
        with pytest.raises(ValueError, match="rows"):
            companion_matrix(np.zeros((3, 4)), n_lags=2)

    def test_rejects_one_dimensional_input(self):
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            companion_matrix(np.zeros(4), n_lags=1)


class TestCompanionEigenvalues:
    def test_known_scalar_ar2_roots(self):
        # Scalar AR(2): y_t = 0.5 y_{t-1} + 0.2 y_{t-2}. The companion
        # eigenvalues are the roots of lambda^2 - 0.5 lambda - 0.2.
        B = np.array([[0.5, 0.2]])
        eigs = companion_eigenvalues(B, n_lags=2)
        expected = np.roots([1.0, -0.5, -0.2])
        np.testing.assert_allclose(np.sort(eigs.real), np.sort(expected), atol=1e-12)
        np.testing.assert_allclose(np.abs(eigs).max(), 0.7623475, atol=1e-6)

    def test_diagonal_var1_eigenvalues_are_diagonal_entries(self):
        B = np.diag([0.9, 0.3])
        eigs = companion_eigenvalues(B, n_lags=1)
        np.testing.assert_allclose(np.sort(eigs.real), [0.3, 0.9], atol=1e-12)

    def test_complex_case_returns_complex_dtype_and_real_radius(self):
        # A rotation-like block has a conjugate pair of complex eigenvalues.
        B = np.array([[0.0, -0.8], [0.8, 0.0]])
        eigs = companion_eigenvalues(B, n_lags=1)
        assert np.iscomplexobj(eigs)
        assert np.abs(eigs.imag).max() > 0.5
        radius = spectral_radius(B, n_lags=1)
        assert not np.iscomplexobj(radius)
        np.testing.assert_allclose(radius, 0.8, atol=1e-12)

    def test_chunking_does_not_change_result(self):
        rng = np.random.default_rng(3)
        B = rng.standard_normal((3, 7, 2, 4)) * 0.3
        small = spectral_radius(B, n_lags=2, chunk_size=1)
        large = spectral_radius(B, n_lags=2, chunk_size=10_000)
        np.testing.assert_allclose(small, large)

    def test_rejects_non_positive_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            companion_eigenvalues(np.zeros((2, 2)), n_lags=1, chunk_size=0)


class TestSpectralRadius:
    def test_matches_naive_loop(self):
        rng = np.random.default_rng(4)
        B = rng.standard_normal((2, 6, 3, 6)) * 0.2
        vectorised = spectral_radius(B, n_lags=2)
        naive = np.zeros((2, 6))
        for c in range(2):
            for d in range(6):
                F = companion_matrix(B[c, d], n_lags=2)
                naive[c, d] = np.abs(np.linalg.eigvals(F)).max()
        np.testing.assert_allclose(vectorised, naive)

    def test_batch_shape_preserved(self):
        B = np.zeros((5, 11, 2, 4))
        assert spectral_radius(B, n_lags=2).shape == (5, 11)

    def test_single_draw_returns_scalar(self):
        radius = spectral_radius(np.diag([0.5, 0.5]), n_lags=1)
        assert radius.shape == ()
        np.testing.assert_allclose(float(radius), 0.5)

    def test_explosive_draw_detected(self):
        radius = spectral_radius(np.diag([1.2, 0.5]), n_lags=1)
        assert float(radius) > 1.0
