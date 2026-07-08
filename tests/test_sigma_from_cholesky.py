"""Tests for sigma_from_cholesky helper and Clark reconstruction home.

Equality pins: these tests lock down the numerical output of the
existing per-site einsums and Clark reconstructions so the refactor
to the shared helpers is provably behaviour-preserving.

Testing rule: public interfaces only — no model_construct, no
reaching through .idata internals.
"""

import numpy as np


class TestSigmaFromCholesky:
    """sigma_from_cholesky(L) = L @ L.T over leading batch dims."""

    def test_4d_matches_einsum(self, rng):
        """4-D input (chains, draws, n, n) matches the hand-written einsum."""
        from impulso._linalg import sigma_from_cholesky

        L = rng.standard_normal((3, 50, 2, 2))
        L = np.tril(L)
        L[:, :, range(2), range(2)] = np.abs(L[:, :, range(2), range(2)]) + 0.1

        result = sigma_from_cholesky(L)
        expected = np.einsum("cdij,cdkj->cdik", L, L)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_5d_matches_einsum(self, rng):
        """5-D input (chains, draws, T, n, n) matches the hand-written einsum."""
        from impulso._linalg import sigma_from_cholesky

        L = rng.standard_normal((2, 20, 10, 3, 3))
        L = np.tril(L)
        L[:, :, :, range(3), range(3)] = np.abs(L[:, :, :, range(3), range(3)]) + 0.1

        result = sigma_from_cholesky(L)
        expected = np.einsum("cdtij,cdtkj->cdtik", L, L)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_output_is_symmetric(self, rng):
        """Σ = L @ L.T is always symmetric."""
        from impulso._linalg import sigma_from_cholesky

        L = np.tril(rng.standard_normal((2, 10, 4, 4)))
        sigma = sigma_from_cholesky(L)
        np.testing.assert_allclose(sigma, np.swapaxes(sigma, -1, -2), atol=1e-12)

    def test_output_is_positive_definite(self, rng):
        """Σ = L @ L.T is PD when L has positive diagonal."""
        from impulso._linalg import sigma_from_cholesky

        L = np.tril(rng.standard_normal((2, 10, 3, 3)))
        L[:, :, range(3), range(3)] = np.abs(L[:, :, range(3), range(3)]) + 0.5
        sigma = sigma_from_cholesky(L)
        eigvals = np.linalg.eigvalsh(sigma)
        assert np.all(eigvals > 0)

    def test_identity_cholesky_gives_identity(self):
        """L = I → Σ = I."""
        from impulso._linalg import sigma_from_cholesky

        L = np.eye(3)
        result = sigma_from_cholesky(L)
        np.testing.assert_allclose(result, np.eye(3), atol=1e-12)


class TestClarkReconstructionSingleHome:
    """All three Clark callers (cholesky_at, cholesky_path,
    forecast_cholesky_path) produce lower-triangular L_t consistent
    with each other, after the refactor to the shared private helper.

    Equality pins compare against manual ``exp(h/2) * R_chol``
    computed from the fixture posterior, so a wrong shared helper
    cannot hide behind all three callers sharing the same bug.
    """

    def test_cholesky_at_equals_manual_exp_h_half_R_chol(self, synthetic_sv_idata_2v):
        """Pin: cholesky_at(t) == diag(exp(h_t/2)) @ R_chol, computed manually."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        posterior = synthetic_sv_idata_2v.posterior
        h = posterior["h"].values  # (2, 50, 20, 2)
        R_chol = posterior["R_chol"].values  # (2, 50, 2, 2)

        for t in [0, 7, 19]:
            L_t = sv.cholesky_at(posterior, t=t)
            expected = np.exp(h[:, :, t, :] / 2)[:, :, :, None] * R_chol
            np.testing.assert_allclose(L_t, expected, rtol=1e-12, err_msg=f"cholesky_at({t}) != manual")

    def test_cholesky_path_equals_manual_exp_h_half_R_chol(self, synthetic_sv_idata_2v):
        """Pin: cholesky_path == diag(exp(h/2)) @ R_chol broadcast over T."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        posterior = synthetic_sv_idata_2v.posterior
        h = posterior["h"].values  # (2, 50, 20, 2)
        R_chol = posterior["R_chol"].values  # (2, 50, 2, 2)

        path = sv.cholesky_path(posterior, T=20)
        sigma_t = np.exp(h / 2)  # (2, 50, 20, 2)
        expected = sigma_t[:, :, :, :, None] * R_chol[:, :, None, :, :]
        np.testing.assert_allclose(path, expected, rtol=1e-12)

    def test_cholesky_at_is_lower_triangular(self, synthetic_sv_idata_2v):
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        for t in [0, 5, 19]:
            L_t = sv.cholesky_at(synthetic_sv_idata_2v.posterior, t=t)
            assert L_t.shape == (2, 50, 2, 2)
            assert np.allclose(np.triu(L_t, k=1), 0.0)

    def test_cholesky_path_slices_match_cholesky_at(self, synthetic_sv_idata_2v):
        """Each time slice of cholesky_path must equal cholesky_at(t)."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        path = sv.cholesky_path(synthetic_sv_idata_2v.posterior, T=20)
        assert path.shape == (2, 50, 20, 2, 2)

        for t in range(20):
            L_at = sv.cholesky_at(synthetic_sv_idata_2v.posterior, t=t)
            np.testing.assert_allclose(path[:, :, t, :, :], L_at, rtol=1e-12)

    def test_cholesky_path_is_lower_triangular_per_slice(self, synthetic_sv_idata_2v):
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        path = sv.cholesky_path(synthetic_sv_idata_2v.posterior, T=20)
        for t in range(20):
            assert np.allclose(np.triu(path[:, :, t, :, :], k=1), 0.0)

    def test_forecast_cholesky_path_is_lower_triangular(self, synthetic_sv_idata_2v):
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        rng = np.random.default_rng(99)
        path = sv.forecast_cholesky_path(
            synthetic_sv_idata_2v.posterior,
            steps=5,
            rng=rng,
        )
        assert path.shape == (2, 50, 5, 2, 2)
        for s in range(5):
            assert np.allclose(np.triu(path[:, :, s, :, :], k=1), 0.0)

    def test_forecast_cholesky_path_extends_beyond_sample(self, synthetic_sv_idata_2v):
        """Forecast L_t values must differ from in-sample last-period L_t
        (with high probability under RW dynamics), confirming the Clark
        reconstruction is called on extrapolated h, not a broadcast."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        rng = np.random.default_rng(99)
        forecast_path = sv.forecast_cholesky_path(
            synthetic_sv_idata_2v.posterior,
            steps=5,
            rng=rng,
        )
        last_in_sample = sv.cholesky_at(synthetic_sv_idata_2v.posterior, t=None)
        # At least one forecast step should differ from the last in-sample slice.
        differs = [not np.allclose(forecast_path[:, :, s, :, :], last_in_sample, atol=1e-10) for s in range(5)]
        assert any(differs), "Forecast path is identical to last in-sample — Clark not called on extrapolated h"

    def test_forecast_cholesky_path_reproducible_with_rng(self, synthetic_sv_idata_2v):
        """Same rng seed → identical forecast path."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        path_a = sv.forecast_cholesky_path(
            synthetic_sv_idata_2v.posterior,
            steps=3,
            rng=np.random.default_rng(42),
        )
        path_b = sv.forecast_cholesky_path(
            synthetic_sv_idata_2v.posterior,
            steps=3,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(path_a, path_b)
