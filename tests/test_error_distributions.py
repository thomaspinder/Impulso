"""Tests for the observation error-distribution seam (issue #152).

All fast — no MCMC. The innovation-draw tests deliberately avoid sample
moments that do not exist or converge slowly: sample kurtosis has infinite
variance for nu <= 8 and is useless as a test statistic (it measures ~4.4
against a theoretical 6.0 even at 160k draws). Exact-distribution and
algebraic-identity checks are used instead.
"""

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError
from scipy import stats

from impulso.observation import Gaussian, StudentT
from impulso.protocols import ErrorDistribution


def _posterior(nu, shape=(2, 50)):
    """Minimal posterior Dataset carrying `nu` over (chain, draw)."""
    arr = np.full(shape, nu, dtype=float) if np.isscalar(nu) else np.asarray(nu, dtype=float)
    return xr.Dataset({"nu": xr.DataArray(arr, dims=["chain", "draw"])})


class TestConfiguration:
    def test_student_t_defaults_to_inferred_nu(self):
        assert StudentT().nu == "infer"

    def test_student_t_accepts_fixed_nu(self):
        assert StudentT(nu=5.0).nu == 5.0

    @pytest.mark.parametrize("bad_nu", [2.0, 1.5, 0.5, -1.0])
    def test_student_t_rejects_nu_at_or_below_two(self, bad_nu):
        with pytest.raises(ValidationError, match="must be > 2"):
            StudentT(nu=bad_nu)

    def test_student_t_accepts_nu_just_above_two(self):
        assert StudentT(nu=2.0001).nu == pytest.approx(2.0001)

    @pytest.mark.parametrize(("field", "value"), [("prior_alpha", 0.0), ("prior_beta", -1.0)])
    def test_non_positive_prior_hyperparameters_rejected(self, field, value):
        with pytest.raises(ValidationError):
            StudentT(**{field: value})

    @pytest.mark.parametrize("adapter", [Gaussian(), StudentT(nu=5.0)])
    def test_frozen(self, adapter):
        with pytest.raises(ValidationError):
            adapter.name = "something_else"

    def test_is_heavy_tailed_flags(self):
        assert Gaussian().is_heavy_tailed is False
        assert StudentT().is_heavy_tailed is True

    def test_discriminator_mismatch_rejected(self):
        with pytest.raises(ValidationError):
            StudentT(name="gaussian")
        with pytest.raises(ValidationError):
            Gaussian(name="student_t")

    def test_both_adapters_satisfy_the_protocol(self):
        assert isinstance(Gaussian(), ErrorDistribution)
        assert isinstance(StudentT(), ErrorDistribution)


class TestGaussianInnovations:
    def test_stream_is_bit_identical_to_plain_standard_normal(self):
        """The RNG contract: Gaussian must consume exactly one standard_normal."""
        shape = (2, 50, 3)
        expected = np.random.default_rng(7).standard_normal(shape)
        actual = Gaussian().draw_standardised_innovations(shape, np.random.default_rng(7), _posterior(5.0))
        np.testing.assert_array_equal(actual, expected)

    def test_generator_left_in_the_same_state(self):
        """Nothing is consumed after the standard normals."""
        rng_a = np.random.default_rng(3)
        Gaussian().draw_standardised_innovations((2, 10, 2), rng_a, _posterior(5.0, (2, 10)))
        rng_b = np.random.default_rng(3)
        rng_b.standard_normal((2, 10, 2))
        assert rng_a.standard_normal() == rng_b.standard_normal()

    def test_variance_inflation_is_one(self):
        assert Gaussian().variance_inflation(_posterior(5.0)) == 1.0


class TestStudentTInnovations:
    def test_marginals_are_exactly_standard_t(self):
        """KS test against t_nu — an exact-distribution check, not a moment."""
        nu = 5.0
        shape = (4, 50_000, 2)
        xi = StudentT(nu=nu).draw_standardised_innovations(shape, np.random.default_rng(0), _posterior(nu, shape[:2]))
        assert stats.kstest(xi[..., 0].ravel(), "t", args=(nu,)).pvalue > 0.01

    def test_mixing_variable_is_shared_across_components(self):
        """The load-bearing identity: xi_0/xi_1 == z_0/z_1, exactly.

        One chi-square per *observation vector* means the mixing factor
        cancels in the ratio of two components. A naive implementation that
        draws one chi-square per component (giving independent t marginals
        rather than a multivariate t) fails this.
        """
        shape = (2, 500, 3)
        z = np.random.default_rng(11).standard_normal(shape)  # matched seed
        xi = StudentT(nu=6.0).draw_standardised_innovations(
            shape, np.random.default_rng(11), _posterior(6.0, shape[:2])
        )
        np.testing.assert_allclose(xi[..., 0] / xi[..., 1], z[..., 0] / z[..., 1], rtol=1e-12)
        np.testing.assert_allclose(xi[..., 2] / xi[..., 1], z[..., 2] / z[..., 1], rtol=1e-12)

    def test_standard_normal_is_consumed_first(self):
        """RNG contract: the t branch's normals match the Gaussian branch's."""
        shape = (2, 200, 2)
        gauss = Gaussian().draw_standardised_innovations(shape, np.random.default_rng(5), _posterior(5.0, shape[:2]))
        t = StudentT(nu=5.0).draw_standardised_innovations(shape, np.random.default_rng(5), _posterior(5.0, shape[:2]))
        # xi = z / sqrt(g/nu) with g > 0, so sign(xi) == sign(z) elementwise.
        np.testing.assert_array_equal(np.sign(t), np.sign(gauss))

    def test_tails_are_fatter_than_gaussian(self):
        """Quantile ratio at 99.9%: t_5/N is ~1.91 in the population."""
        shape = (4, 50_000, 2)
        nu = 5.0
        xi = StudentT(nu=nu).draw_standardised_innovations(shape, np.random.default_rng(2), _posterior(nu, shape[:2]))
        z = np.random.default_rng(2).standard_normal(shape)
        ratio = np.quantile(xi, 0.999) / np.quantile(z, 0.999)
        assert 1.5 < ratio < 2.5

    def test_variance_matches_nu_over_nu_minus_two(self):
        nu = 8.0  # nu=8 keeps the sample variance's own variance finite-ish
        shape = (4, 100_000, 2)
        xi = StudentT(nu=nu).draw_standardised_innovations(shape, np.random.default_rng(4), _posterior(nu, shape[:2]))
        assert xi.var() == pytest.approx(nu / (nu - 2), rel=0.05)

    def test_mean_is_zero(self):
        shape = (4, 50_000, 2)
        xi = StudentT(nu=5.0).draw_standardised_innovations(shape, np.random.default_rng(9), _posterior(5.0, shape[:2]))
        assert abs(xi.mean()) < 0.05

    def test_large_nu_converges_to_the_normal(self):
        shape = (4, 20_000, 2)
        xi = StudentT(nu=1e6).draw_standardised_innovations(shape, np.random.default_rng(1), _posterior(1e6, shape[:2]))
        assert stats.kstest(xi[..., 0].ravel(), "norm").pvalue > 0.01

    def test_nu_varies_across_draws(self):
        """A (chain, draw) nu array gives per-draw tail weight."""
        n_chains, n_draws = 2, 20_000
        nu = np.empty((n_chains, n_draws))
        nu[:, : n_draws // 2] = 3.0
        nu[:, n_draws // 2 :] = 50.0
        xi = StudentT().draw_standardised_innovations((n_chains, n_draws, 2), np.random.default_rng(6), _posterior(nu))
        assert xi.shape == (n_chains, n_draws, 2)
        assert np.isfinite(xi).all()
        fat = np.quantile(np.abs(xi[:, : n_draws // 2]), 0.999)
        thin = np.quantile(np.abs(xi[:, n_draws // 2 :]), 0.999)
        assert thin < fat

    def test_missing_nu_raises(self):
        with pytest.raises(ValueError, match="no 'nu' variable"):
            StudentT(nu=5.0).draw_standardised_innovations((2, 5, 2), np.random.default_rng(0), xr.Dataset())

    def test_mismatched_nu_shape_raises(self):
        with pytest.raises(ValueError, match="need nu of shape"):
            StudentT(nu=5.0).draw_standardised_innovations(
                (2, 5, 2), np.random.default_rng(0), _posterior(5.0, (2, 50))
            )


class TestVarianceInflation:
    def test_student_t_inflation_is_nu_over_nu_minus_two(self):
        nu = np.array([[3.0, 4.0], [6.0, 10.0]])
        got = StudentT().variance_inflation(_posterior(nu))
        np.testing.assert_allclose(got, np.array([[3.0, 2.0], [1.5, 1.25]]))

    def test_inflation_shape_matches_chain_draw(self):
        got = StudentT(nu=5.0).variance_inflation(_posterior(5.0, (3, 17)))
        assert got.shape == (3, 17)

    def test_inflation_missing_nu_raises(self):
        with pytest.raises(ValueError, match="no 'nu' variable"):
            StudentT(nu=5.0).variance_inflation(xr.Dataset())
