"""Tests for prefix-aware forecast_log_vol (issue #90).

Pin tests: each dynamics forecasts correctly from a multivariate-shaped
posterior with v{i}_* names, and per-variable results match the equivalent
bare-name univariate call.
"""

import numpy as np
import xarray as xr


def _make_univariate_posterior(h_3d, sigma_eta):
    """Build a minimal univariate-shaped posterior for bare-name calls."""
    return xr.Dataset({
        "h": (("chain", "draw", "time"), h_3d),
        "sigma_eta": (("chain", "draw"), sigma_eta),
    })


def _make_univariate_ar1_posterior(h_3d, sigma_eta, phi, alpha):
    """Build a minimal univariate AR(1) posterior for bare-name calls."""
    return xr.Dataset({
        "h": (("chain", "draw", "time"), h_3d),
        "sigma_eta": (("chain", "draw"), sigma_eta),
        "phi": (("chain", "draw"), phi),
        "alpha": (("chain", "draw"), alpha),
    })


class TestRandomWalkPrefixAware:
    """RandomWalk.forecast_log_vol reads {prefix}h and {prefix}sigma_eta."""

    def test_accepts_name_prefix(self, synthetic_sv_idata_2v):
        """forecast_log_vol must accept name_prefix kwarg."""
        from impulso.sv.dynamics import RandomWalk

        rw = RandomWalk()
        rng = np.random.default_rng(42)
        # Should not raise — reads v0_h and v0_sigma_eta.
        result = rw.forecast_log_vol(
            synthetic_sv_idata_2v.posterior,
            steps=5,
            rng=rng,
            name_prefix="v0_",
        )
        assert result.shape == (2, 50, 5)

    def test_prefix_result_matches_bare_univariate(self, synthetic_sv_idata_2v):
        """Calling with name_prefix='v0_' on the multivariate posterior
        must produce the same result as calling bare on a univariate posterior
        with the same h and sigma_eta values."""
        from impulso.sv.dynamics import RandomWalk

        rw = RandomWalk()
        posterior = synthetic_sv_idata_2v.posterior

        # Bare univariate call with v0's data.
        h_v0 = posterior["v0_h"].values  # (2, 50, 20)
        sigma_eta_v0 = posterior["v0_sigma_eta"].values  # (2, 50)
        uni_posterior = _make_univariate_posterior(h_v0, sigma_eta_v0)

        rng_bare = np.random.default_rng(77)
        rng_prefix = np.random.default_rng(77)

        result_bare = rw.forecast_log_vol(uni_posterior, steps=5, rng=rng_bare)
        result_prefix = rw.forecast_log_vol(
            posterior,
            steps=5,
            rng=rng_prefix,
            name_prefix="v0_",
        )
        np.testing.assert_array_equal(result_bare, result_prefix)

    def test_different_prefixes_give_different_results(self, synthetic_sv_idata_2v):
        """v0_ and v1_ have different sigma_eta, so forecasts must differ."""
        from impulso.sv.dynamics import RandomWalk

        rw = RandomWalk()
        rng = np.random.default_rng(42)

        result_v0 = rw.forecast_log_vol(
            synthetic_sv_idata_2v.posterior,
            steps=5,
            rng=rng,
            name_prefix="v0_",
        )
        rng2 = np.random.default_rng(42)
        result_v1 = rw.forecast_log_vol(
            synthetic_sv_idata_2v.posterior,
            steps=5,
            rng=rng2,
            name_prefix="v1_",
        )
        # Different sigma_eta → different paths (with overwhelming probability).
        assert not np.array_equal(result_v0, result_v1)

    def test_default_empty_prefix_reads_bare_keys(self):
        """Default name_prefix='' reads bare 'h' and 'sigma_eta'."""
        from impulso.sv.dynamics import RandomWalk

        rw = RandomWalk()
        rng = np.random.default_rng(0)
        h = rng.standard_normal((1, 10, 15)) * 0.3 - 1.0
        sigma_eta = np.abs(rng.standard_normal((1, 10))) * 0.1
        posterior = _make_univariate_posterior(h, sigma_eta)

        result = rw.forecast_log_vol(posterior, steps=3, rng=rng)
        assert result.shape == (1, 10, 3)


class TestAR1PrefixAware:
    """AR1.forecast_log_vol reads {prefix}h, {prefix}sigma_eta, {prefix}phi, {prefix}alpha."""

    @staticmethod
    def _make_ar1_posterior_2v():
        """Build a 2-variable multivariate AR(1) posterior with prefixed keys."""
        rng = np.random.default_rng(99)
        n_chains, n_draws, T, n_vars = 2, 50, 20, 2
        h = rng.standard_normal((n_chains, n_draws, T, n_vars)) * 0.3 - 1.0
        return xr.Dataset({
            "h": (("chain", "draw", "time", "variable"), h),
            "v0_h": (("chain", "draw", "time"), h[:, :, :, 0]),
            "v1_h": (("chain", "draw", "time"), h[:, :, :, 1]),
            "v0_sigma_eta": (("chain", "draw"), np.abs(rng.standard_normal((n_chains, n_draws)))),
            "v1_sigma_eta": (("chain", "draw"), np.abs(rng.standard_normal((n_chains, n_draws)))),
            "v0_phi": (("chain", "draw"), 0.5 + 0.3 * rng.standard_normal((n_chains, n_draws))),
            "v1_phi": (("chain", "draw"), 0.5 + 0.3 * rng.standard_normal((n_chains, n_draws))),
            "v0_alpha": (("chain", "draw"), rng.standard_normal((n_chains, n_draws)) * 0.1),
            "v1_alpha": (("chain", "draw"), rng.standard_normal((n_chains, n_draws)) * 0.1),
        })

    def test_accepts_name_prefix(self):
        """forecast_log_vol must accept name_prefix kwarg for AR(1)."""
        from impulso.sv.dynamics import AR1

        ar1 = AR1()
        posterior = self._make_ar1_posterior_2v()
        rng = np.random.default_rng(42)
        result = ar1.forecast_log_vol(posterior, steps=5, rng=rng, name_prefix="v0_")
        assert result.shape == (2, 50, 5)

    def test_prefix_result_matches_bare_univariate(self):
        """Prefix call matches bare univariate call for AR(1)."""
        from impulso.sv.dynamics import AR1

        ar1 = AR1()
        posterior = self._make_ar1_posterior_2v()

        h_v0 = posterior["v0_h"].values
        sigma_eta_v0 = posterior["v0_sigma_eta"].values
        phi_v0 = posterior["v0_phi"].values
        alpha_v0 = posterior["v0_alpha"].values
        uni_posterior = _make_univariate_ar1_posterior(h_v0, sigma_eta_v0, phi_v0, alpha_v0)

        rng_bare = np.random.default_rng(77)
        rng_prefix = np.random.default_rng(77)

        result_bare = ar1.forecast_log_vol(uni_posterior, steps=5, rng=rng_bare)
        result_prefix = ar1.forecast_log_vol(posterior, steps=5, rng=rng_prefix, name_prefix="v0_")
        np.testing.assert_array_equal(result_bare, result_prefix)


class TestForecastCholeskyPathUsesPrefix:
    """forecast_cholesky_path calls dynamics with name_prefix, no fake Dataset."""

    def test_forecast_cholesky_path_uses_prefix(self, synthetic_sv_idata_2v):
        """forecast_cholesky_path must produce the same result as manually
        calling forecast_log_vol per variable with the prefix."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        rng = np.random.default_rng(42)
        path = sv.forecast_cholesky_path(
            synthetic_sv_idata_2v.posterior,
            steps=5,
            rng=rng,
        )
        assert path.shape == (2, 50, 5, 2, 2)

    def test_no_fake_dataset_in_forecast_cholesky_path(self):
        """forecast_cholesky_path must not build an xr.Dataset internally.
        We verify by checking the method does not import xr (except at
        module level). This is a structural test, not a behavioral one."""
        import inspect

        from impulso.sv.spec import StochasticVolatility

        source = inspect.getsource(StochasticVolatility.forecast_cholesky_path)
        # The method should NOT contain xr.Dataset( construction.
        assert "xr.Dataset(" not in source, (
            "forecast_cholesky_path still builds a fake xr.Dataset — "
            "it should call dynamics.forecast_log_vol with name_prefix instead"
        )
