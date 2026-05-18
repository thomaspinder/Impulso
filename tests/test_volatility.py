"""Tests for the volatility-process seam and adapters."""

import numpy as np
import pytest

from impulso.protocols import VolatilityProcess
from impulso.samplers import NUTSSampler
from impulso.spec import VAR
from impulso.volatility import Constant


class TestConstantAdapter:
    def test_constant_satisfies_protocol(self):
        adapter = Constant()
        assert isinstance(adapter, VolatilityProcess)

    def test_constant_name(self):
        assert Constant().name == "constant"

    def test_constant_is_time_varying_false(self):
        assert Constant().is_time_varying is False

    def test_is_time_varying_settable_per_instance(self):
        # Plain instance field (not ClassVar), so a subclass or wrapper can
        # override at construction time.
        assert Constant(is_time_varying=True).is_time_varying is True

    def test_constant_is_frozen(self):
        from pydantic import ValidationError

        adapter = Constant()
        with pytest.raises(ValidationError):
            # Deliberately violate the `Literal["constant"]` static type to
            # assert that the frozen Pydantic model raises at runtime.
            adapter.name = "other"  # ty: ignore[invalid-assignment]


class TestConstantBuildPymcLatent:
    def test_returns_lower_triangular_for_n_vars_3(self):
        import pymc as pm

        adapter = Constant()
        with pm.Model():
            L_tensor = adapter.build_pymc_latent(n_vars=3, T=100)
            L_value = L_tensor.eval()  # uses prior values; deterministic shape

        assert L_value.shape == (3, 3)
        # Strictly lower-triangular plus positive diagonal: upper triangle is zero.
        upper = np.triu(L_value, k=1)
        assert np.allclose(upper, 0.0)

    def test_handles_n_vars_1(self):
        import pymc as pm

        adapter = Constant()
        with pm.Model():
            L_tensor = adapter.build_pymc_latent(n_vars=1, T=50)
            assert L_tensor.eval().shape == (1, 1)

    def test_registers_expected_pymc_vars(self):
        import pymc as pm

        adapter = Constant()
        with pm.Model() as model:
            adapter.build_pymc_latent(n_vars=3, T=100)

        var_names = {v.name for v in model.unobserved_RVs}
        assert "sigma_sd" in var_names
        assert "tril_offdiag" in var_names

    def test_n_vars_1_skips_tril_offdiag(self):
        import pymc as pm

        adapter = Constant()
        with pm.Model() as model:
            adapter.build_pymc_latent(n_vars=1, T=50)

        var_names = {v.name for v in model.unobserved_RVs}
        assert "sigma_sd" in var_names
        assert "tril_offdiag" not in var_names

    def test_registers_L_deterministic(self):
        """L is exposed as a posterior deterministic so cholesky_at can read it
        without re-decomposing Σ on every call."""
        import pymc as pm

        adapter = Constant()
        with pm.Model() as model:
            adapter.build_pymc_latent(n_vars=3, T=100)

        det_names = {v.name for v in model.deterministics}
        assert "L" in det_names

    def test_accepts_and_ignores_data_kwarg(self):
        """Constant accepts the `data` kwarg for Protocol parity with
        stochastic adapters but ignores it (Σ is data-independent)."""
        import pymc as pm

        adapter = Constant()
        fake_resid = np.zeros((50, 3))
        with pm.Model() as model:
            adapter.build_pymc_latent(n_vars=3, T=50, data=fake_resid)

        assert "L" in {v.name for v in model.deterministics}

    def test_n_vars_2_indexing_and_diagonal(self):
        """Smallest non-trivial off-diagonal case.

        Verifies the single off-diagonal sits at ``L[1, 0]`` (lower-triangular
        cell, not upper) and the diagonal carries the ``sigma_sd`` draws
        (not zeros). A regression that swapped the ``(i, j)`` indexing or
        dropped the ``set_subtensor`` for the diagonal would slip past the
        existing ``n_vars=3`` upper-triangular check.
        """
        import pymc as pm

        adapter = Constant()
        with pm.Model() as model:
            L_tensor = adapter.build_pymc_latent(n_vars=2, T=50)
            L_value, sd_value = pm.draw([L_tensor, model["sigma_sd"]], random_seed=42)

        assert L_value.shape == (2, 2)
        assert L_value[0, 1] == 0.0  # upper-triangular cell stays zero
        assert L_value[1, 0] != 0.0  # off-diagonal placed in the lower-triangular cell
        np.testing.assert_array_equal(np.diag(L_value), sd_value)


class TestPosteriorEquivalence:
    """Sanity check: VAR(lags=1) with default volatility="constant" reproduces
    the *shape* and *variable names* of today's posterior.

    Marked slow because it runs MCMC (cores=1, draws=200, tune=200).
    Intent is regression detection on the seam refactor, not statistical
    correctness — the latter is covered by existing test_fitted.py /
    test_identified.py tests, which we also run to confirm.

    The byte-for-byte equivalence assertion below pins ``nuts_sampler="pymc"``
    explicitly so the determinism contract doesn't depend on whether nutpie
    happens to be installed in the runner's environment. Coverage today is
    only ``lags=1, n_vars=2, no exog``; extending the gate across
    ``(lags, n_vars, exog)`` combinations is tracked as future work.
    """

    @pytest.mark.slow
    def test_default_volatility_posterior_shape_unchanged(self, var_data_2v):
        sampler = NUTSSampler(cores=1, chains=2, draws=200, tune=200, random_seed=42, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)

        posterior_vars = set(fitted.idata.posterior.data_vars)
        # Variables today's pipeline produces. ``L`` is the cached Cholesky
        # factor of Σ — see Constant.cholesky_at.
        expected = {"intercept", "B", "sigma_sd", "tril_offdiag", "L", "Sigma"}
        assert expected <= posterior_vars, f"Missing posterior variables: {expected - posterior_vars}"

        # Shape contracts: (chains, draws, n_vars[, n_vars]).
        n_chains, n_draws = sampler.chains, sampler.draws
        n_vars = var_data_2v.endog.shape[1]
        assert fitted.idata.posterior["intercept"].shape == (n_chains, n_draws, n_vars)
        assert fitted.idata.posterior["B"].shape == (n_chains, n_draws, n_vars, n_vars)
        assert fitted.idata.posterior["Sigma"].shape == (n_chains, n_draws, n_vars, n_vars)
        # Sigma is symmetric per draw (positive-definiteness is enforced upstream
        # by HalfCauchy(sigma_sd) > 0 and the MvNormal likelihood).
        sigma = fitted.idata.posterior["Sigma"].values
        assert np.allclose(sigma, np.swapaxes(sigma, -1, -2)), "Sigma not symmetric"

    @pytest.mark.slow
    def test_explicit_constant_matches_string_default(self, var_data_2v):
        """VAR(lags=1, volatility="constant") and VAR(lags=1, volatility=Constant())
        and VAR(lags=1) all produce identical posteriors given the same seed."""

        def _fit(spec_kwargs):
            sampler = NUTSSampler(cores=1, chains=2, draws=100, tune=100, random_seed=42, nuts_sampler="pymc")
            return VAR(lags=1, **spec_kwargs).fit(var_data_2v, sampler=sampler)

        fit_default = _fit({})
        fit_string = _fit({"volatility": "constant"})
        fit_object = _fit({"volatility": Constant()})

        for name in ["intercept", "B", "L", "Sigma"]:
            np.testing.assert_array_equal(
                fit_default.idata.posterior[name].values,
                fit_string.idata.posterior[name].values,
                err_msg=f"default vs string disagree on {name}",
            )
            np.testing.assert_array_equal(
                fit_default.idata.posterior[name].values,
                fit_object.idata.posterior[name].values,
                err_msg=f"default vs Constant() instance disagree on {name}",
            )


class TestConstantCholeskyAt:
    def test_returns_cholesky_of_sigma(self, synthetic_idata_2v):
        adapter = Constant()
        L = adapter.cholesky_at(synthetic_idata_2v.posterior, t=None)

        sigma = synthetic_idata_2v.posterior["Sigma"].values
        # L @ L.T should reproduce Sigma.
        reconstructed = np.einsum("cdij,cdkj->cdik", L, L)
        np.testing.assert_allclose(reconstructed, sigma, rtol=1e-6)

    def test_t_argument_ignored_for_constant(self, synthetic_idata_2v):
        """For constant volatility, any value of t returns the same L."""
        adapter = Constant()
        L_none = adapter.cholesky_at(synthetic_idata_2v.posterior, t=None)
        L_zero = adapter.cholesky_at(synthetic_idata_2v.posterior, t=0)
        L_arbitrary = adapter.cholesky_at(synthetic_idata_2v.posterior, t=42)
        np.testing.assert_array_equal(L_none, L_zero)
        np.testing.assert_array_equal(L_none, L_arbitrary)

    def test_returns_lower_triangular(self, synthetic_idata_2v):
        adapter = Constant()
        L = adapter.cholesky_at(synthetic_idata_2v.posterior, t=None)
        # Strictly upper-triangular block must be zero.
        upper = np.triu(L, k=1)
        assert np.allclose(upper, 0.0)

    def test_reads_cached_L_not_decomposed_sigma(self):
        """cholesky_at must read posterior["L"] directly, not recompute
        chol(Σ). Pinned by making L and Σ inconsistent in the fixture and
        asserting the returned array equals L exactly."""
        import arviz as az
        import xarray as xr

        # L is a fixed lower-triangular factor; Σ is something *different*
        # (identity). A correct implementation returns L. A re-decomposing
        # implementation returns chol(I) = I, which would differ.
        L_truth = np.array([[2.0, 0.0], [0.5, 1.5]])
        L_arr = np.broadcast_to(L_truth, (1, 1, 2, 2)).copy()
        sigma_arr = np.broadcast_to(np.eye(2), (1, 1, 2, 2)).copy()
        posterior = xr.Dataset({
            "L": xr.DataArray(L_arr, dims=["chain", "draw", "var1", "var2"]),
            "Sigma": xr.DataArray(sigma_arr, dims=["chain", "draw", "var1", "var2"]),
        })
        idata = az.InferenceData(posterior=posterior)

        out = Constant().cholesky_at(idata.posterior, t=None)
        np.testing.assert_array_equal(out, L_arr)


class TestConstantForecastCholeskyPath:
    def test_returns_broadcast_shape(self, synthetic_idata_2v):
        adapter = Constant()
        rng = np.random.default_rng(0)
        path = adapter.forecast_cholesky_path(synthetic_idata_2v.posterior, steps=5, rng=rng)
        # (chains, draws, steps, n_vars, n_vars)
        assert path.shape == (2, 50, 5, 2, 2)

    def test_constant_across_steps(self, synthetic_idata_2v):
        """For constant volatility, every forecast step has the same L."""
        adapter = Constant()
        rng = np.random.default_rng(0)
        path = adapter.forecast_cholesky_path(synthetic_idata_2v.posterior, steps=10, rng=rng)
        # path[..., 0, :, :] must equal path[..., k, :, :] for all k.
        np.testing.assert_array_equal(path[..., 0, :, :], path[..., 9, :, :])

    def test_rng_unused_for_constant(self, synthetic_idata_2v):
        """Constant is deterministic given the posterior — rng is accepted but unused."""
        adapter = Constant()
        path_a = adapter.forecast_cholesky_path(synthetic_idata_2v.posterior, steps=3, rng=np.random.default_rng(0))
        path_b = adapter.forecast_cholesky_path(synthetic_idata_2v.posterior, steps=3, rng=np.random.default_rng(99999))
        np.testing.assert_array_equal(path_a, path_b)


class TestConstantCholeskyPath:
    def test_returns_broadcast_shape_for_T(self, synthetic_idata_2v):
        adapter = Constant()
        path = adapter.cholesky_path(synthetic_idata_2v.posterior, T=10)
        # (chains, draws, T, n_vars, n_vars)
        assert path.shape == (2, 50, 10, 2, 2)

    def test_constant_across_time(self, synthetic_idata_2v):
        adapter = Constant()
        path = adapter.cholesky_path(synthetic_idata_2v.posterior, T=5)
        np.testing.assert_array_equal(path[..., 0, :, :], path[..., 4, :, :])
