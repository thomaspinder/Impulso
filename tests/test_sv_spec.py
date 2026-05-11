"""Tests for the StochasticVolatility spec."""

import numpy as np
import pytest
from pydantic import ValidationError

from impulso.sv.priors import SVDefaultPrior
from impulso.sv.spec import StochasticVolatility


def test_sv_spec_defaults():
    sv = StochasticVolatility()
    assert sv.dynamics == "random_walk"
    assert sv.prior == "default"


def test_sv_spec_resolved_prior_from_string():
    sv = StochasticVolatility(prior="default")
    resolved = sv.resolved_prior
    assert isinstance(resolved, SVDefaultPrior)


def test_sv_spec_resolved_prior_passes_through_object():
    custom = SVDefaultPrior(sigma_eta_scale=0.05)
    sv = StochasticVolatility(prior=custom)
    assert sv.resolved_prior is custom


def test_sv_spec_invalid_dynamics_rejected():
    with pytest.raises(ValidationError):
        StochasticVolatility(dynamics="invalid")


def test_sv_spec_resolved_dynamics_from_string():
    from impulso.sv.dynamics import AR1, RandomWalk

    assert isinstance(StochasticVolatility(dynamics="random_walk").resolved_dynamics, RandomWalk)
    assert isinstance(StochasticVolatility(dynamics="ar1").resolved_dynamics, AR1)


def test_sv_spec_resolved_dynamics_passes_through_object():
    from impulso.sv.dynamics import AR1

    custom = AR1()
    sv = StochasticVolatility(dynamics=custom)
    assert sv.resolved_dynamics is custom


def test_sv_default_sampler_is_safe():
    """Default sampler must pin cores=1 and use a conservative target_accept."""
    from impulso.samplers import NUTSSampler

    sampler = StochasticVolatility._default_sampler()
    assert isinstance(sampler, NUTSSampler)
    assert sampler.cores == 1
    assert sampler.target_accept >= 0.9


@pytest.mark.slow
def test_sv_spec_fit_random_walk_smoke():
    """Smoke test: fit() runs end-to-end with tiny MCMC."""
    import pandas as pd

    from impulso.samplers import NUTSSampler
    from impulso.sv.data import SVData
    from impulso.sv.fitted import FittedSV

    rng = np.random.default_rng(0)
    T = 80
    y = 0.1 * rng.standard_normal(T)
    index = pd.date_range("2000-01-01", periods=T, freq="MS")
    data = SVData(y=y, name="sim", index=index)

    sampler = NUTSSampler(draws=30, tune=30, chains=1, cores=1, random_seed=1)
    fitted = StochasticVolatility(dynamics="random_walk").fit(data, sampler=sampler)

    assert isinstance(fitted, FittedSV)
    assert fitted.log_volatility.shape == (1, 30, T)
    assert fitted.dynamics.name == "random_walk"


def test_sv_spec_build_ar1_model():
    """AR(1) branch builds a valid PyMC model without running MCMC."""
    import pandas as pd
    import pymc as pm

    from impulso.sv.data import SVData

    rng = np.random.default_rng(0)
    T = 30
    y = rng.standard_normal(T)
    index = pd.date_range("2000-01-01", periods=T, freq="MS")
    data = SVData(y=y, name="sim", index=index)

    sv = StochasticVolatility(dynamics="ar1")
    prior_params = sv.resolved_prior.build_priors(data.y)
    model = sv._build_pymc_model(data.y, prior_params, sv.resolved_dynamics)

    assert isinstance(model, pm.Model)
    assert "phi" in model.named_vars
    assert "alpha" in model.named_vars


@pytest.mark.slow
def test_sv_spec_fit_ar1_smoke():
    """Smoke test: AR(1) fit runs end-to-end."""
    import pandas as pd

    from impulso.samplers import NUTSSampler
    from impulso.sv.data import SVData

    rng = np.random.default_rng(1)
    T = 80
    y = 0.1 * rng.standard_normal(T)
    index = pd.date_range("2000-01-01", periods=T, freq="MS")
    data = SVData(y=y, name="sim", index=index)

    sampler = NUTSSampler(draws=30, tune=30, chains=1, cores=1, random_seed=2)
    fitted = StochasticVolatility(dynamics="ar1").fit(data, sampler=sampler)

    assert fitted.dynamics.name == "ar1"
    assert "phi" in fitted.idata.posterior
    assert "alpha" in fitted.idata.posterior


class TestSVDynamicsDiscriminator:
    """Locks in the Literal[...] discriminator semantics for the SV dynamics
    adapters (RandomWalk, AR1) — see CONTEXT.md "Conventions" and the
    project-wide convention adopted in the volatility-process P1 work.
    """

    def test_random_walk_rejects_wrong_name_at_construction(self):
        from impulso.sv.dynamics import RandomWalk

        with pytest.raises(ValidationError):
            RandomWalk(name="ar1")

    def test_ar1_rejects_wrong_name_at_construction(self):
        from impulso.sv.dynamics import AR1

        with pytest.raises(ValidationError):
            AR1(name="random_walk")

    def test_random_walk_name_is_frozen(self):
        from impulso.sv.dynamics import RandomWalk

        rw = RandomWalk()
        with pytest.raises(ValidationError):
            rw.name = "ar1"  # ty: ignore[invalid-assignment]

    def test_ar1_name_is_frozen(self):
        from impulso.sv.dynamics import AR1

        ar = AR1()
        with pytest.raises(ValidationError):
            ar.name = "random_walk"  # ty: ignore[invalid-assignment]


class TestStochasticVolatilityIsVolatilityProcess:
    def test_satisfies_protocol(self):
        """SV is a VolatilityProcess at runtime (after stub methods land)."""
        from impulso.protocols import VolatilityProcess
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        assert isinstance(sv, VolatilityProcess)

    def test_name_is_sv(self):
        """Discriminator field for the registry."""
        from impulso.sv.spec import StochasticVolatility

        assert StochasticVolatility().name == "sv"

    def test_name_is_frozen(self):
        from pydantic import ValidationError

        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        with pytest.raises(ValidationError):
            sv.name = "constant"  # ty: ignore[invalid-assignment]


class TestSVMultivariateBuild:
    def test_returns_3d_cholesky_factor(self):
        """SV.build_pymc_latent returns L of shape (T, n_vars, n_vars)."""
        import pymc as pm

        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility(dynamics="random_walk")
        with pm.Model():
            L = sv.build_pymc_latent(n_vars=3, T=50)
            L_value = L.eval()

        assert L_value.shape == (50, 3, 3)

    def test_registers_per_variable_log_vol_paths(self):
        """Multivariate SV registers per-variable log-vol paths (v{i}_h) plus a stacked h."""
        import pymc as pm

        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility(dynamics="random_walk")
        with pm.Model() as model:
            sv.build_pymc_latent(n_vars=2, T=50)

        rv_names = {v.name for v in model.unobserved_RVs}
        det_names = {v.name for v in model.deterministics}
        # Stacked path registered as a Deterministic.
        assert "h" in det_names
        # Per-variable latent paths registered as Deterministics
        # (RandomWalk.build_latent_path wraps in pm.Deterministic).
        assert "v0_h" in det_names
        assert "v1_h" in det_names
        # Per-variable hyperparameters registered as RVs.
        assert "v0_mu" in rv_names and "v1_mu" in rv_names
        assert "v0_sigma_eta" in rv_names and "v1_sigma_eta" in rv_names

    def test_registers_correlation_cholesky(self):
        """The constant correlation Cholesky R_chol must be in the model."""
        import pymc as pm

        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility(dynamics="random_walk")
        with pm.Model() as model:
            sv.build_pymc_latent(n_vars=3, T=50)

        var_names = {v.name for v in model.unobserved_RVs} | {v.name for v in model.deterministics}
        assert "R_chol" in var_names or "R_chol_offdiag" in var_names


class TestSVCholeskyAt:
    def test_returns_correct_shape(self, synthetic_sv_idata_2v):
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        L_last = sv.cholesky_at(synthetic_sv_idata_2v.posterior, t=None)
        assert L_last.shape == (2, 50, 2, 2)

    def test_t_indexes_into_time(self, synthetic_sv_idata_2v):
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        L_0 = sv.cholesky_at(synthetic_sv_idata_2v.posterior, t=0)
        L_5 = sv.cholesky_at(synthetic_sv_idata_2v.posterior, t=5)
        assert not np.allclose(L_0, L_5)

    def test_t_none_defaults_to_last(self, synthetic_sv_idata_2v):
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        T = synthetic_sv_idata_2v.posterior["h"].shape[2]
        L_none = sv.cholesky_at(synthetic_sv_idata_2v.posterior, t=None)
        L_last = sv.cholesky_at(synthetic_sv_idata_2v.posterior, t=T - 1)
        np.testing.assert_array_equal(L_none, L_last)

    def test_diagonal_equals_exp_half_h(self, synthetic_sv_idata_2v):
        """L's diagonal must equal exp(h_t / 2) — the core math identity."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        posterior = synthetic_sv_idata_2v.posterior
        h = posterior["h"].values  # (C, D, T, n)
        L = sv.cholesky_at(posterior, t=3)
        np.testing.assert_allclose(
            np.diagonal(L, axis1=-2, axis2=-1),
            np.exp(h[:, :, 3, :] / 2),
        )

    def test_upper_triangle_is_zero(self, synthetic_sv_idata_2v):
        """L must be lower-triangular."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        L = sv.cholesky_at(synthetic_sv_idata_2v.posterior, t=3)
        # Check zero upper triangle for the first (chain, draw) draw.
        np.testing.assert_array_equal(np.triu(L[0, 0], 1), 0.0)

    def test_negative_t_raises(self, synthetic_sv_idata_2v):
        """Negative t must raise ValueError, not silently wrap around."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        with pytest.raises(ValueError, match="out of range"):
            sv.cholesky_at(synthetic_sv_idata_2v.posterior, t=-1)

    def test_out_of_range_t_raises(self, synthetic_sv_idata_2v):
        """t >= T must raise ValueError."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        T = synthetic_sv_idata_2v.posterior["h"].shape[2]
        with pytest.raises(ValueError, match="out of range"):
            sv.cholesky_at(synthetic_sv_idata_2v.posterior, t=T)


class TestSVCholeskyPath:
    def test_returns_full_path(self, synthetic_sv_idata_2v):
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        path = sv.cholesky_path(synthetic_sv_idata_2v.posterior, T=20)
        assert path.shape == (2, 50, 20, 2, 2)

    def test_path_matches_cholesky_at_per_t(self, synthetic_sv_idata_2v):
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        path = sv.cholesky_path(synthetic_sv_idata_2v.posterior, T=20)
        L_3 = sv.cholesky_at(synthetic_sv_idata_2v.posterior, t=3)
        np.testing.assert_array_equal(path[:, :, 3, :, :], L_3)

    def test_path_diagonal_equals_exp_half_h(self, synthetic_sv_idata_2v):
        """Across all t, diag(L_t) must equal exp(h_t / 2) — vectorised check."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        posterior = synthetic_sv_idata_2v.posterior
        h = posterior["h"].values  # (C, D, T, n)
        T = h.shape[2]
        path = sv.cholesky_path(posterior, T=T)
        # path: (C, D, T, n, n) → diagonals along last two axes: (C, D, T, n)
        diagonals = np.diagonal(path, axis1=-2, axis2=-1)
        np.testing.assert_allclose(diagonals, np.exp(h / 2))

    def test_path_upper_triangle_is_zero(self, synthetic_sv_idata_2v):
        """Each L_t in the path must be lower-triangular."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility()
        path = sv.cholesky_path(synthetic_sv_idata_2v.posterior, T=20)
        # Check first (chain, draw, t) slice.
        np.testing.assert_array_equal(np.triu(path[0, 0, 7], 1), 0.0)


class TestSVForecastCholeskyPath:
    def test_returns_correct_shape(self, synthetic_sv_idata_2v):
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility(dynamics="random_walk")
        rng = np.random.default_rng(0)
        path = sv.forecast_cholesky_path(synthetic_sv_idata_2v.posterior, steps=10, rng=rng)
        # (chains, draws, steps, n_vars, n_vars)
        assert path.shape == (2, 50, 10, 2, 2)

    def test_random_walk_forecast_shape(self, synthetic_sv_idata_2v):
        """Random-walk dynamics produce a forecast path with the expected shape.

        AR1 path shape is exercised in the integration tests, since it requires
        ``phi``/``alpha`` in the posterior that the synthetic fixture does not
        provide.
        """
        from impulso.sv.spec import StochasticVolatility

        rng = np.random.default_rng(0)
        path_rw = StochasticVolatility(dynamics="random_walk").forecast_cholesky_path(
            synthetic_sv_idata_2v.posterior, steps=5, rng=rng
        )
        assert path_rw.shape == (2, 50, 5, 2, 2)

    def test_output_is_lower_triangular(self, synthetic_sv_idata_2v):
        """Forecast Cholesky factors must be lower-triangular at every step."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility(dynamics="random_walk")
        path = sv.forecast_cholesky_path(synthetic_sv_idata_2v.posterior, steps=10, rng=np.random.default_rng(0))
        # Check each forecast step's upper triangle is zero (use first chain/draw).
        for t in range(path.shape[2]):
            np.testing.assert_array_equal(np.triu(path[0, 0, t], 1), 0.0)

    def test_diagonal_is_strictly_positive(self, synthetic_sv_idata_2v):
        """Cholesky diagonals must be strictly positive (diag = exp(h_forecast / 2) > 0)."""
        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility(dynamics="random_walk")
        path = sv.forecast_cholesky_path(synthetic_sv_idata_2v.posterior, steps=5, rng=np.random.default_rng(0))
        diags = np.diagonal(path, axis1=-2, axis2=-1)  # (C, D, steps, n_vars)
        assert np.all(diags > 0)
        assert not np.any(np.isnan(diags))
