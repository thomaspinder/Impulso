"""Tests for the StochasticVolatility spec."""

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
    import numpy as np
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
    import numpy as np
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
    import numpy as np
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
        """For n_vars=3, expect h_0, h_1, h_2 (or h with shape (T, 3))."""
        import pymc as pm

        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility(dynamics="random_walk")
        with pm.Model() as model:
            sv.build_pymc_latent(n_vars=2, T=50)

        var_names = {v.name for v in model.unobserved_RVs} | {v.name for v in model.deterministics}
        # Either separate h_0/h_1 or a stacked h with shape (T, 2) - pick one in implementation.
        assert "h_0" in var_names or "h" in var_names

    def test_registers_correlation_cholesky(self):
        """The constant correlation Cholesky R_chol must be in the model."""
        import pymc as pm

        from impulso.sv.spec import StochasticVolatility

        sv = StochasticVolatility(dynamics="random_walk")
        with pm.Model() as model:
            sv.build_pymc_latent(n_vars=3, T=50)

        var_names = {v.name for v in model.unobserved_RVs} | {v.name for v in model.deterministics}
        assert "R_chol" in var_names or "R_chol_offdiag" in var_names
