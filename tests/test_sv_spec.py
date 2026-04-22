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
