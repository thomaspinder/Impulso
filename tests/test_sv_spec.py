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
    assert fitted.dynamics == "random_walk"
