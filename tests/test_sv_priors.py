"""Tests for SV priors."""

import numpy as np
import pytest
from pydantic import ValidationError

from impulso.sv.priors import SVDefaultPrior, SVPrior


def test_sv_default_prior_shape():
    prior = SVDefaultPrior()
    y = np.array([0.1, 0.2, -0.1, 0.05, -0.02, 0.15])
    params = prior.build_priors(y)
    # Required keys for random walk + optional AR(1) keys
    assert "mu_mu" in params
    assert "mu_sigma" in params
    assert "h0_mu" in params
    assert "h0_sigma" in params
    assert "sigma_eta_scale" in params
    assert "phi_a" in params
    assert "phi_b" in params
    assert "alpha_mu" in params
    assert "alpha_sigma" in params


def test_sv_default_prior_values_depend_on_data():
    prior = SVDefaultPrior()
    y1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * 100
    params1 = prior.build_priors(y1)
    params2 = prior.build_priors(y2)
    # mu_sigma scales with sample std; h0_mu = log(sample variance)
    assert params2["mu_sigma"] > params1["mu_sigma"]
    assert params2["h0_mu"] > params1["h0_mu"]


def test_sv_default_prior_is_svprior():
    assert isinstance(SVDefaultPrior(), SVPrior)


def test_sv_default_prior_custom_scales():
    prior = SVDefaultPrior(mu_scale=1.0, sigma_eta_scale=0.5)
    y = np.array([1.0, 2.0, 3.0])
    params = prior.build_priors(y)
    assert params["sigma_eta_scale"] == 0.5


def test_sv_default_prior_frozen():
    prior = SVDefaultPrior()
    with pytest.raises(ValidationError):  # pydantic frozen error
        prior.mu_scale = 999.0


def test_sv_default_prior_mu_sigma_floor_when_near_constant():
    """Floor kicks in for near-constant input so mu_sigma stays valid."""
    prior = SVDefaultPrior()
    y = np.full(30, 1.0) + 1e-15 * np.arange(30)  # numerically near-constant
    params = prior.build_priors(y)
    assert params["mu_sigma"] > 0.0
    assert np.isfinite(params["mu_sigma"])
    assert params["mu_sigma"] >= 1e-6


@pytest.mark.parametrize(
    "y",
    [
        np.ones(30),  # exactly constant
        np.array([1.0] * 29 + [1.0 + 1e-16]),  # ulp-level perturbation
        np.full(30, 1e-300),  # underflow-scale values
    ],
)
def test_sv_default_prior_mu_sigma_positive_finite_for_any_input(y):
    """mu_sigma must be strictly positive and finite for any finite y."""
    prior = SVDefaultPrior()
    params = prior.build_priors(y)
    assert params["mu_sigma"] > 0.0
    assert np.isfinite(params["mu_sigma"])
