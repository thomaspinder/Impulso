"""Tests for the deterministic volatility break (BREAK node).

Gates the Lenza-Primiceri (2020) COVID break adapter:

1. `log_scales` / `s_t` path values: 1 before `t*`, the three sampled scales at
   `t*..t*+2`, and `1 + (s_may - 1) * rho**(j - 2)` along the decay.
2. Query methods return the correct `(chains, draws, [T|steps], n, n)` shapes,
   are lower-triangular and finite, and equal `s_t * L_base`.
3. `hyperparameter_priors()` yields three Pareto(1, 1) scale priors and one Beta
   decay prior with the expected parameters and a working `.logpdf`.
4. `is_time_varying` is `True`.
"""

import numpy as np
import pytest
import xarray as xr
from scipy import stats

from impulso.conjugate_volatility import PandemicBreak, Prior1D
from impulso.protocols import VolatilityProcess

START = 4
THETA = {"s_march": 5.0, "s_april": 3.0, "s_may": 2.0, "rho": 0.8}


@pytest.fixture
def pandemic():
    return PandemicBreak(start=START)


@pytest.fixture
def posterior():
    """Fabricated posterior: base Cholesky `L` plus scale-hyperparameter draws."""
    rng = np.random.default_rng(0)
    n_chains, n_draws, n = 2, 5, 3

    base = np.zeros((n_chains, n_draws, n, n))
    for c in range(n_chains):
        for d in range(n_draws):
            a = rng.standard_normal((n, n))
            base[c, d] = np.linalg.cholesky(a @ a.T + n * np.eye(n))

    def draws(low, high):
        return xr.DataArray(rng.uniform(low, high, (n_chains, n_draws)), dims=["chain", "draw"])

    return xr.Dataset({
        "L": xr.DataArray(base, dims=["chain", "draw", "var1", "var2"]),
        "s_march": draws(1.0, 6.0),
        "s_april": draws(1.0, 6.0),
        "s_may": draws(1.0, 6.0),
        "rho": draws(0.1, 0.95),
    })


def _oracle_scales(theta, start, indices):
    """Independent `s_t` reference at `indices`; scalars or `(chains, draws)` arrays."""
    s_march, s_april, s_may, rho = theta["s_march"], theta["s_april"], theta["s_may"], theta["rho"]
    batch = np.shape(s_may)
    out = np.empty((*batch, len(indices)))
    for m, t in enumerate(indices):
        j = t - start
        if j == 0:
            out[..., m] = s_march
        elif j == 1:
            out[..., m] = s_april
        elif j == 2:
            out[..., m] = s_may
        elif j >= 3:
            out[..., m] = 1.0 + (s_may - 1.0) * rho ** (j - 2)
        else:
            out[..., m] = 1.0
    return out


def _posterior_theta(posterior):
    return {k: posterior[k].values for k in ("s_march", "s_april", "s_may", "rho")}


# --- Gate 1: log_scales / s_t path ---------------------------------------------------


def test_log_scales_shape_and_pre_break(pandemic):
    T = 12
    log_s = pandemic.log_scales(THETA, T)
    assert log_s.shape == (T,)
    # s_t == 1 for t < t*  ->  log s_t == 0.
    assert np.all(log_s[:START] == 0.0)


def test_log_scales_outbreak_and_decay(pandemic):
    T = 12
    scales = np.exp(pandemic.log_scales(THETA, T))
    np.testing.assert_allclose(scales, _oracle_scales(THETA, START, np.arange(T)))
    # The three outbreak scales appear verbatim at t*, t*+1, t*+2.
    np.testing.assert_allclose(scales[START], THETA["s_march"])
    np.testing.assert_allclose(scales[START + 1], THETA["s_april"])
    np.testing.assert_allclose(scales[START + 2], THETA["s_may"])
    # Geometric decay 1 + (s_may - 1) * rho**(j - 2) for t >= t*+3.
    for j in range(3, T - START):
        expected = 1.0 + (THETA["s_may"] - 1.0) * THETA["rho"] ** (j - 2)
        np.testing.assert_allclose(scales[START + j], expected)
    # Decay is continuous with the outbreak: rho**0 at j == 2 reproduces s_may.
    np.testing.assert_allclose(scales[START + 3], 1.0 + (THETA["s_may"] - 1.0) * THETA["rho"])


# --- Gate 2: query methods ------------------------------------------------------------


def _assert_lower_tri_finite(factors):
    assert np.all(np.isfinite(factors))
    np.testing.assert_allclose(factors, np.tril(factors))


def test_cholesky_path(pandemic, posterior):
    base = posterior["L"].values
    n_chains, n_draws, n, _ = base.shape
    T = 10

    path = pandemic.cholesky_path(posterior, T)
    assert path.shape == (n_chains, n_draws, T, n, n)
    _assert_lower_tri_finite(path)

    scale = _oracle_scales(_posterior_theta(posterior), START, np.arange(T))
    np.testing.assert_allclose(path, scale[:, :, :, None, None] * base[:, :, None, :, :])


def test_cholesky_at(pandemic, posterior):
    base = posterior["L"].values
    n_chains, n_draws, n, _ = base.shape
    t = START + 1  # April outbreak scale

    factor = pandemic.cholesky_at(posterior, t)
    assert factor.shape == (n_chains, n_draws, n, n)
    _assert_lower_tri_finite(factor)

    scale = _oracle_scales(_posterior_theta(posterior), START, [t])[..., 0]
    np.testing.assert_allclose(scale, posterior["s_april"].values)
    np.testing.assert_allclose(factor, scale[:, :, None, None] * base)

    # t=None -> baseline factor (s_t = 1).
    np.testing.assert_allclose(pandemic.cholesky_at(posterior, None), base)


def test_forecast_cholesky_path(pandemic, posterior):
    base = posterior["L"].values
    n_chains, n_draws, n, _ = base.shape
    steps = 6

    path = pandemic.forecast_cholesky_path(posterior, steps, np.random.default_rng(1))
    assert path.shape == (n_chains, n_draws, steps, n, n)
    _assert_lower_tri_finite(path)

    # Forecast continues the decay: step k lives at absolute index t*+3+k.
    scale = _oracle_scales(_posterior_theta(posterior), START, START + 3 + np.arange(steps))
    np.testing.assert_allclose(path, scale[:, :, :, None, None] * base[:, :, None, :, :])
    # Step 0 (June 2020) uses 1 + (s_may - 1) * rho.
    expected0 = 1.0 + (posterior["s_may"].values - 1.0) * posterior["rho"].values
    np.testing.assert_allclose(scale[..., 0], expected0)


# --- Gate 3: hyperparameter priors ----------------------------------------------------


def test_hyperparameter_priors_scales_are_pareto(pandemic):
    priors = pandemic.hyperparameter_priors()
    assert set(priors) == {"s_march", "s_april", "s_may", "rho"}

    reference = stats.pareto(b=1.0, scale=1.0)
    for key in ("s_march", "s_april", "s_may"):
        prior = priors[key]
        assert isinstance(prior, Prior1D)
        assert prior.support == (1.0, np.inf)
        for x in (1.0, 2.0, 5.5):
            assert prior.logpdf(x) == pytest.approx(reference.logpdf(x))
        assert np.isneginf(prior.logpdf(0.5))  # below support


def test_hyperparameter_priors_decay_is_beta(pandemic):
    prior = pandemic.hyperparameter_priors()["rho"]
    assert isinstance(prior, Prior1D)
    assert prior.support == (0.0, 1.0)

    a, b = 3.03568545, 1.50892136
    reference = stats.beta(a=a, b=b)
    for x in (0.2, 0.5, 0.8):
        assert prior.logpdf(x) == pytest.approx(reference.logpdf(x))
    # The frozen (a, b) encode mode 0.8, sd 0.2.
    assert (a - 1.0) / (a + b - 2.0) == pytest.approx(0.8, abs=1e-6)
    assert reference.std() == pytest.approx(0.2, abs=1e-6)


# --- Gate 4: adapter metadata ---------------------------------------------------------


def test_is_time_varying(pandemic):
    assert pandemic.is_time_varying is True


def test_satisfies_volatility_process(pandemic):
    assert isinstance(pandemic, VolatilityProcess)
    assert pandemic.name == "pandemic_break"
