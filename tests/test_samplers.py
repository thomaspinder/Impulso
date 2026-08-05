"""Tests for sampler specifications."""

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from impulso._arviz_compat import make_idata
from impulso.protocols import Sampler
from impulso.samplers import NUTSSampler


def _synthetic_idata(nan_pad_chain: int | None = None, pad_from: int = 30, scattered_nan: bool = False):
    """Posterior with 2 chains x 50 draws of one matrix and one scalar variable.

    ``nan_pad_chain`` reproduces nutpie's interrupted-chain signature: every
    value of every variable is NaN from ``pad_from`` onward in that chain,
    while ``diverging`` stays False (nutpie zero-fills non-float buffers).
    ``scattered_nan`` instead poisons a few isolated cells of a single
    variable, which a legitimate posterior may contain.
    """
    rng = np.random.default_rng(0)
    b = rng.normal(size=(2, 50, 3, 3))
    mu = rng.normal(size=(2, 50))
    if nan_pad_chain is not None:
        b[nan_pad_chain, pad_from:] = np.nan
        mu[nan_pad_chain, pad_from:] = np.nan
    if scattered_nan:
        b[0, 5, 1, 2] = np.nan
        b[1, 40, 0, 0] = np.nan
    posterior = xr.Dataset({
        "B": (("chain", "draw", "var", "coeff"), b),
        "mu": (("chain", "draw"), mu),
    })
    sample_stats = xr.Dataset({"diverging": (("chain", "draw"), np.zeros((2, 50), dtype=bool))})
    return make_idata(posterior=posterior, sample_stats=sample_stats)


class TestIncompleteDrawGuard:
    """`NUTSSampler.sample` must not hand back a truncated or NaN-padded posterior.

    An interrupted or dead chain does not raise: nutpie aborts and returns
    the partial trace, NaN-padding slower chains (zero-filling `diverging`)
    and truncating the draw dimension when every chain stopped early — the
    proxy-svar CI incident. The guard turns both shapes into a loud error
    at the sampling seam.
    """

    @staticmethod
    def _sample_with(monkeypatch, idata):
        import pymc as pm

        monkeypatch.setattr(pm, "sample", lambda *args, **kwargs: idata)
        with pm.Model():
            return NUTSSampler(draws=50, chains=2).sample(pm.Model())

    def test_nan_padded_chain_raises(self, monkeypatch):
        idata = _synthetic_idata(nan_pad_chain=1)
        with pytest.raises(RuntimeError, match=r"chain 1 delivered 30 of 50"):
            self._sample_with(monkeypatch, idata)

    def test_error_names_interrupt_and_remedy(self, monkeypatch):
        idata = _synthetic_idata(nan_pad_chain=0, pad_from=1)
        with pytest.raises(RuntimeError, match=r"(?i)interrupt"):
            self._sample_with(monkeypatch, idata)

    def test_truncated_posterior_raises(self, monkeypatch):
        """All chains stopped early: shorter draw dim, no NaN anywhere."""
        idata = _synthetic_idata()
        import pymc as pm

        monkeypatch.setattr(pm, "sample", lambda *args, **kwargs: idata)
        with pm.Model(), pytest.raises(RuntimeError, match=r"at most 50 of 200"):
            NUTSSampler(draws=200, chains=2).sample(pm.Model())

    def test_interrupt_during_tuning_zero_draws_raises(self, monkeypatch):
        """Interrupt during tuning: nutpie returns a zero-length draw dim."""
        import pymc as pm

        posterior = xr.Dataset(
            {"B": (("chain", "draw", "var"), np.empty((2, 0, 3)))},
        )
        idata = make_idata(posterior=posterior)
        monkeypatch.setattr(pm, "sample", lambda *args, **kwargs: idata)
        with pm.Model(), pytest.raises(RuntimeError, match=r"at most 0 of 50"):
            NUTSSampler(draws=50, chains=2).sample(pm.Model())

    def test_scattered_nan_values_pass_through(self, monkeypatch):
        """Isolated NaN in one variable is not the padding signature."""
        idata = _synthetic_idata(scattered_nan=True)
        assert self._sample_with(monkeypatch, idata) is idata

    def test_clean_posterior_passes_through(self, monkeypatch):
        idata = _synthetic_idata()
        assert self._sample_with(monkeypatch, idata) is idata


class TestNUTSSampler:
    def test_default_construction(self):
        sampler = NUTSSampler()
        assert sampler.draws == 1000
        assert sampler.tune == 1000
        assert sampler.chains == 4
        assert sampler.cores is None
        assert sampler.target_accept == 0.8
        assert sampler.random_seed is None
        assert sampler.nuts_sampler in ("pymc", "nutpie")

    def test_custom_construction(self):
        sampler = NUTSSampler(draws=2000, chains=2, random_seed=42)
        assert sampler.draws == 2000
        assert sampler.chains == 2
        assert sampler.random_seed == 42

    def test_frozen(self):
        sampler = NUTSSampler()
        with pytest.raises(ValidationError):
            sampler.draws = 500

    def test_satisfies_sampler_protocol(self):
        sampler = NUTSSampler()
        assert isinstance(sampler, Sampler)

    @pytest.mark.parametrize("bad_draws", [0, -1])
    def test_rejects_invalid_draws(self, bad_draws):
        with pytest.raises(ValidationError):
            NUTSSampler(draws=bad_draws)

    def test_default_nuts_sampler_is_nutpie_when_available(self):
        """nutpie is installed in dev, so default should be 'nutpie'."""
        sampler = NUTSSampler()
        assert sampler.nuts_sampler == "nutpie"

    def test_explicit_pymc_backend(self):
        sampler = NUTSSampler(nuts_sampler="pymc")
        assert sampler.nuts_sampler == "pymc"

    def test_rejects_invalid_nuts_sampler(self):
        with pytest.raises(ValidationError):
            NUTSSampler(nuts_sampler="invalid")
