"""Tests for sampler specifications."""

import pytest
from pydantic import ValidationError

from impulso.protocols import Sampler
from impulso.samplers import NUTSSampler


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
