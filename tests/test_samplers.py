"""Tests for sampler specifications."""

import pytest
from impulso.protocols import Sampler
from impulso.samplers import NUTSSampler
from pydantic import ValidationError


class TestNUTSSampler:
    def test_default_construction(self):
        sampler = NUTSSampler()
        assert sampler.draws == 1000
        assert sampler.tune == 1000
        assert sampler.chains == 4
        assert sampler.cores is None
        assert sampler.target_accept == 0.8
        assert sampler.random_seed is None

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

    def test_rejects_zero_draws(self):
        with pytest.raises(ValidationError):
            NUTSSampler(draws=0)
