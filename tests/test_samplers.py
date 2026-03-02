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
