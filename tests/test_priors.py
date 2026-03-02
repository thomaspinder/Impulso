"""Tests for prior specifications."""

import pytest
from pydantic import ValidationError

from impulso.priors import MinnesotaPrior
from impulso.protocols import Prior


class TestMinnesotaPrior:
    def test_default_construction(self):
        prior = MinnesotaPrior()
        assert prior.tightness == 0.1
        assert prior.decay == "harmonic"
        assert prior.cross_shrinkage == 0.5

    def test_custom_construction(self):
        prior = MinnesotaPrior(tightness=0.2, decay="geometric", cross_shrinkage=0.8)
        assert prior.tightness == 0.2
        assert prior.decay == "geometric"

    def test_frozen(self):
        prior = MinnesotaPrior()
        with pytest.raises(ValidationError):
            prior.tightness = 0.5

    def test_satisfies_prior_protocol(self):
        prior = MinnesotaPrior()
        assert isinstance(prior, Prior)

    @pytest.mark.parametrize("bad_tightness", [0.0, -0.1, -1.0])
    def test_rejects_invalid_tightness(self, bad_tightness):
        with pytest.raises(ValidationError):
            MinnesotaPrior(tightness=bad_tightness)

    def test_build_priors_returns_dict(self):
        prior = MinnesotaPrior()
        result = prior.build_priors(n_vars=3, n_lags=2)
        assert isinstance(result, dict)
        assert "B_mu" in result
        assert "B_sigma" in result
