"""Tests for VAR specification."""

import pytest
from pydantic import ValidationError

from litterman.priors import MinnesotaPrior
from litterman.spec import VAR


class TestVARSpec:
    def test_fixed_lags(self):
        spec = VAR(lags=4, prior="minnesota")
        assert spec.lags == 4
        assert spec.max_lags is None

    def test_string_lags(self):
        spec = VAR(lags="bic", prior="minnesota")
        assert spec.lags == "bic"

    def test_string_lags_with_max(self):
        spec = VAR(lags="aic", max_lags=12, prior="minnesota")
        assert spec.max_lags == 12

    def test_rejects_max_lags_with_fixed(self):
        with pytest.raises(ValueError, match="max_lags is only valid"):
            VAR(lags=4, max_lags=12, prior="minnesota")

    def test_prior_string_resolves(self):
        spec = VAR(lags=2, prior="minnesota")
        assert isinstance(spec.resolved_prior, MinnesotaPrior)

    def test_prior_object(self):
        prior = MinnesotaPrior(tightness=0.2)
        spec = VAR(lags=2, prior=prior)
        assert spec.resolved_prior.tightness == 0.2

    def test_frozen(self):
        spec = VAR(lags=2, prior="minnesota")
        with pytest.raises(ValidationError):
            spec.lags = 3

    def test_rejects_zero_lags(self):
        with pytest.raises(ValidationError):
            VAR(lags=0, prior="minnesota")
