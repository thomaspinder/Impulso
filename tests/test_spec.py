"""Tests for VAR specification."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from impulso.priors import MinnesotaPrior
from impulso.samplers import NUTSSampler
from impulso.spec import VAR


class TestVARSpec:
    def test_fixed_lags(self):
        spec = VAR(lags=4, prior="minnesota")
        assert spec.lags == 4
        assert spec.max_lags is None

    @pytest.mark.parametrize("criterion", ["aic", "bic", "hq"])
    def test_string_lags(self, criterion):
        spec = VAR(lags=criterion, prior="minnesota")
        assert spec.lags == criterion

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

    def test_default_sampler_returns_nuts_with_safe_defaults(self):
        """_default_sampler() must return NUTSSampler(cores=1, chains=4)."""
        sampler = VAR._default_sampler()
        assert isinstance(sampler, NUTSSampler)
        assert sampler.cores == 1
        assert sampler.chains == 4

    def test_fit_with_no_sampler_uses_default(self, var_data_3v):
        """fit(sampler=None) should call _default_sampler() and use its result."""
        spec = VAR(lags=1, prior="minnesota")
        default = VAR._default_sampler()

        with patch.object(VAR, "_default_sampler", return_value=default):
            with patch.object(NUTSSampler, "sample") as mock_sample:
                mock_idata = MagicMock()
                mock_sample.return_value = mock_idata
                spec.fit(var_data_3v, sampler=None)

                # _default_sampler must have been called
                VAR._default_sampler.assert_called_once()
                # sample must have been called on the default sampler
                mock_sample.assert_called_once()

    def test_default_sampler_is_callable_without_instance(self):
        """_default_sampler() works as a static method (callable on class)."""
        sampler = VAR._default_sampler()
        assert hasattr(sampler, "sample")
        assert callable(sampler.sample)

    def test_default_sampler_returns_fresh_instance_each_call(self):
        """Each call to _default_sampler() returns a new instance."""
        s1 = VAR._default_sampler()
        s2 = VAR._default_sampler()
        assert s1 is not s2
        assert s1.cores == s2.cores == 1
        assert s1.chains == s2.chains == 4
