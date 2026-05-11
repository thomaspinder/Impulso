"""Regression test: standalone StochasticVolatility.fit(SVData) is
byte-for-byte unchanged after the P3 multivariate-adapter work."""

import numpy as np
import pandas as pd
import pytest

from impulso.samplers import NUTSSampler
from impulso.sv.data import SVData
from impulso.sv.spec import StochasticVolatility


@pytest.mark.slow
def test_standalone_sv_fit_matches_reference():
    rng = np.random.default_rng(42)
    T = 50
    y = 0.5 * rng.standard_normal(T)
    data = SVData(y=y, name="ref", index=pd.date_range("2000-01-01", periods=T, freq="MS"))
    sampler = NUTSSampler(cores=1, chains=2, draws=100, tune=100, random_seed=42, nuts_sampler="pymc")
    fitted = StochasticVolatility(dynamics="ar1").fit(data, sampler=sampler)

    ref = np.load("tests/data/sv_standalone_reference.npz")
    np.testing.assert_array_equal(fitted.idata.posterior["h"].values, ref["h"])
    np.testing.assert_array_equal(fitted.idata.posterior["mu"].values, ref["mu"])
    np.testing.assert_array_equal(fitted.idata.posterior["sigma_eta"].values, ref["sigma_eta"])
