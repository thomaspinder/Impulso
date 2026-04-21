"""Tests for SV result types."""

import pandas as pd

from impulso.results import VolatilityResult


def test_volatility_result_median_shape(synthetic_sv_idata):
    result = VolatilityResult(
        idata=synthetic_sv_idata,
        series_name="sim",
        index=pd.date_range("2000-01-01", periods=100, freq="MS"),
    )
    med = result.median()
    assert isinstance(med, pd.DataFrame)
    assert med.shape == (100, 1)
    assert "sim" in med.columns


def test_volatility_result_median_positive(synthetic_sv_idata):
    """exp(h/2) must be strictly positive."""
    result = VolatilityResult(
        idata=synthetic_sv_idata,
        series_name="sim",
        index=pd.date_range("2000-01-01", periods=100, freq="MS"),
    )
    assert (result.median().values > 0).all()


def test_volatility_result_hdi_bounds_ordered(synthetic_sv_idata):
    result = VolatilityResult(
        idata=synthetic_sv_idata,
        series_name="sim",
        index=pd.date_range("2000-01-01", periods=100, freq="MS"),
    )
    hdi_out = result.hdi(prob=0.89)
    assert hdi_out.prob == 0.89
    assert (hdi_out.lower.values <= hdi_out.upper.values).all()


def test_volatility_result_to_dataframe_has_index(synthetic_sv_idata):
    idx = pd.date_range("2000-01-01", periods=100, freq="MS")
    result = VolatilityResult(idata=synthetic_sv_idata, series_name="sim", index=idx)
    df = result.to_dataframe()
    assert list(df.index) == list(idx)
