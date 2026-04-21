"""Tests for SVData."""

import numpy as np
import pandas as pd
import pytest

from impulso.sv.data import SVData


def test_svdata_construct_from_arrays():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(30)
    index = pd.date_range("2000-01-01", periods=30, freq="MS")
    data = SVData(y=y, name="inflation", index=index)
    assert data.y.shape == (30,)
    assert data.name == "inflation"
    assert len(data.index) == 30
    assert not data.y.flags.writeable  # readonly after construction


def test_svdata_rejects_2d_array():
    y = np.zeros((5, 2))
    index = pd.date_range("2000-01-01", periods=5, freq="MS")
    with pytest.raises(ValueError, match="y must be 1-D"):
        SVData(y=y, name="x", index=index)


def test_svdata_rejects_length_mismatch():
    y = np.zeros(5)
    index = pd.date_range("2000-01-01", periods=10, freq="MS")
    with pytest.raises(ValueError, match="index length"):
        SVData(y=y, name="x", index=index)


def test_svdata_rejects_nan():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(30)
    y[5] = np.nan
    index = pd.date_range("2000-01-01", periods=30, freq="MS")
    with pytest.raises(ValueError, match="NaN or Inf"):
        SVData(y=y, name="x", index=index)


def test_svdata_rejects_short_series():
    y = np.linspace(0.0, 1.0, 23)  # 23 < SVData._MIN_OBS (24)
    index = pd.date_range("2000-01-01", periods=23, freq="MS")
    with pytest.raises(ValueError, match="at least 24 observations"):
        SVData(y=y, name="x", index=index)


def test_svdata_rejects_constant_series():
    y = np.ones(50)
    index = pd.date_range("2000-01-01", periods=50, freq="MS")
    with pytest.raises(ValueError, match="y is constant"):
        SVData(y=y, name="x", index=index)


def test_svdata_from_series_uses_series_name():
    rng = np.random.default_rng(0)
    s = pd.Series(
        rng.standard_normal(30),
        index=pd.date_range("2000-01-01", periods=30, freq="MS"),
        name="gdp",
    )
    data = SVData.from_series(s)
    assert data.name == "gdp"
    assert data.y.shape == (30,)


def test_svdata_from_series_override_name():
    rng = np.random.default_rng(0)
    s = pd.Series(
        rng.standard_normal(30),
        index=pd.date_range("2000-01-01", periods=30, freq="MS"),
    )
    data = SVData.from_series(s, name="override")
    assert data.name == "override"


def test_svdata_from_series_rejects_non_datetime_index():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.standard_normal(30), name="x")  # default RangeIndex
    with pytest.raises(TypeError, match="DatetimeIndex"):
        SVData.from_series(s)


def test_svdata_from_series_requires_name_when_unnamed():
    rng = np.random.default_rng(0)
    s = pd.Series(
        rng.standard_normal(30),
        index=pd.date_range("2000-01-01", periods=30, freq="MS"),
    )
    with pytest.raises(ValueError, match="name is required"):
        SVData.from_series(s)
