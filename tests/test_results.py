"""Tests for result objects."""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from impulso.results import (
    FEVDResult,
    ForecastResult,
    HDIResult,
    HistoricalDecompositionResult,
    IRFResult,
    LagOrderResult,
    VARResultBase,
)


class TestHDIResult:
    def test_construction(self):
        lower = pd.DataFrame({"a": [1.0], "b": [2.0]})
        upper = pd.DataFrame({"a": [3.0], "b": [4.0]})
        hdi = HDIResult(lower=lower, upper=upper, prob=0.89)
        assert hdi.prob == 0.89

    def test_frozen(self):
        lower = pd.DataFrame({"a": [1.0]})
        upper = pd.DataFrame({"a": [2.0]})
        hdi = HDIResult(lower=lower, upper=upper, prob=0.89)
        with pytest.raises(ValidationError):
            hdi.prob = 0.95


class TestLagOrderResult:
    def test_construction(self):
        table = pd.DataFrame({"aic": [100, 95], "bic": [110, 105], "hq": [105, 100]}, index=[1, 2])
        result = LagOrderResult(aic=2, bic=2, hq=2, criteria_table=table)
        assert result.aic == 2
        assert result.bic == 2

    def test_summary_returns_table(self):
        table = pd.DataFrame({"aic": [100, 95], "bic": [110, 105], "hq": [105, 100]}, index=[1, 2])
        result = LagOrderResult(aic=2, bic=2, hq=2, criteria_table=table)
        summary = result.summary()
        assert isinstance(summary, pd.DataFrame)

    def test_criteria_table_excluded_from_repr(self):
        table = pd.DataFrame({"aic": [100]}, index=[1])
        result = LagOrderResult(aic=1, bic=1, hq=1, criteria_table=table)
        assert "criteria_table" not in repr(result)


class TestForecastResult:
    def test_is_subclass_of_base(self):
        assert issubclass(ForecastResult, VARResultBase)


class TestStructuralResults:
    def test_irf_result_is_subclass(self):
        assert issubclass(IRFResult, VARResultBase)

    def test_fevd_result_is_subclass(self):
        assert issubclass(FEVDResult, VARResultBase)

    def test_hd_result_is_subclass(self):
        assert issubclass(HistoricalDecompositionResult, VARResultBase)


class TestForecastResultMethods:
    def _make_result(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, 50, 4, 2))
        da = xr.DataArray(
            data,
            dims=["chain", "draw", "step", "variable"],
            coords={"variable": ["y1", "y2"]},
            name="forecast",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"forecast": da}))
        return ForecastResult.model_construct(idata=idata, steps=4, var_names=["y1", "y2"])

    def test_median_shape(self):
        result = self._make_result()
        med = result.median()
        assert med.shape == (4, 2)
        assert list(med.columns) == ["y1", "y2"]

    def test_hdi_bounds_ordered(self):
        result = self._make_result()
        hdi = result.hdi(prob=0.89)
        assert (hdi.upper.values >= hdi.lower.values).all()

    def test_to_dataframe_shape(self):
        result = self._make_result()
        df = result.to_dataframe()
        assert df.shape == (4, 2)


class TestIRFResultMethods:
    def test_median_shape(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, 50, 11, 2, 2))
        da = xr.DataArray(
            data,
            dims=["chain", "draw", "horizon", "response", "shock"],
            coords={"response": ["y1", "y2"], "shock": ["y1", "y2"], "horizon": np.arange(11)},
            name="irf",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"irf": da}))
        result = IRFResult.model_construct(idata=idata, horizon=10, var_names=["y1", "y2"])
        med = result.median()
        assert med.shape == (11, 4)  # (horizon+1) x (n_vars * n_vars)
