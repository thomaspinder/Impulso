"""Tests for result objects."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from impulso._arviz_compat import make_idata
from impulso.results import (
    ConditionalForecastResult,
    FEVDResult,
    ForecastResult,
    HDIResult,
    HistoricalDecompositionResult,
    IRFResult,
    LagOrderResult,
    VARResultBase,
)
from impulso.scenario import VariablePath


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
        idata = make_idata(posterior_predictive=xr.Dataset({"forecast": da}))
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


def _make_irf_result(responses=("gdp", "inf"), shocks=("e_gdp", "e_inf"), horizon=10):
    rng = np.random.default_rng(42)
    n_resp = len(responses)
    n_shock = len(shocks)
    data = rng.standard_normal((2, 50, horizon + 1, n_resp, n_shock))
    da = xr.DataArray(
        data,
        dims=["chain", "draw", "horizon", "response", "shock"],
        coords={
            "response": list(responses),
            "shock": list(shocks),
            "horizon": np.arange(horizon + 1),
        },
        name="irf",
    )
    idata = make_idata(posterior_predictive=xr.Dataset({"irf": da}))
    return IRFResult.model_construct(idata=idata, horizon=horizon, var_names=list(responses))


def _make_fevd_result(responses=("gdp", "inf"), shocks=("e_gdp", "e_inf"), horizon=10):
    rng = np.random.default_rng(43)
    n_resp = len(responses)
    n_shock = len(shocks)
    data = rng.uniform(size=(2, 50, horizon + 1, n_resp, n_shock))
    data = data / data.sum(axis=-1, keepdims=True)  # rows sum to 1, like an FEVD
    da = xr.DataArray(
        data,
        dims=["chain", "draw", "horizon", "response", "shock"],
        coords={
            "response": list(responses),
            "shock": list(shocks),
            "horizon": np.arange(horizon + 1),
        },
        name="fevd",
    )
    idata = make_idata(posterior_predictive=xr.Dataset({"fevd": da}))
    return FEVDResult.model_construct(idata=idata, horizon=horizon, var_names=list(responses))


def _make_hd_result(responses=("gdp", "inf"), shocks=("e_gdp", "e_inf"), n_periods=24):
    rng = np.random.default_rng(44)
    n_resp = len(responses)
    n_shock = len(shocks)
    data = rng.standard_normal((2, 50, n_periods, n_resp, n_shock))
    time_index = pd.date_range("2000-01-01", periods=n_periods, freq="QS")
    da = xr.DataArray(
        data,
        dims=["chain", "draw", "time", "response", "shock"],
        coords={
            "response": list(responses),
            "shock": list(shocks),
            "time": ("time", time_index),
        },
        name="hd",
    )
    idata = make_idata(posterior_predictive=xr.Dataset({"hd": da}))
    return HistoricalDecompositionResult.model_construct(idata=idata, var_names=list(responses))


class TestIRFResultMethods:
    def test_median_shape_and_labels(self):
        """IRF.median() returns wide DataFrame indexed by horizon with
        MultiIndex(['response','shock']) on columns.
        """
        result = _make_irf_result(horizon=10)
        med = result.median()
        assert isinstance(med, pd.DataFrame)
        assert med.shape == (11, 4)
        assert isinstance(med.columns, pd.MultiIndex)
        assert med.columns.names == ["response", "shock"]
        assert set(med.columns.tolist()) == {
            ("gdp", "e_gdp"),
            ("gdp", "e_inf"),
            ("inf", "e_gdp"),
            ("inf", "e_inf"),
        }
        # Row index is the integer horizon 0..H
        assert list(med.index) == list(range(11))

    def test_median_values_round_trip_labels(self):
        """Values selected by label must match values pulled from the raw
        DataArray, proving the column labelling isn't swapped or permuted.
        """
        result = _make_irf_result(horizon=10)
        med = result.median()
        raw = result.idata.posterior_predictive["irf"].median(dim=("chain", "draw"))
        for resp in ("gdp", "inf"):
            for shock in ("e_gdp", "e_inf"):
                np.testing.assert_allclose(
                    med[(resp, shock)].values,
                    raw.sel(response=resp, shock=shock).values,
                )

    def test_hdi_shape_and_labels(self):
        """HDIResult.lower / .upper must mirror median()'s shape and labels."""
        result = _make_irf_result(horizon=10)
        hdi = result.hdi(prob=0.89)
        for frame in (hdi.lower, hdi.upper):
            assert isinstance(frame, pd.DataFrame)
            assert frame.shape == (11, 4)
            assert isinstance(frame.columns, pd.MultiIndex)
            assert frame.columns.names == ["response", "shock"]
            assert set(frame.columns.tolist()) == {
                ("gdp", "e_gdp"),
                ("gdp", "e_inf"),
                ("inf", "e_gdp"),
                ("inf", "e_inf"),
            }
            assert list(frame.index) == list(range(11))
        assert (hdi.upper.values >= hdi.lower.values).all()

    def test_shock_names_follow_shock_coord(self):
        """shock_names reads the `shock` coordinate, not var_names."""
        result = _make_irf_result(shocks=("supply", "demand"))
        assert result.shock_names == ["supply", "demand"]
        assert all(isinstance(s, str) for s in result.shock_names)


class TestFEVDResultMethods:
    def test_median_shape_and_labels(self):
        """FEVD.median() returns wide DataFrame indexed by horizon with
        MultiIndex(['response','shock']) on columns.
        """
        result = _make_fevd_result(horizon=10)
        med = result.median()
        assert isinstance(med, pd.DataFrame)
        assert med.shape == (11, 4)
        assert isinstance(med.columns, pd.MultiIndex)
        assert med.columns.names == ["response", "shock"]
        assert set(med.columns.tolist()) == {
            ("gdp", "e_gdp"),
            ("gdp", "e_inf"),
            ("inf", "e_gdp"),
            ("inf", "e_inf"),
        }
        assert list(med.index) == list(range(11))

    def test_median_values_round_trip_labels(self):
        result = _make_fevd_result(horizon=10)
        med = result.median()
        raw = result.idata.posterior_predictive["fevd"].median(dim=("chain", "draw"))
        for resp in ("gdp", "inf"):
            for shock in ("e_gdp", "e_inf"):
                np.testing.assert_allclose(
                    med[(resp, shock)].values,
                    raw.sel(response=resp, shock=shock).values,
                )

    def test_hdi_shape_and_labels(self):
        result = _make_fevd_result(horizon=10)
        hdi = result.hdi(prob=0.89)
        for frame in (hdi.lower, hdi.upper):
            assert isinstance(frame, pd.DataFrame)
            assert frame.shape == (11, 4)
            assert isinstance(frame.columns, pd.MultiIndex)
            assert frame.columns.names == ["response", "shock"]
            assert list(frame.index) == list(range(11))
        assert (hdi.upper.values >= hdi.lower.values).all()

    def test_shock_names_follow_shock_coord(self):
        """shock_names reads the `shock` coordinate, not var_names."""
        result = _make_fevd_result(shocks=("supply", "demand"))
        assert result.shock_names == ["supply", "demand"]
        assert all(isinstance(s, str) for s in result.shock_names)


class TestHistoricalDecompositionResultMethods:
    def test_median_indexed_by_time_with_multiindex_columns(self):
        """HD.median() returns wide DataFrame indexed by a DatetimeIndex
        (the per-t in-sample dates) with MultiIndex(['response','shock'])
        on columns. The shape is (T, n_resp * n_shock).
        """
        result = _make_hd_result(n_periods=24)
        med = result.median()
        assert isinstance(med, pd.DataFrame)
        assert med.shape == (24, 4)
        assert isinstance(med.index, pd.DatetimeIndex)
        expected_index = pd.date_range("2000-01-01", periods=24, freq="QS")
        assert med.index.name == "time"
        np.testing.assert_array_equal(med.index.values, expected_index.values)
        assert isinstance(med.columns, pd.MultiIndex)
        assert med.columns.names == ["response", "shock"]
        assert set(med.columns.tolist()) == {
            ("gdp", "e_gdp"),
            ("gdp", "e_inf"),
            ("inf", "e_gdp"),
            ("inf", "e_inf"),
        }

    def test_median_values_round_trip_labels(self):
        result = _make_hd_result(n_periods=24)
        med = result.median()
        raw = result.idata.posterior_predictive["hd"].median(dim=("chain", "draw"))
        for resp in ("gdp", "inf"):
            for shock in ("e_gdp", "e_inf"):
                np.testing.assert_allclose(
                    med[(resp, shock)].values,
                    raw.sel(response=resp, shock=shock).values,
                )

    def test_hdi_shape_and_labels(self):
        result = _make_hd_result(n_periods=24)
        hdi = result.hdi(prob=0.89)
        expected_index = pd.date_range("2000-01-01", periods=24, freq="QS")
        for frame in (hdi.lower, hdi.upper):
            assert isinstance(frame, pd.DataFrame)
            assert frame.shape == (24, 4)
            assert isinstance(frame.index, pd.DatetimeIndex)
            assert frame.index.name == "time"
            np.testing.assert_array_equal(frame.index.values, expected_index.values)
            assert isinstance(frame.columns, pd.MultiIndex)
            assert frame.columns.names == ["response", "shock"]
        assert (hdi.upper.values >= hdi.lower.values).all()

    def test_shock_names_follow_shock_coord(self):
        """shock_names reads the `shock` coordinate, so a partially-identified
        decomposition surfaces its `unidentified_remainder` column.
        """
        result = _make_hd_result(shocks=("target", "unidentified_remainder"))
        assert result.shock_names == ["target", "unidentified_remainder"]

    def test_deviation_is_median_of_shock_sum(self):
        """deviation() is the median of the per-draw sum over shocks, with
        the index and columns of the per-variable frames.
        """
        result = _make_hd_result(n_periods=24)
        dev = result.deviation()
        raw = result.idata.posterior_predictive["hd"]
        expected = raw.sum("shock").median(dim=("chain", "draw")).transpose("time", "response")
        assert isinstance(dev.index, pd.DatetimeIndex)
        assert dev.index.name == "time"
        assert list(dev.columns) == ["gdp", "inf"]
        np.testing.assert_allclose(dev.values, expected.values)


def _make_conditional_forecast_result(conditions=(), attrs=None, steps=4, n_vars=2):
    rng = np.random.default_rng(45)
    names = [f"y{i + 1}" for i in range(n_vars)]
    data = rng.standard_normal((2, 50, steps, n_vars))
    da = xr.DataArray(
        data,
        dims=["chain", "draw", "step", "variable"],
        coords={"variable": names},
        name="forecast",
    )
    ds = xr.Dataset({"forecast": da}, attrs=dict(attrs or {}))
    idata = make_idata(posterior_predictive=ds)
    return ConditionalForecastResult.model_construct(
        idata=idata,
        steps=steps,
        var_names=names,
        conditions=list(conditions),
    )


class TestConditionalForecastResultAccessors:
    def test_n_restrictions_reads_dataset_attrs(self):
        result = _make_conditional_forecast_result(attrs={"n_restrictions": 3})
        assert result.n_restrictions == 3

    def test_n_restrictions_defaults_to_zero_without_metadata(self):
        """A hand-built result without plausibility attrs reports zero."""
        result = _make_conditional_forecast_result()
        assert result.n_restrictions == 0

    def test_pinned_values_resolves_scalars_arrays_and_nans(self):
        """A scalar broadcasts to every step; an array pins a leading run
        with NaN entries left free; unconditioned variables map to [].
        """
        conditions = [
            VariablePath(variable="y1", values=np.array([0.4, np.nan, -0.2])),
            VariablePath(variable="y2", values=1.5),
        ]
        result = _make_conditional_forecast_result(conditions=conditions, steps=4)
        pins = result.pinned_values()
        assert pins["y1"] == [(1, 0.4), (3, -0.2)]
        assert pins["y2"] == [(1, 1.5), (2, 1.5), (3, 1.5), (4, 1.5)]

    def test_pinned_values_empty_without_conditions(self):
        result = _make_conditional_forecast_result()
        assert result.pinned_values() == {"y1": [], "y2": []}
