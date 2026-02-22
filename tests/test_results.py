"""Tests for result objects."""

import pandas as pd
import pytest
from pydantic import ValidationError

from litterman.results import HDIResult, LagOrderResult


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
