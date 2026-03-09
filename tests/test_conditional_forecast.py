"""Tests for conditional forecasting."""

import numpy as np
import pytest
from pydantic import ValidationError

from impulso.conditions import ForecastCondition


class TestForecastCondition:
    def test_basic_construction(self):
        fc = ForecastCondition(variable="y1", periods=[0, 1, 2], values=[1.0, 1.0, 1.0])
        assert fc.variable == "y1"
        assert fc.periods == [0, 1, 2]
        assert fc.constraint_type == "hard"

    def test_frozen(self):
        fc = ForecastCondition(variable="y1", periods=[0], values=[1.0])
        with pytest.raises(ValidationError):
            fc.variable = "y2"

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValidationError, match="periods length"):
            ForecastCondition(variable="y1", periods=[0, 1], values=[1.0])

    def test_rejects_empty_periods(self):
        with pytest.raises(ValidationError, match="non-empty"):
            ForecastCondition(variable="y1", periods=[], values=[])

    def test_rejects_negative_periods(self):
        with pytest.raises(ValidationError, match="non-negative"):
            ForecastCondition(variable="y1", periods=[-1], values=[1.0])


class TestConditionalForecastOnFittedVAR:
    def test_returns_forecast_result(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        conditions = [
            ForecastCondition(variable="y1", periods=[0, 1, 2, 3], values=[0.5, 0.5, 0.5, 0.5]),
        ]
        result = fitted.conditional_forecast(steps=8, conditions=conditions)
        assert result.median().shape == (8, 2)

    def test_constrained_periods_match_target(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        target = 0.5
        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        conditions = [
            ForecastCondition(variable="y1", periods=[0, 1], values=[target, target]),
        ]
        result = fitted.conditional_forecast(steps=4, conditions=conditions)
        median = result.median()
        # Constrained periods should be close to target (exact for hard constraints)
        np.testing.assert_allclose(median.iloc[0]["y1"], target, atol=1e-6)
        np.testing.assert_allclose(median.iloc[1]["y1"], target, atol=1e-6)

    def test_unconstrained_variable_differs_from_unconditional(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        conditions = [
            ForecastCondition(variable="y1", periods=[0, 1, 2, 3], values=[5.0, 5.0, 5.0, 5.0]),
        ]
        unconditional = fitted.forecast(steps=4).median()
        conditional = fitted.conditional_forecast(steps=4, conditions=conditions).median()
        # y2 should differ because y1 is forced far from its unconditional path
        assert not np.allclose(unconditional["y2"].values, conditional["y2"].values)

    def test_rejects_unknown_variable(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        conditions = [
            ForecastCondition(variable="unknown", periods=[0], values=[1.0]),
        ]
        with pytest.raises(ValueError, match="unknown"):
            fitted.conditional_forecast(steps=4, conditions=conditions)

    def test_rejects_period_out_of_range(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        conditions = [
            ForecastCondition(variable="y1", periods=[10], values=[1.0]),
        ]
        with pytest.raises(ValueError, match="out of range"):
            fitted.conditional_forecast(steps=4, conditions=conditions)


class TestConditionalForecastOnIdentifiedVAR:
    def test_returns_forecast_result(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR
        from impulso.identification import Cholesky

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        conditions = [
            ForecastCondition(variable="y1", periods=[0, 1], values=[0.5, 0.5]),
        ]
        result = identified.conditional_forecast(steps=4, conditions=conditions)
        assert result.median().shape == (4, 2)

    def test_with_shock_conditions(self, var_data_2v):
        from impulso.conjugate import ConjugateVAR
        from impulso.identification import Cholesky

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(var_data_2v)
        identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        conditions = [
            ForecastCondition(variable="y1", periods=[0, 1], values=[0.5, 0.5]),
        ]
        shock_conditions = [
            ForecastCondition(variable="y1", periods=[0, 1], values=[0.0, 0.0]),
        ]
        result = identified.conditional_forecast(steps=4, conditions=conditions, shock_conditions=shock_conditions)
        assert result.median().shape == (4, 2)
