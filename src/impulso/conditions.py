"""Forecast condition definitions for conditional forecasting."""

from typing import Literal, Self

from pydantic import model_validator

from impulso._base import ImpulsoModel


class ForecastCondition(ImpulsoModel):
    """A constraint on a variable's future path for conditional forecasting.

    Attributes:
        variable: Name of the variable to constrain.
        periods: Forecast steps to constrain (0-indexed).
        values: Target values at those periods.
        constraint_type: Type of constraint. Only 'hard' is currently supported.
    """

    variable: str
    periods: list[int]
    values: list[float]
    constraint_type: Literal["hard"] = "hard"

    @model_validator(mode="after")
    def _validate_periods_values_match(self) -> Self:
        if len(self.periods) != len(self.values):
            raise ValueError(f"periods length ({len(self.periods)}) must equal values length ({len(self.values)})")
        if len(self.periods) == 0:
            raise ValueError("periods must be non-empty")
        if any(p < 0 for p in self.periods):
            raise ValueError("All periods must be non-negative")
        return self
