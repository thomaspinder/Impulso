"""Condition vocabulary for scenario analysis.

Typed, frozen spec objects expressing scenario content (ADR-0005):
`ShockPath` sets a structural shock's path, `VariablePath` pins a future
endogenous path. Each scenario method accepts only the condition types
that are legal for it, so illegal combinations are unrepresentable rather
than validated away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import field_validator, model_validator

from impulso._base import ImpulsoBaseModel


def _coerce_values(value: float | np.ndarray) -> float | np.ndarray:
    """Normalise a values field: scalars to float, arrays to read-only float64.

    Scalars broadcast at application time (to the full resolved window
    in-sample, to all steps on the forecast axis), so they stay scalar
    here. Arrays are coerced to 1-D float64 and made read-only so the
    frozen-model contract covers contents, not just attribute rebinding.
    """
    if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
        return float(value)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"values must be a scalar or 1-D array, got {arr.ndim}-D")
    if arr.size == 0:
        raise ValueError("values array must not be empty")
    arr = arr.copy()
    arr.setflags(write=False)
    return arr


class ShockPath(ImpulsoBaseModel):
    """Set a structural shock's path over a window.

    Values are in one-standard-deviation shock units — `0.0` switches the
    shock off. A scalar broadcasts to the full window; an explicit array
    must match the resolved window length. `start`/`end` are in-sample
    timestamps resolved against the lag-trimmed index (the
    `historical_decomposition` convention) and default to the full sample;
    forecast-side prescriptions (`structural_scenario`) are positional from
    step 1 and must not carry `start`/`end`.

    Attributes:
        shock: Name of the structural shock to set (a shock coordinate of
            the identification scheme; `unidentified_*` columns are
            rejected at application time).
        values: Scalar (broadcast) or 1-D array of replacement values in
            one-standard-deviation units.
        start: Optional window start (in-sample edits only).
        end: Optional window end, inclusive (in-sample edits only).
    """

    shock: str
    values: float | np.ndarray
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

    @field_validator("values", mode="before")
    @classmethod
    def _validate_values(cls, value: float | np.ndarray) -> float | np.ndarray:
        return _coerce_values(value)

    @field_validator("start", "end", mode="before")
    @classmethod
    def _coerce_timestamps(cls, value: object) -> object:
        if isinstance(value, str):
            return pd.Timestamp(value)
        return value

    @model_validator(mode="after")
    def _validate_window(self) -> ShockPath:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"ShockPath window is empty: start={self.start} is after end={self.end}")
        return self


class VariablePath(ImpulsoBaseModel):
    """Pin a future endogenous variable's path (hard condition, forecast axis).

    Values run from forecast step 1; a scalar broadcasts to all steps, an
    array of length `L < steps` pins steps `1..L` and leaves the rest
    free, and `NaN` entries mark unconstrained steps. Consumed by
    `conditional_forecast` and `structural_scenario`.

    Attributes:
        variable: Name of the endogenous variable to pin.
        values: Scalar (broadcast) or 1-D array of pinned values with
            `NaN` marking free steps.
    """

    variable: str
    values: float | np.ndarray

    @field_validator("values", mode="before")
    @classmethod
    def _validate_values(cls, value: float | np.ndarray) -> float | np.ndarray:
        return _coerce_values(value)
