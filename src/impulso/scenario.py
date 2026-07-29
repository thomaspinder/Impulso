"""Condition vocabulary for scenario analysis.

Typed, frozen spec objects expressing scenario content (ADR-0005):
`ShockPath` sets a structural shock's path, `VariablePath` pins a future
endogenous path, and the *targets* — `ProbabilityTarget`, `MomentTarget` —
state distributional facts a forecast should satisfy after entropic
tilting (ADR-0009). Each scenario method accepts only the condition types
that are legal for it, so illegal combinations are unrepresentable rather
than validated away.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import field_validator, model_validator

from impulso._base import ImpulsoBaseModel, ImpulsoModel


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
    forecast-side prescriptions (consumed by `structural_scenario`, arriving
    with that method) are positional from step 1 and must not carry
    `start`/`end`.

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
    free, and `NaN` entries mark unconstrained steps. Designed for the
    forecast-side conditioning methods (`conditional_forecast`,
    `structural_scenario`), which arrive with the next layers of the
    scenario stack.

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


def _check_finite(value: float, field: str) -> float:
    """Reject NaN/inf on a target field."""
    if not np.isfinite(value):
        raise ValueError(f"{field} must be finite, got {value}")
    return float(value)


def _check_horizon(value: int) -> int:
    """Reject non-positive horizons (steps are 1-based, as on `VariablePath`)."""
    if value < 1:
        raise ValueError(f"horizon is 1-based (step 1 is the first forecast step) and must be >= 1, got {value}")
    return value


class ProbabilityTarget(ImpulsoModel):
    """Require an event to carry a given probability after tilting.

    The event is `{y[horizon] < threshold}` (`direction="below"`, the
    default) or `{y[horizon] > threshold}` — strict inequalities, so a
    draw sitting exactly on the threshold is outside the event. Tilting
    reweights the existing forecast draws to hit `probability` while
    staying as close as possible (in relative entropy) to the untilted
    forecast; it never moves a draw, so nothing the draws already satisfy
    can be broken. `probability=1.0` is pure conditioning: draws outside
    the event get weight zero.

    Attributes:
        variable: Name of the endogenous variable the event refers to.
        horizon: Forecast step the event refers to, 1-based (step 1 is
            the first forecast step, matching `VariablePath`).
        threshold: The event's threshold, in the variable's own units.
        probability: Requested probability of the event, `0 < p <= 1`.
        direction: `"below"` for `y < threshold` (default) or `"above"`
            for `y > threshold`.
    """

    variable: str
    horizon: int
    threshold: float
    probability: float
    direction: Literal["below", "above"] = "below"

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, value: int) -> int:
        return _check_horizon(value)

    @field_validator("threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        return _check_finite(value, "threshold")

    @field_validator("probability")
    @classmethod
    def _validate_probability(cls, value: float) -> float:
        value = _check_finite(value, "probability")
        if not 0.0 < value <= 1.0:
            raise ValueError(f"probability must satisfy 0 < p <= 1, got {value}")
        return value


class MomentTarget(ImpulsoModel):
    """Require a variable's tilted forecast mean at one horizon.

    The moment condition is `E_w[y[horizon]] = mean`, imposed on the
    existing draws by reweighting. The requested mean must lie strictly
    inside the range the draws already span — tilting cannot move mass
    where there is none.

    Attributes:
        variable: Name of the endogenous variable.
        horizon: Forecast step, 1-based (step 1 is the first forecast
            step, matching `VariablePath`).
        mean: Requested tilted mean, in the variable's own units.
    """

    variable: str
    horizon: int
    mean: float

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, value: int) -> int:
        return _check_horizon(value)

    @field_validator("mean")
    @classmethod
    def _validate_mean(cls, value: float) -> float:
        return _check_finite(value, "mean")
