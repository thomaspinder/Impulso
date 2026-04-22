"""SVData — validated, immutable data container for univariate SV models."""

from typing import Self

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from impulso._base import ImpulsoBaseModel


class SVData(ImpulsoBaseModel):
    """Immutable, validated container for univariate SV estimation data.

    Attributes:
        y: 1-D array of observed values, length T.
        name: Name of the series.
        index: DatetimeIndex of length T.
    """

    y: np.ndarray = Field(repr=False)
    name: str
    index: pd.DatetimeIndex = Field(repr=False)

    _MIN_OBS: int = 24  # two years of monthly data; SV estimation is uninformative below this

    @model_validator(mode="after")
    def _validate(self) -> Self:
        self._validate_shape()
        self._validate_length()
        self._validate_finite()
        self._validate_nonconstant()
        self._make_readonly()
        return self

    def _validate_shape(self) -> None:
        if self.y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {self.y.shape}")
        if len(self.index) != self.y.shape[0]:
            raise ValueError(f"index length {len(self.index)} != y length {self.y.shape[0]}")

    def _validate_length(self) -> None:
        if self.y.shape[0] < self._MIN_OBS:
            raise ValueError(
                f"y must have at least {self._MIN_OBS} observations for SV estimation, got {self.y.shape[0]}"
            )

    def _validate_finite(self) -> None:
        if not np.isfinite(self.y).all():
            raise ValueError("y contains NaN or Inf values")

    def _validate_nonconstant(self) -> None:
        if np.ptp(self.y) == 0.0:
            raise ValueError("y is constant; SV requires non-zero variance")

    def _make_readonly(self) -> None:
        y_readonly = self.y.astype(np.float64, copy=True)
        y_readonly.flags.writeable = False
        object.__setattr__(self, "y", y_readonly)

    @classmethod
    def from_series(cls, series: pd.Series, name: str | None = None) -> Self:
        """Construct SVData from a pandas Series with a DatetimeIndex.

        Args:
            series: Series with DatetimeIndex.
            name: Optional override for the series name. Defaults to series.name.

        Returns:
            Validated SVData instance.
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError(f"series must have a DatetimeIndex, got {type(series.index).__name__}")
        resolved_name = name if name is not None else series.name
        if resolved_name is None:
            raise ValueError("name is required when series has no .name attribute")
        return cls(
            y=series.to_numpy(dtype=np.float64),
            name=str(resolved_name),
            index=series.index,
        )
