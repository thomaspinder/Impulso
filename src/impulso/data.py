"""VARData — validated, immutable data container for VAR models."""

from typing import Self

from typing import Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator


class VARData(BaseModel):
    """Immutable, validated container for VAR estimation data.

    Attributes:
        endog: Endogenous variable array of shape (T, n) where T >= 1 and n >= 2.
        endog_names: Names for each endogenous variable.
        exog: Optional exogenous variable array of shape (T, k).
        exog_names: Names for each exogenous variable. Required if exog is provided.
        index: DatetimeIndex of length T.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    endog: np.ndarray
    endog_names: list[str]
    exog: np.ndarray | None = None
    exog_names: list[str] | None = None
    index: pd.DatetimeIndex

    @model_validator(mode="after")
    def _validate(self) -> Self:
        t, n = self.endog.shape
        self._validate_shapes(t, n)
        self._validate_exog(t)
        self._validate_finite()
        self._make_readonly()
        return self

    def _validate_shapes(self, t: int, n: int) -> None:
        if n < 2:
            raise ValueError(f"Minimum 2 endogenous variables required, got {n}")
        if len(self.endog_names) != n:
            raise ValueError(f"endog_names length {len(self.endog_names)} != endog columns {n}")
        if len(self.index) != t:
            raise ValueError(f"index length {len(self.index)} != endog rows {t}")

    def _validate_exog(self, t: int) -> None:
        if self.exog is not None:
            if self.exog.shape[0] != t:
                raise ValueError(f"exog rows {self.exog.shape[0]} != endog rows {t}")
            if self.exog_names is None:
                raise ValueError("exog_names required when exog is provided")
            if len(self.exog_names) != self.exog.shape[1]:
                raise ValueError(f"exog_names length {len(self.exog_names)} != exog columns {self.exog.shape[1]}")
        elif self.exog_names is not None:
            raise ValueError("exog_names provided without exog")

    def _validate_finite(self) -> None:
        if not np.isfinite(self.endog).all():
            raise ValueError("endog contains NaN or Inf values")
        if self.exog is not None and not np.isfinite(self.exog).all():
            raise ValueError("exog contains NaN or Inf values")

    def _make_readonly(self) -> None:
        endog_copy = self.endog.copy()
        endog_copy.flags.writeable = False
        object.__setattr__(self, "endog", endog_copy)
        if self.exog is not None:
            exog_copy = self.exog.copy()
            exog_copy.flags.writeable = False
            object.__setattr__(self, "exog", exog_copy)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        endog: list[str],
        exog: list[str] | None = None,
    ) -> Self:
        """Construct VARData from a pandas DataFrame.

        Args:
            df: DataFrame with a DatetimeIndex.
            endog: Column names for endogenous variables.
            exog: Column names for exogenous variables (optional).

        Returns:
            Validated VARData instance.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"DataFrame must have a DatetimeIndex, got {type(df.index).__name__}")

        endog_arr = df[endog].to_numpy(dtype=np.float64)
        exog_arr = df[exog].to_numpy(dtype=np.float64) if exog is not None else None

        return cls(
            endog=endog_arr,
            endog_names=endog,
            exog=exog_arr,
            exog_names=exog,
            index=df.index,
        )
