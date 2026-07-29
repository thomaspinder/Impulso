"""VARData — validated, immutable data container for VAR models."""

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Self

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from impulso._base import ImpulsoBaseModel


def _duplicates(names: Iterable[str]) -> list[str]:
    """Return the values appearing more than once, in first-appearance order."""
    names = list(names)
    counts = Counter(names)
    return [name for name in dict.fromkeys(names) if counts[name] > 1]


def _format_names(names: Sequence[str]) -> str:
    """Render a name list for error messages."""
    return ", ".join(repr(name) for name in names)


class VARData(ImpulsoBaseModel):
    """Immutable, validated container for VAR estimation data.

    Variable names must be unique. `endog_names` and `exog_names` are each
    checked for internal duplicates, and the two must not share any name —
    a single label cannot refer to both an endogenous and an exogenous column.

    Attributes:
        endog: Endogenous variable array of shape (T, n) where T >= 1 and n >= 2.
        endog_names: Names for each endogenous variable. Must be unique.
        exog: Optional exogenous variable array of shape (T, k).
        exog_names: Names for each exogenous variable. Required if exog is provided.
            Must be unique and disjoint from `endog_names`.
        index: DatetimeIndex of length T.
    """

    endog: np.ndarray = Field(repr=False)
    endog_names: list[str]
    exog: np.ndarray | None = Field(default=None, repr=False)
    exog_names: list[str] | None = None
    index: pd.DatetimeIndex = Field(repr=False)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        t, n = self.endog.shape
        self._validate_shapes(t, n)
        self._validate_exog(t)
        self._validate_unique_names()
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

    def _validate_unique_names(self) -> None:
        endog_dupes = _duplicates(self.endog_names)
        if endog_dupes:
            raise ValueError(
                f"endog_names must be unique, got duplicates: {_format_names(endog_dupes)}. "
                "Rename the repeated columns (or drop the repeats) before constructing VARData — "
                "duplicate names make variable selection, plotting, and result labelling ambiguous."
            )
        if self.exog_names is None:
            return
        exog_dupes = _duplicates(self.exog_names)
        if exog_dupes:
            raise ValueError(
                f"exog_names must be unique, got duplicates: {_format_names(exog_dupes)}. "
                "Rename the repeated columns (or drop the repeats) before constructing VARData."
            )
        endog_set = set(self.endog_names)
        overlap = [name for name in self.exog_names if name in endog_set]
        if overlap:
            raise ValueError(
                f"endog_names and exog_names must not overlap, got shared names: {_format_names(overlap)}. "
                "A variable cannot be both endogenous and exogenous; rename one of the columns."
            )

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

        Column names must be unique within `endog`, within `exog`, and across
        the two — pandas silently widens the selection when a label is repeated
        or when `df` itself carries duplicate column labels, which would produce
        arrays that no longer match their names.

        Args:
            df: DataFrame with a DatetimeIndex.
            endog: Column names for endogenous variables.
            exog: Column names for exogenous variables (optional).

        Returns:
            Validated VARData instance.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"DataFrame must have a DatetimeIndex, got {type(df.index).__name__}")

        cls._validate_df_columns(df, endog, exog)

        endog_arr = df[endog].to_numpy(dtype=np.float64)
        exog_arr = df[exog].to_numpy(dtype=np.float64) if exog is not None else None

        return cls(
            endog=endog_arr,
            endog_names=endog,
            exog=exog_arr,
            exog_names=exog,
            index=df.index,
        )

    @staticmethod
    def _validate_df_columns(df: pd.DataFrame, endog: list[str], exog: list[str] | None) -> None:
        """Reject selections that pandas would silently widen into misaligned arrays."""
        requested = [*endog, *(exog or [])]
        df_dupes = _duplicates(name for name in df.columns if name in set(requested))
        if df_dupes:
            raise ValueError(
                f"DataFrame has duplicate column labels for selected variables: {_format_names(df_dupes)}. "
                "pandas returns every matching column, so the extracted array would not line up with its "
                "names. De-duplicate the DataFrame columns before calling from_df."
            )
