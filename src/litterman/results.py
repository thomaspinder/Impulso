"""Result objects for VAR post-estimation output."""

from __future__ import annotations

from abc import abstractmethod

import arviz as az
import pandas as pd
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field


class HDIResult(BaseModel):
    """Structured HDI output with separate lower/upper bounds.

    Args:
        lower: DataFrame of lower HDI bounds.
        upper: DataFrame of upper HDI bounds.
        prob: HDI probability level.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    lower: pd.DataFrame
    upper: pd.DataFrame
    prob: float


class VARResultBase(BaseModel):
    """Base class for VAR post-estimation results.

    Args:
        idata: ArviZ InferenceData holding the result draws.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    idata: az.InferenceData

    def median(self) -> pd.DataFrame:
        """Compute posterior median of the result."""
        raise NotImplementedError

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """Compute highest density interval.

        Args:
            prob: Probability mass for the HDI. Default 0.89.
        """
        raise NotImplementedError

    def to_dataframe(self) -> pd.DataFrame:
        """Convert result to a tidy DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def plot(self) -> Figure:
        """Plot the result. Subclasses must implement."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class LagOrderResult(BaseModel):
    """Result from lag order selection.

    Args:
        aic: Optimal lag order by AIC.
        bic: Optimal lag order by BIC.
        hq: Optimal lag order by Hannan-Quinn.
        criteria_table: DataFrame of all criteria values by lag order.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    aic: int
    bic: int
    hq: int
    criteria_table: pd.DataFrame = Field(repr=False)

    def summary(self) -> pd.DataFrame:
        """Return the full criteria table.

        Returns:
            DataFrame with information criteria for each lag order.
        """
        return self.criteria_table
