"""Result objects for VAR post-estimation output."""

from abc import abstractmethod

import arviz as az
import pandas as pd
from matplotlib.figure import Figure
from pydantic import Field

from impulso._base import ImpulsoBaseModel


class HDIResult(ImpulsoBaseModel):
    """Structured HDI output with separate lower/upper bounds.

    Attributes:
        lower: DataFrame of lower HDI bounds.
        upper: DataFrame of upper HDI bounds.
        prob: HDI probability level.
    """

    lower: pd.DataFrame
    upper: pd.DataFrame
    prob: float


class VARResultBase(ImpulsoBaseModel):
    """Base class for VAR post-estimation results.

    Attributes:
        idata: ArviZ InferenceData holding the result draws.
    """

    idata: az.InferenceData = Field(repr=False)

    @abstractmethod
    def median(self) -> pd.DataFrame:
        """Compute posterior median of the result."""
        raise NotImplementedError

    @abstractmethod
    def hdi(self, prob: float = 0.89) -> HDIResult:
        """Compute highest density interval.

        Args:
            prob: Probability mass for the HDI. Default 0.89.
        """
        raise NotImplementedError

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Convert result to a tidy DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def plot(self) -> Figure:
        """Plot the result. Subclasses must implement."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ForecastResult(VARResultBase):
    """Result from VAR forecasting.

    Attributes:
        idata: ArviZ InferenceData with forecast draws.
        steps: Number of forecast steps.
        var_names: Names of forecasted variables.
    """

    steps: int
    var_names: list[str]

    def median(self) -> pd.DataFrame:
        """Posterior median forecast."""
        forecast = self.idata.posterior_predictive["forecast"]
        med = forecast.median(dim=("chain", "draw")).values
        return pd.DataFrame(med, columns=self.var_names)

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for forecast."""
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["forecast"]
        lower = pd.DataFrame(hdi_data.sel(hdi="lower").values, columns=self.var_names)
        upper = pd.DataFrame(hdi_data.sel(hdi="higher").values, columns=self.var_names)
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to long-format DataFrame."""
        forecast = self.idata.posterior_predictive["forecast"]
        med = forecast.median(dim=("chain", "draw")).values
        df = pd.DataFrame(med, columns=self.var_names)
        df.index.name = "step"
        return df

    def plot(self) -> Figure:
        """Plot forecast fan chart."""
        from impulso.plotting import plot_forecast

        return plot_forecast(self)


class IRFResult(VARResultBase):
    """Result from impulse response function computation.

    Attributes:
        idata: ArviZ InferenceData with IRF draws.
        horizon: Number of IRF horizons.
        var_names: Names of variables.
    """

    horizon: int
    var_names: list[str]

    def median(self) -> pd.DataFrame:
        """Posterior median IRF."""
        irf = self.idata.posterior_predictive["irf"]
        return pd.DataFrame(irf.median(dim=("chain", "draw")).values.reshape(self.horizon + 1, -1))

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for IRF."""
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["irf"]
        lower = pd.DataFrame(hdi_data.sel(hdi="lower").values.reshape(self.horizon + 1, -1))
        upper = pd.DataFrame(hdi_data.sel(hdi="higher").values.reshape(self.horizon + 1, -1))
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert IRF to DataFrame."""
        return self.median()

    def plot(self) -> Figure:
        """Plot impulse response functions."""
        from impulso.plotting import plot_irf

        return plot_irf(self)


class FEVDResult(VARResultBase):
    """Result from forecast error variance decomposition.

    Attributes:
        idata: ArviZ InferenceData with FEVD draws.
        horizon: Number of FEVD horizons.
        var_names: Names of variables.
    """

    horizon: int
    var_names: list[str]

    def median(self) -> pd.DataFrame:
        """Posterior median FEVD."""
        fevd = self.idata.posterior_predictive["fevd"]
        return pd.DataFrame(fevd.median(dim=("chain", "draw")).values.reshape(self.horizon + 1, -1))

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for FEVD."""
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["fevd"]
        lower = pd.DataFrame(hdi_data.sel(hdi="lower").values.reshape(self.horizon + 1, -1))
        upper = pd.DataFrame(hdi_data.sel(hdi="higher").values.reshape(self.horizon + 1, -1))
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert FEVD to DataFrame."""
        return self.median()

    def plot(self) -> Figure:
        """Plot FEVD."""
        from impulso.plotting import plot_fevd

        return plot_fevd(self)


class HistoricalDecompositionResult(VARResultBase):
    """Result from historical decomposition.

    Attributes:
        idata: ArviZ InferenceData with decomposition draws.
        var_names: Names of variables.
    """

    var_names: list[str]

    def median(self) -> pd.DataFrame:
        """Posterior median historical decomposition."""
        hd = self.idata.posterior_predictive["hd"]
        return pd.DataFrame(hd.median(dim=("chain", "draw")).values.reshape(-1, len(self.var_names)))

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for historical decomposition."""
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["hd"]
        lower = pd.DataFrame(hdi_data.sel(hdi="lower").values.reshape(-1, len(self.var_names)))
        upper = pd.DataFrame(hdi_data.sel(hdi="higher").values.reshape(-1, len(self.var_names)))
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert historical decomposition to DataFrame."""
        return self.median()

    def plot(self) -> Figure:
        """Plot historical decomposition."""
        from impulso.plotting import plot_historical_decomposition

        return plot_historical_decomposition(self)


class LagOrderResult(ImpulsoBaseModel):
    """Result from lag order selection.

    Attributes:
        aic: Optimal lag order by AIC.
        bic: Optimal lag order by BIC.
        hq: Optimal lag order by Hannan-Quinn.
        criteria_table: DataFrame of all criteria values by lag order.
    """

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
