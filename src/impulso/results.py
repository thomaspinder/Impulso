"""Result objects for VAR post-estimation output."""

from abc import abstractmethod
from typing import ClassVar

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from pydantic import Field

from impulso._base import ImpulsoBaseModel


def _wide_frame(da: xr.DataArray, row_dim: str) -> pd.DataFrame:
    """Reshape a (row_dim, response, shock) DataArray into a wide DataFrame.

    The returned frame is indexed by the coord values of `row_dim` and has
    a `MultiIndex(['response', 'shock'])` on columns built from those coords
    in the order they appear on the DataArray.
    """
    da = da.transpose(row_dim, "response", "shock")
    row_values = da.coords[row_dim].values
    row_index = pd.DatetimeIndex(row_values, name="time") if row_dim == "time" else pd.Index(row_values, name=row_dim)
    columns = pd.MultiIndex.from_product(
        [da.coords["response"].values.tolist(), da.coords["shock"].values.tolist()],
        names=["response", "shock"],
    )
    return pd.DataFrame(da.values.reshape(len(row_index), -1), index=row_index, columns=columns)


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

    Subclasses that hold a single named DataArray in
    `idata.posterior_predictive` (IRF, FEVD) declare its key via the
    class-level `_PRIMARY_KEY`; this drives the shared
    `_guard_no_time_dim` check.

    Attributes:
        idata: ArviZ InferenceData holding the result draws.
    """

    idata: az.InferenceData = Field(repr=False)

    # Empty default — subclasses with a `time`-aware median override it.
    _PRIMARY_KEY: ClassVar[str] = ""

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

    def _guard_no_time_dim(self) -> None:
        """Refuse `median`/`hdi`/`to_dataframe` on a time-aware result.

        The reshape-based aggregations assume a 5-D `(C, D, H+1, n, n)`
        DataArray. For `at='all'` the array is 6-D
        `(C, D, T, H+1, n, n)` and `.reshape(H+1, -1)` would silently
        scramble the time and variable dims into the column axis. Refuse
        instead and point the user at the underlying DataArray.
        """
        key = self._PRIMARY_KEY
        if not key:
            raise NotImplementedError(
                f"{type(self).__name__} did not declare _PRIMARY_KEY; the time-dim guard cannot be evaluated."
            )
        if "time" in self.idata.posterior_predictive[key].dims:
            cls_name = type(self).__name__
            raise NotImplementedError(
                f"{cls_name}.median()/hdi()/to_dataframe() do not support "
                f"time-varying {key.upper()}s (at='all'). Access the "
                f"underlying DataArray directly via "
                f"result.idata.posterior_predictive[{key!r}] and aggregate "
                f"manually, or use at='last' / at=<int> / at=None for a "
                f"single-time {key.upper()}."
            )


class ForecastResult(VARResultBase):
    """Result from VAR forecasting.

    Attributes:
        idata: ArviZ InferenceData with forecast draws.
        steps: Number of forecast steps.
        var_names: Names of forecasted variables.
        mode: ``"density"`` or ``"mean"`` — which forecast mode produced
            this result.
    """

    steps: int
    var_names: list[str]
    mode: str = "density"

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

    _PRIMARY_KEY: ClassVar[str] = "irf"

    horizon: int
    var_names: list[str]

    def median(self) -> pd.DataFrame:
        """Posterior median IRF.

        Returns:
            DataFrame indexed by horizon (integer 0..H) with a
            `MultiIndex(['response', 'shock'])` on columns.
        """
        self._guard_no_time_dim()
        irf = self.idata.posterior_predictive["irf"]
        return _wide_frame(irf.median(dim=("chain", "draw")), "horizon")

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for IRF.

        Returns:
            HDIResult whose `lower` / `upper` DataFrames mirror the shape and
            labels of `median()`.
        """
        self._guard_no_time_dim()
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["irf"]
        lower = _wide_frame(hdi_data.sel(hdi="lower"), "horizon")
        upper = _wide_frame(hdi_data.sel(hdi="higher"), "horizon")
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert IRF to DataFrame (passthrough to `median()`)."""
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

    _PRIMARY_KEY: ClassVar[str] = "fevd"

    horizon: int
    var_names: list[str]

    def median(self) -> pd.DataFrame:
        """Posterior median FEVD.

        Returns:
            DataFrame indexed by horizon (integer 0..H) with a
            `MultiIndex(['response', 'shock'])` on columns.
        """
        self._guard_no_time_dim()
        fevd = self.idata.posterior_predictive["fevd"]
        return _wide_frame(fevd.median(dim=("chain", "draw")), "horizon")

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for FEVD.

        Returns:
            HDIResult whose `lower` / `upper` DataFrames mirror the shape and
            labels of `median()`.
        """
        self._guard_no_time_dim()
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["fevd"]
        lower = _wide_frame(hdi_data.sel(hdi="lower"), "horizon")
        upper = _wide_frame(hdi_data.sel(hdi="higher"), "horizon")
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert FEVD to DataFrame (passthrough to `median()`)."""
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
        """Posterior median historical decomposition.

        Returns:
            DataFrame indexed by a `DatetimeIndex` over the in-sample period
            (after lag-trimming and any `start` / `end` filter applied at
            decomposition time), with a `MultiIndex(['response', 'shock'])`
            on columns.
        """
        hd = self.idata.posterior_predictive["hd"]
        return _wide_frame(hd.median(dim=("chain", "draw")), "time")

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for historical decomposition.

        Returns:
            HDIResult whose `lower` / `upper` DataFrames mirror the shape and
            labels of `median()`.
        """
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["hd"]
        lower = _wide_frame(hdi_data.sel(hdi="lower"), "time")
        upper = _wide_frame(hdi_data.sel(hdi="higher"), "time")
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert historical decomposition to DataFrame (passthrough to `median()`)."""
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


class VolatilityResult(VARResultBase):
    """Result from univariate SV fit — posterior of conditional SD.

    Conditional SD is sigma_t = exp(h_t / 2), where h_t is the
    posterior log-volatility path.

    Attributes:
        idata: InferenceData with 'h' in posterior.
        series_name: Name of the fitted series.
        index: DatetimeIndex aligned with the fitted series.
    """

    series_name: str
    index: pd.DatetimeIndex = Field(repr=False)

    def _sigma_da(self):
        """exp(h/2) DataArray over chains, draws, time."""
        return np.exp(0.5 * self.idata.posterior["h"])

    def median(self) -> pd.DataFrame:
        """Posterior median of the conditional SD path."""
        sigma = self._sigma_da()
        med = sigma.median(dim=("chain", "draw")).values
        return pd.DataFrame({self.series_name: med}, index=self.index)

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """Highest-density interval for the conditional SD path."""
        import xarray as xr

        sigma = self._sigma_da()
        # az.hdi expects a Dataset
        ds = xr.Dataset({"sigma": sigma})
        hdi_data = az.hdi(ds, hdi_prob=prob)["sigma"]
        lower = pd.DataFrame(
            {self.series_name: hdi_data.sel(hdi="lower").values},
            index=self.index,
        )
        upper = pd.DataFrame(
            {self.series_name: hdi_data.sel(hdi="higher").values},
            index=self.index,
        )
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        """Conditional SD posterior median as a DataFrame."""
        return self.median()

    def plot(self) -> Figure:
        """Plot the posterior volatility path with HDI bands."""
        from impulso.plotting import plot_volatility

        return plot_volatility(self)


class SVForecastResult(VARResultBase):
    """Density forecast from a univariate SV model.

    Attributes:
        idata: InferenceData with 'forecast' in posterior_predictive.
        series_name: Name of the forecast series.
        steps: Number of forecast steps.
    """

    series_name: str
    steps: int

    def median(self) -> pd.DataFrame:
        """Posterior median of the density forecast.

        Returns:
            DataFrame of median forecasts indexed by step.
        """
        forecast = self.idata.posterior_predictive["forecast"]
        med = forecast.median(dim=("chain", "draw")).values
        df = pd.DataFrame({self.series_name: med})
        df.index.name = "step"
        return df

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """Highest-density interval for the density forecast.

        Args:
            prob: Probability mass for the HDI. Default 0.89.

        Returns:
            HDIResult with lower/upper DataFrames for each forecast step.
        """
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["forecast"]
        lower = pd.DataFrame({self.series_name: hdi_data.sel(hdi="lower").values})
        upper = pd.DataFrame({self.series_name: hdi_data.sel(hdi="higher").values})
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        """Density forecast posterior median as a DataFrame.

        Returns:
            DataFrame of median forecasts indexed by step.
        """
        return self.median()

    def plot(self) -> Figure:
        """Plot the density forecast with HDI bands.

        Returns:
            Matplotlib Figure of the density forecast.
        """
        from impulso.plotting import plot_sv_forecast

        return plot_sv_forecast(self)
