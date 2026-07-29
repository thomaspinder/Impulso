"""Result objects for VAR post-estimation output."""

from abc import abstractmethod
from typing import ClassVar, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from pydantic import Field, model_validator

from impulso._base import ImpulsoBaseModel
from impulso.scenario import MomentTarget, ProbabilityTarget, ShockPath, VariablePath

# Targets accepted by the tilting entry points.
Target = ProbabilityTarget | MomentTarget


def _wide_frame(da: xr.DataArray, row_dim: str, col_dim: str = "shock") -> pd.DataFrame:
    """Reshape a (row_dim, response, col_dim) DataArray into a wide DataFrame.

    The returned frame is indexed by the coord values of `row_dim` and has
    a `MultiIndex(['response', col_dim])` on columns built from those coords
    in the order they appear on the DataArray.

    Args:
        da: Source array carrying `response`, `col_dim`, and `row_dim` dims.
        row_dim: Dimension to use as the frame index.
        col_dim: Dimension pairing with `response` on the columns. Defaults
            to `shock` for structural results; exogenous results pass `exog`.
    """
    da = da.transpose(row_dim, "response", col_dim)
    row_values = da.coords[row_dim].values
    row_index = pd.DatetimeIndex(row_values, name="time") if row_dim == "time" else pd.Index(row_values, name=row_dim)
    columns = pd.MultiIndex.from_product(
        [da.coords["response"].values.tolist(), da.coords[col_dim].values.tolist()],
        names=["response", col_dim],
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

    def tilt(self, targets: list[Target], ess_warn_fraction: float = 0.1) -> "TiltedForecastResult":
        """Reweight these draws to satisfy distributional targets (entropic tilting).

        See `TiltedForecastResult` for what comes back and
        `ADR-0009` for why this is a post-hoc reweighting layer rather
        than a re-solve.

        Args:
            targets: `ProbabilityTarget` / `MomentTarget` list. A target
                repeated verbatim is kept once; two targets constraining
                the same quantity with different values are rejected.
            ess_warn_fraction: Warn when the effective sample size falls
                below this fraction of the draw count. Default 0.1.

        Returns:
            TiltedForecastResult carrying these draws by reference plus
            the tilting weights and diagnostics.

        Raises:
            ValueError: If this is a mean forecast, or if the targets are
                unachievable by reweighting these draws.
        """
        from impulso._tilting import tilt_result

        return tilt_result(self, list(targets), ess_warn_fraction)


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


class DynamicMultiplierResult(VARResultBase):
    """Result from exogenous dynamic-multiplier computation.

    Attributes:
        idata: ArviZ InferenceData with dynamic-multiplier draws.
        horizon: Highest horizon computed; the result spans 0..horizon.
        var_names: Names of the endogenous (response) variables.
        exog_names: Names of the exogenous (driver) variables.
        cumulative: Whether the draws are cumulative (step-response)
            multipliers rather than per-horizon impulse multipliers.
    """

    _PRIMARY_KEY: ClassVar[str] = "dynamic_multiplier"

    horizon: int
    var_names: list[str]
    exog_names: list[str]
    cumulative: bool = False

    def median(self) -> pd.DataFrame:
        """Posterior median dynamic multiplier.

        Returns:
            DataFrame indexed by horizon (integer 0..H) with a
            `MultiIndex(['response', 'exog'])` on columns.
        """
        self._guard_no_time_dim()
        dm = self.idata.posterior_predictive["dynamic_multiplier"]
        return _wide_frame(dm.median(dim=("chain", "draw")), "horizon", col_dim="exog")

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for the dynamic multiplier.

        Returns:
            HDIResult whose `lower` / `upper` DataFrames mirror the shape and
            labels of `median()`.
        """
        self._guard_no_time_dim()
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["dynamic_multiplier"]
        lower = _wide_frame(hdi_data.sel(hdi="lower"), "horizon", col_dim="exog")
        upper = _wide_frame(hdi_data.sel(hdi="higher"), "horizon", col_dim="exog")
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the dynamic multiplier to a DataFrame (passthrough to `median()`)."""
        return self.median()

    def plot(self) -> Figure:
        """Plot dynamic multipliers."""
        from impulso.plotting import plot_dynamic_multiplier

        return plot_dynamic_multiplier(self)


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
    """Result from the propagated historical decomposition.

    The posterior-predictive Dataset carries two variables: `"hd"` — the
    propagated contribution of each structural shock — and `"baseline"` —
    the deterministic path implied by the initial conditions, intercept,
    and any exogenous regressors with all shocks set to zero. Baseline plus
    the contributions summed over shocks reproduces the observed series
    exactly for every posterior draw.

    Attributes:
        idata: ArviZ InferenceData with decomposition draws.
        var_names: Names of variables.
    """

    var_names: list[str]

    def baseline(self) -> pd.DataFrame:
        """Posterior median of the deterministic baseline path.

        Returns:
            DataFrame indexed by the same `DatetimeIndex` as `median()`,
            with one column per variable.

        Raises:
            ValueError: If the result carries no `"baseline"` variable
                (e.g. a hand-built result predating the propagated
                decomposition).
        """
        if "baseline" not in self.idata.posterior_predictive:
            raise ValueError(
                "This result carries no 'baseline' variable; it was not "
                "produced by IdentifiedVAR.historical_decomposition."
            )
        da = self.idata.posterior_predictive["baseline"]
        med = da.median(dim=("chain", "draw")).transpose("time", "response")
        index = pd.DatetimeIndex(da.coords["time"].values, name="time")
        return pd.DataFrame(med.values, index=index, columns=self.var_names)

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


class ConditionalForecastResult(VARResultBase):
    """Result from conditional forecasting.

    The posterior-predictive Dataset carries `"forecast"`
    (chain, draw, step, variable) plus the per-draw plausibility
    statistics `"plausibility"` (`q`, chi-squared reference) and
    `"plausibility_calibrated"` (`q_cal ∈ [0.5, 1]`, ADPRR binomial
    calibration); Dataset attrs hold `n_restrictions` and the chi-squared
    tail probability of the median `q`.

    Attributes:
        idata: ArviZ InferenceData with the draws and statistics.
        steps: Number of forecast steps.
        var_names: Names of forecasted variables.
        mode: `"density"` or `"mean"`.
        path_uncertainty: `"none"` (hard pins) or `"unconditional"`.
        conditions: The `VariablePath` conditions echoed from the call.
    """

    steps: int
    var_names: list[str]
    mode: str = "density"
    path_uncertainty: Literal["none", "unconditional"] = "none"
    conditions: list[VariablePath] = Field(default_factory=list, repr=False)

    def median(self) -> pd.DataFrame:
        """Posterior median conditional forecast (step-indexed)."""
        forecast = self.idata.posterior_predictive["forecast"]
        med = forecast.median(dim=("chain", "draw")).values
        df = pd.DataFrame(med, columns=self.var_names)
        df.index.name = "step"
        return df

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for the conditional forecast.

        Args:
            prob: Probability mass for the HDI. Default 0.89.

        Returns:
            HDIResult whose `lower` / `upper` DataFrames mirror `median()`.
        """
        da = self.idata.posterior_predictive["forecast"]
        hdi_data = az.hdi(da, hdi_prob=prob)["forecast"]
        lower = pd.DataFrame(hdi_data.sel(hdi="lower").values, columns=self.var_names)
        upper = pd.DataFrame(hdi_data.sel(hdi="higher").values, columns=self.var_names)
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def plausibility(self, prob: float = 0.89) -> dict[str, float]:
        """Posterior summary of the plausibility statistic.

        `q` is the per-draw squared Mahalanobis distance of the pinned
        values from their unconditional law (`chi^2_r` reference when all
        shocks adjust); `q_cal` is the ADPRR-calibrated companion on
        `[0.5, 1]` — values near 1 flag scenarios the model considers
        incredible.

        Args:
            prob: Probability mass for the HDI bounds. Default 0.89.

        Returns:
            Dict with `q_median`, `q_hdi_lower`, `q_hdi_upper`,
            `q_calibrated_median`, `n_restrictions`, and
            `tail_probability` (`P(chi^2_r >= median q)`; 1.0 with no
            restrictions).
        """
        pp = self.idata.posterior_predictive
        q = pp["plausibility"]
        hdi_q = az.hdi(q, hdi_prob=prob)["plausibility"]
        return {
            "q_median": float(q.median()),
            "q_hdi_lower": float(hdi_q.sel(hdi="lower")),
            "q_hdi_upper": float(hdi_q.sel(hdi="higher")),
            "q_calibrated_median": float(pp["plausibility_calibrated"].median()),
            "n_restrictions": int(pp.attrs["n_restrictions"]),
            "tail_probability": float(pp.attrs["chi2_tail_of_median"]),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Conditional-forecast posterior median as a DataFrame (passthrough to `median()`)."""
        return self.median()

    def plot(self) -> Figure:
        """Plot the conditional forecast fan chart with pinned values marked."""
        from impulso.plotting import plot_conditional_forecast

        return plot_conditional_forecast(self)

    def tilt(self, targets: list[Target], ess_warn_fraction: float = 0.1) -> "TiltedForecastResult":
        """Reweight these draws to satisfy distributional targets (entropic tilting).

        Chaining hard conditioning with soft targets is the supported way
        to mix them: the pins hold pathwise on *every* draw here, and
        reweighting never moves a draw, so the pins survive the tilt
        exactly — a theorem, not a code path.

        Args:
            targets: `ProbabilityTarget` / `MomentTarget` list. A target
                repeated verbatim is kept once; two targets constraining
                the same quantity with different values are rejected.
            ess_warn_fraction: Warn when the effective sample size falls
                below this fraction of the draw count. Default 0.1.

        Returns:
            TiltedForecastResult carrying these draws by reference plus
            the tilting weights and diagnostics.

        Raises:
            ValueError: If this is a mean forecast, or if the targets are
                unachievable by reweighting these draws.
        """
        from impulso._tilting import tilt_result

        return tilt_result(self, list(targets), ess_warn_fraction)


class ScenarioResult(ConditionalForecastResult):
    """Result from structural scenario analysis.

    Extends `ConditionalForecastResult` with the structural ingredients —
    the adjusting set and any prescribed shock paths; the forecast and
    plausibility surface (`median`, `hdi`, `plausibility`) is inherited.
    The per-draw `plausibility` includes the prescribed shocks' own
    magnitude `|v_S|^2` in one-standard-deviation units; the chi-squared
    tail probability in attrs refers to the condition-only part.

    Attributes:
        adjusting: Names of the shocks permitted to absorb the conditions,
            echoed verbatim from the call: `None` means every shock was
            free to adjust; an empty list means none were (the pure
            substitution case).
        shocks: The prescribed `ShockPath` sequences echoed from the call.
    """

    adjusting: list[str] | None = None
    shocks: list[ShockPath] = Field(default_factory=list, repr=False)

    def plot(self) -> Figure:
        """Plot the structural scenario fan chart with pinned values marked."""
        from impulso.plotting import plot_structural_scenario

        return plot_structural_scenario(self)


class _WeightedResultMixin:
    """Shared weighted summaries for tilting-derived results.

    Both tilted forecasts and reverse-stress results summarise the same
    `"forecast"` draws under a `"tilting_weights"` variable, so the
    weighted median / HDI / DataFrame surface is written once here.
    """

    def _weights_flat(self) -> np.ndarray:
        """Normalised tilting weights flattened over `(chain, draw)`."""
        return self.idata.posterior_predictive["tilting_weights"].values.ravel()

    def _forecast_flat(self) -> np.ndarray:
        """Forecast draws reshaped to `(N, steps, n_vars)`."""
        da = self.idata.posterior_predictive["forecast"]
        n_chains, n_draws = da.shape[:2]
        return da.values.reshape(n_chains * n_draws, *da.shape[2:])

    @property
    def weights(self) -> np.ndarray:
        """Tilting weights, shape `(chain, draw)`, summing to 1."""
        return self.idata.posterior_predictive["tilting_weights"].values

    def median(self) -> pd.DataFrame:
        """Weighted posterior median forecast (step-indexed)."""
        from impulso._tilting import weighted_quantile

        med = weighted_quantile(self._forecast_flat(), self._weights_flat(), 0.5)
        df = pd.DataFrame(np.asarray(med), columns=self.var_names)
        df.index.name = "step"
        return df

    def base_median(self) -> pd.DataFrame:
        """Untilted posterior median, for comparison against `median()`."""
        med = self.idata.posterior_predictive["forecast"].median(dim=("chain", "draw")).values
        df = pd.DataFrame(med, columns=self.var_names)
        df.index.name = "step"
        return df

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """Weighted highest-density interval for the forecast.

        Args:
            prob: Probability mass for the HDI. Default 0.89.

        Returns:
            HDIResult whose `lower` / `upper` DataFrames mirror `median()`.
        """
        from impulso._tilting import weighted_hdi

        lower, upper = weighted_hdi(self._forecast_flat(), self._weights_flat(), prob)
        return HDIResult(
            lower=pd.DataFrame(lower, columns=self.var_names),
            upper=pd.DataFrame(upper, columns=self.var_names),
            prob=prob,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Weighted posterior median as a DataFrame (passthrough to `median()`)."""
        return self.median()


class TiltedForecastResult(_WeightedResultMixin, VARResultBase):
    """Forecast draws reweighted by entropic tilting (ADR-0009).

    The posterior-predictive Dataset carries `"forecast"` — the parent
    result's draws, held by reference, never copied or moved — plus
    `"tilting_weights"` `(chain, draw)` and the per-target
    `"requested"` / `"achieved"` / `"event_draws"` vectors over a
    `target` coordinate. Dataset attrs hold `ess`, `ess_fraction`, and
    `kl_divergence`.

    Every summary on this object is weighted: `median()` and `hdi()`
    read the tilted distribution, while `base_median()` returns the
    untilted median so the two can be compared directly.

    Attributes:
        idata: InferenceData with the parent draws and the weights.
        steps: Number of forecast steps.
        var_names: Names of forecasted variables.
        targets: The targets echoed from the call.
    """

    steps: int
    var_names: list[str]
    targets: list[Target] = Field(default_factory=list, repr=False)

    def summary(self) -> dict[str, object]:
        """Diagnostics and per-target achievement.

        Returns:
            Dict with `ess`, `ess_fraction`, `kl_divergence`, `n_draws`,
            and a `targets` list of per-target dicts holding `target`,
            `requested`, `achieved`, and `draws_in_event` (`None` for
            moment targets).
        """
        pp = self.idata.posterior_predictive
        rows = []
        for k, label in enumerate(pp["target"].values.tolist()):
            count = float(pp["event_draws"].values[k])
            rows.append({
                "target": label,
                "requested": float(pp["requested"].values[k]),
                "achieved": float(pp["achieved"].values[k]),
                "draws_in_event": None if np.isnan(count) else int(count),
            })
        return {
            "ess": float(pp.attrs["ess"]),
            "ess_fraction": float(pp.attrs["ess_fraction"]),
            "kl_divergence": float(pp.attrs["kl_divergence"]),
            "n_draws": int(pp["tilting_weights"].size),
            "targets": rows,
        }

    def plot(self) -> Figure:
        """Plot the tilted fan chart against the untilted median."""
        from impulso.plotting import plot_tilted_forecast

        return plot_tilted_forecast(self)


class ReverseStressResult(_WeightedResultMixin, VARResultBase):
    """Shock cocktail behind a stress event, from reverse stress testing.

    The posterior-predictive Dataset carries `"forecast"`
    (chain, draw, step, variable), the structural shocks that generated
    those draws (`"structural_shocks"`, chain, draw, step, shock), the
    `"tilting_weights"` that condition on the event, and the
    `"shock_cocktail"` (step, shock) — the tilted-weighted mean of the
    retained structural shocks, in one-standard-deviation units. Dataset
    attrs hold `baseline_probability`, `achieved_probability`, `ess`,
    `ess_fraction`, `kl_divergence`, `q`, and `q_cal`.

    Attributes:
        idata: InferenceData with the draws, weights, and cocktail.
        steps: Number of forecast steps.
        var_names: Names of forecasted variables.
        shock_names: Structural shock coordinate labels.
        variable: The stressed variable.
        threshold: The stress threshold, in the variable's units.
        horizon: The 1-based forecast step the event refers to.
        direction: `"below"` or `"above"`.
        probability: Requested probability of the stress event.
    """

    steps: int
    var_names: list[str]
    shock_names: list[str]
    variable: str
    threshold: float
    horizon: int
    direction: Literal["below", "above"] = "below"
    probability: float = 1.0

    def shock_cocktail(self) -> pd.DataFrame:
        """The shock cocktail as a step-indexed DataFrame.

        Returns:
            DataFrame indexed by forecast step (1-based) with one column
            per structural shock, in one-standard-deviation units.
        """
        da = self.idata.posterior_predictive["shock_cocktail"]
        df = pd.DataFrame(da.values, columns=self.shock_names, index=pd.RangeIndex(1, self.steps + 1, name="step"))
        return df

    def summary(self) -> dict[str, float]:
        """Event probabilities, tilt diagnostics, and cocktail plausibility.

        Returns:
            Dict with `baseline_probability`, `requested_probability`,
            `achieved_probability`, `ess`, `ess_fraction`,
            `kl_divergence`, `n_draws`, `q`, and `q_cal`.
        """
        pp = self.idata.posterior_predictive
        return {
            "baseline_probability": float(pp.attrs["baseline_probability"]),
            "requested_probability": float(self.probability),
            "achieved_probability": float(pp.attrs["achieved_probability"]),
            "ess": float(pp.attrs["ess"]),
            "ess_fraction": float(pp.attrs["ess_fraction"]),
            "kl_divergence": float(pp.attrs["kl_divergence"]),
            "n_draws": int(pp["tilting_weights"].size),
            "q": float(pp.attrs["q"]),
            "q_cal": float(pp.attrs["q_cal"]),
        }

    def plot(self) -> Figure:
        """Plot the stressed variable's tilted fan and the shock cocktail."""
        from impulso.plotting import plot_reverse_stress

        return plot_reverse_stress(self)


class CounterfactualResult(VARResultBase):
    """Historical counterfactual paths alongside the actual data.

    The posterior-predictive Dataset carries `"counterfactual"`
    (chain, draw, time, variable) and `"actual"` (time, variable) over the
    same returned window. Counterfactual draws are built from the realised
    structural shocks — edited, never re-drawn — so their spread reflects
    parameter and identification uncertainty only.

    Attributes:
        idata: ArviZ InferenceData with counterfactual draws + actual path.
        var_names: Names of variables.
    """

    var_names: list[str]

    def _time_index(self) -> pd.DatetimeIndex:
        values = self.idata.posterior_predictive["counterfactual"].coords["time"].values
        return pd.DatetimeIndex(values, name="time")

    def median(self) -> pd.DataFrame:
        """Posterior median counterfactual path.

        Returns:
            DataFrame indexed by the returned window's `DatetimeIndex`
            with one column per variable.
        """
        da = self.idata.posterior_predictive["counterfactual"]
        med = da.median(dim=("chain", "draw")).transpose("time", "variable")
        return pd.DataFrame(med.values, index=self._time_index(), columns=self.var_names)

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for the counterfactual path.

        Args:
            prob: Probability mass for the HDI. Default 0.89.

        Returns:
            HDIResult whose `lower` / `upper` DataFrames mirror `median()`.
        """
        da = self.idata.posterior_predictive["counterfactual"]
        hdi_data = az.hdi(da, hdi_prob=prob)["counterfactual"]
        index = self._time_index()
        lower = pd.DataFrame(hdi_data.sel(hdi="lower").values, index=index, columns=self.var_names)
        upper = pd.DataFrame(hdi_data.sel(hdi="higher").values, index=index, columns=self.var_names)
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def actual(self) -> pd.DataFrame:
        """The observed path over the returned window.

        Returns:
            DataFrame shaped like `median()`.
        """
        da = self.idata.posterior_predictive["actual"]
        return pd.DataFrame(da.values, index=self._time_index(), columns=self.var_names)

    def difference(self) -> pd.DataFrame:
        """Posterior median effect of the edits: `actual - counterfactual`.

        The actual path is constant across draws, so
        `actual - median(counterfactual)` equals
        `median(actual - counterfactual)` exactly.

        Returns:
            DataFrame shaped like `median()`.
        """
        return self.actual() - self.median()

    def to_dataframe(self) -> pd.DataFrame:
        """Counterfactual posterior median as a DataFrame (passthrough to `median()`)."""
        return self.median()

    def plot(self) -> Figure:
        """Plot actual vs counterfactual paths with HDI bands."""
        from impulso.plotting import plot_counterfactual

        return plot_counterfactual(self)


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
        index: Forecast axis, normally supplied by `FittedSV.forecast` — a
            `DatetimeIndex` continuing the observed calendar when the data's
            frequency is detectable. `None` (the default) falls back to a
            step-numbered `RangeIndex`.
    """

    series_name: str
    steps: int
    index: pd.Index | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def _check_index_length(self) -> "SVForecastResult":
        """Reject an index that does not match the number of forecast steps."""
        if self.index is not None and len(self.index) != self.steps:
            raise ValueError(f"index length {len(self.index)} != steps {self.steps}")
        return self

    def _axis(self) -> pd.Index:
        """Forecast axis, falling back to a step-numbered RangeIndex."""
        from impulso._time import forecast_index

        return self.index if self.index is not None else forecast_index(None, self.steps)

    def median(self) -> pd.DataFrame:
        """Posterior median of the density forecast.

        Returns:
            DataFrame of median forecasts indexed by the forecast axis —
            calendar dates when available, otherwise step number.
        """
        forecast = self.idata.posterior_predictive["forecast"]
        med = forecast.median(dim=("chain", "draw")).values
        return pd.DataFrame({self.series_name: med}, index=self._axis())

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """Highest-density interval for the density forecast.

        Args:
            prob: Probability mass for the HDI. Default 0.89.

        Returns:
            HDIResult with lower/upper DataFrames sharing the index of
            `median()`.
        """
        hdi_data = az.hdi(self.idata.posterior_predictive, hdi_prob=prob)["forecast"]
        axis = self._axis()
        lower = pd.DataFrame({self.series_name: hdi_data.sel(hdi="lower").values}, index=axis)
        upper = pd.DataFrame({self.series_name: hdi_data.sel(hdi="higher").values}, index=axis)
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        """Density forecast posterior median as a DataFrame.

        Returns:
            DataFrame of median forecasts indexed by the forecast axis.
        """
        return self.median()

    def plot(self) -> Figure:
        """Plot the density forecast with HDI bands.

        Returns:
            Matplotlib Figure of the density forecast.
        """
        from impulso.plotting import plot_sv_forecast

        return plot_sv_forecast(self)
