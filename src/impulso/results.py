"""Result objects for VAR post-estimation output."""

from abc import abstractmethod
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from pydantic import Field, model_validator

from impulso._arviz_compat import InferenceDataLike, get_group_dataset, hdi_bounds
from impulso._base import ImpulsoBaseModel
from impulso.scenario import ShockPath, VariablePath


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
        idata: InferenceData-schema container holding the result draws
            (`arviz.InferenceData` on ArviZ 0, `xarray.DataTree` on ArviZ 1).
    """

    idata: InferenceDataLike = Field(repr=False)

    # Empty default — subclasses with a `time`-aware median override it.
    _PRIMARY_KEY: ClassVar[str] = ""

    def _pp(self) -> xr.Dataset:
        """The `posterior_predictive` group as an `xarray.Dataset`.

        Normalises away the container difference between the two ArviZ
        lines; on ArviZ 1 the raw group is a `DataTree` node, not a Dataset.
        Bind the result to a local when a method reads it more than once.
        """
        return get_group_dataset(self.idata, "posterior_predictive")

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
        if "time" in self._pp()[key].dims:
            cls_name = type(self).__name__
            raise NotImplementedError(
                f"{cls_name}.median()/hdi()/to_dataframe() do not support "
                f"time-varying {key.upper()}s (at='all'). Access the "
                f"underlying DataArray directly via "
                f"result.idata.posterior_predictive[{key!r}] and aggregate "
                f"manually, or use at='last' / at=<int> / at=None for a "
                f"single-time {key.upper()}."
            )

    def _wide_median(self, key: str, row_dim: str, col_dim: str = "shock") -> pd.DataFrame:
        """Posterior median of `_pp()[key]` as a wide DataFrame.

        Args:
            key: Name of the variable in the `posterior_predictive` group.
            row_dim: Dimension to use as the frame index.
            col_dim: Dimension pairing with `response` on the columns.

        Returns:
            DataFrame indexed by `row_dim` with a
            `MultiIndex(['response', col_dim])` on columns.
        """
        da = self._pp()[key]
        return _wide_frame(da.median(dim=("chain", "draw")), row_dim, col_dim=col_dim)

    def _wide_hdi(self, key: str, prob: float, row_dim: str, col_dim: str = "shock") -> HDIResult:
        """HDI of `_pp()[key]` as wide lower/upper DataFrames.

        Args:
            key: Name of the variable in the `posterior_predictive` group.
            prob: Probability mass for the HDI.
            row_dim: Dimension to use as the frame index.
            col_dim: Dimension pairing with `response` on the columns.

        Returns:
            HDIResult whose `lower` / `upper` DataFrames mirror the shape
            and labels of `_wide_median(key, row_dim, col_dim)`.
        """
        lower_da, upper_da = hdi_bounds(self._pp()[key], prob)
        return HDIResult(
            lower=_wide_frame(lower_da, row_dim, col_dim=col_dim),
            upper=_wide_frame(upper_da, row_dim, col_dim=col_dim),
            prob=prob,
        )


class ForecastResult(VARResultBase):
    """Result from VAR forecasting.

    Attributes:
        idata: InferenceData-schema container with forecast draws.
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
        forecast = self._pp()["forecast"]
        med = forecast.median(dim=("chain", "draw")).values
        return pd.DataFrame(med, columns=self.var_names)

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for forecast."""
        lower_da, upper_da = hdi_bounds(self._pp()["forecast"], prob)
        lower = pd.DataFrame(lower_da.values, columns=self.var_names)
        upper = pd.DataFrame(upper_da.values, columns=self.var_names)
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to long-format DataFrame."""
        forecast = self._pp()["forecast"]
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
        idata: InferenceData-schema container with IRF draws.
        horizon: Number of IRF horizons.
        var_names: Names of variables.
    """

    _PRIMARY_KEY: ClassVar[str] = "irf"

    horizon: int
    var_names: list[str]

    @property
    def shock_names(self) -> list[str]:
        """Names of the structural shocks, in the order of the `shock` coordinate."""
        return [str(s) for s in self._pp()["irf"].coords["shock"].values]

    def median(self) -> pd.DataFrame:
        """Posterior median IRF.

        Returns:
            DataFrame indexed by horizon (integer 0..H) with a
            `MultiIndex(['response', 'shock'])` on columns.
        """
        self._guard_no_time_dim()
        return self._wide_median("irf", "horizon")

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for IRF.

        Returns:
            HDIResult whose `lower` / `upper` DataFrames mirror the shape and
            labels of `median()`.
        """
        self._guard_no_time_dim()
        return self._wide_hdi("irf", prob, "horizon")

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
        idata: InferenceData-schema container with dynamic-multiplier draws.
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
        return self._wide_median("dynamic_multiplier", "horizon", col_dim="exog")

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for the dynamic multiplier.

        Returns:
            HDIResult whose `lower` / `upper` DataFrames mirror the shape and
            labels of `median()`.
        """
        self._guard_no_time_dim()
        return self._wide_hdi("dynamic_multiplier", prob, "horizon", col_dim="exog")

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
        idata: InferenceData-schema container with FEVD draws.
        horizon: Number of FEVD horizons.
        var_names: Names of variables.
    """

    _PRIMARY_KEY: ClassVar[str] = "fevd"

    horizon: int
    var_names: list[str]

    @property
    def shock_names(self) -> list[str]:
        """Names of the structural shocks, in the order of the `shock` coordinate."""
        return [str(s) for s in self._pp()["fevd"].coords["shock"].values]

    def median(self) -> pd.DataFrame:
        """Posterior median FEVD.

        Returns:
            DataFrame indexed by horizon (integer 0..H) with a
            `MultiIndex(['response', 'shock'])` on columns.
        """
        self._guard_no_time_dim()
        return self._wide_median("fevd", "horizon")

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for FEVD.

        Returns:
            HDIResult whose `lower` / `upper` DataFrames mirror the shape and
            labels of `median()`.
        """
        self._guard_no_time_dim()
        return self._wide_hdi("fevd", prob, "horizon")

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
        idata: InferenceData-schema container with decomposition draws.
        var_names: Names of variables.
    """

    var_names: list[str]

    @property
    def shock_names(self) -> list[str]:
        """Names of the shock contributions, in the order of the `shock` coordinate.

        Partially-identified decompositions include the
        `unidentified_remainder` column.
        """
        return [str(s) for s in self._pp()["hd"].coords["shock"].values]

    def deviation(self) -> pd.DataFrame:
        """Posterior median of the total deviation from the deterministic baseline.

        Median of the per-draw sum of contributions over shocks, so it
        matches `data - baseline()` exactly; because median-of-sum differs
        from sum-of-medians, it need not equal `median()` summed over the
        shock column level.

        Returns:
            DataFrame indexed by the same `DatetimeIndex` as `median()`,
            with one column per variable.
        """
        da = self._pp()["hd"]
        med = da.sum("shock").median(dim=("chain", "draw")).transpose("time", "response")
        index = pd.DatetimeIndex(da.coords["time"].values, name="time")
        return pd.DataFrame(med.values, index=index, columns=self.var_names)

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
        pp = self._pp()
        if "baseline" not in pp:
            raise ValueError(
                "This result carries no 'baseline' variable; it was not "
                "produced by IdentifiedVAR.historical_decomposition."
            )
        da = pp["baseline"]
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
        return self._wide_median("hd", "time")

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for historical decomposition.

        Returns:
            HDIResult whose `lower` / `upper` DataFrames mirror the shape and
            labels of `median()`.
        """
        return self._wide_hdi("hd", prob, "time")

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
        idata: InferenceData-schema container with the draws and statistics.
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
        forecast = self._pp()["forecast"]
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
        lower_da, upper_da = hdi_bounds(self._pp()["forecast"], prob)
        lower = pd.DataFrame(lower_da.values, columns=self.var_names)
        upper = pd.DataFrame(upper_da.values, columns=self.var_names)
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
        pp = self._pp()
        q = pp["plausibility"]
        q_lower, q_upper = hdi_bounds(q, prob)
        return {
            "q_median": float(q.median()),
            "q_hdi_lower": float(q_lower),
            "q_hdi_upper": float(q_upper),
            "q_calibrated_median": float(pp["plausibility_calibrated"].median()),
            "n_restrictions": int(pp.attrs["n_restrictions"]),
            "tail_probability": float(pp.attrs["chi2_tail_of_median"]),
        }

    @property
    def n_restrictions(self) -> int:
        """Number of binding restrictions recorded on the result.

        Returns:
            The `n_restrictions` Dataset attribute, or 0 when the result
            carries no plausibility metadata (e.g. a hand-built result).
        """
        return int(self._pp().attrs.get("n_restrictions", 0))

    def pinned_values(self) -> dict[str, list[tuple[int, float]]]:
        """Resolved pinned values per variable.

        Resolves the echoed `conditions` against the forecast grid the
        same way the sampler did: a scalar broadcasts to every step, an
        array pins a leading run of steps, and `NaN` entries stay free.

        Returns:
            Dict mapping each variable name to its `(step, value)` pins
            with 1-based steps, in condition order; variables without
            pins map to an empty list.

        Raises:
            ValueError: On unknown variables, scalar-NaN conditions,
                over-length arrays, or duplicate pins.
        """
        from impulso._scenario import resolve_variable_pins

        pins = resolve_variable_pins(list(self.conditions), self.var_names, self.steps)
        resolved: dict[str, list[tuple[int, float]]] = {name: [] for name in self.var_names}
        for var_idx, step_idx, value in pins:
            resolved[self.var_names[var_idx]].append((step_idx + 1, value))
        return resolved

    def to_dataframe(self) -> pd.DataFrame:
        """Conditional-forecast posterior median as a DataFrame (passthrough to `median()`)."""
        return self.median()

    def plot(self) -> Figure:
        """Plot the conditional forecast fan chart with pinned values marked."""
        from impulso.plotting import plot_conditional_forecast

        return plot_conditional_forecast(self)


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


class CounterfactualResult(VARResultBase):
    """Historical counterfactual paths alongside the actual data.

    The posterior-predictive Dataset carries `"counterfactual"`
    (chain, draw, time, variable) and `"actual"` (time, variable) over the
    same returned window. Counterfactual draws are built from the realised
    structural shocks — edited, never re-drawn — so their spread reflects
    parameter and identification uncertainty only.

    Attributes:
        idata: InferenceData-schema container with counterfactual draws + actual path.
        var_names: Names of variables.
    """

    var_names: list[str]

    def _time_index(self) -> pd.DatetimeIndex:
        values = self._pp()["counterfactual"].coords["time"].values
        return pd.DatetimeIndex(values, name="time")

    def median(self) -> pd.DataFrame:
        """Posterior median counterfactual path.

        Returns:
            DataFrame indexed by the returned window's `DatetimeIndex`
            with one column per variable.
        """
        da = self._pp()["counterfactual"]
        med = da.median(dim=("chain", "draw")).transpose("time", "variable")
        return pd.DataFrame(med.values, index=self._time_index(), columns=self.var_names)

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """HDI for the counterfactual path.

        Args:
            prob: Probability mass for the HDI. Default 0.89.

        Returns:
            HDIResult whose `lower` / `upper` DataFrames mirror `median()`.
        """
        lower_da, upper_da = hdi_bounds(self._pp()["counterfactual"], prob)
        index = self._time_index()
        lower = pd.DataFrame(lower_da.values, index=index, columns=self.var_names)
        upper = pd.DataFrame(upper_da.values, index=index, columns=self.var_names)
        return HDIResult(lower=lower, upper=upper, prob=prob)

    def actual(self) -> pd.DataFrame:
        """The observed path over the returned window.

        Returns:
            DataFrame shaped like `median()`.
        """
        da = self._pp()["actual"]
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


class StationarityTestResult(ImpulsoBaseModel):
    """Result from a univariate stationarity or unit-root test.

    One row per tested variable. The meaning of a rejection depends on the
    test, because the two tests have opposite nulls: the Augmented
    Dickey-Fuller (ADF) test has a unit-root null, the
    Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test has a stationarity null.
    The `conclusion` column already accounts for that, so it can be read
    directly.

    Attributes:
        test: Which test produced the table, `"adf"` or `"kpss"`.
        null_hypothesis: Plain-language statement of the null being tested.
        regression: Deterministic terms included, e.g. `"c"` or `"ct"`.
        alpha: Significance level used to form the `reject` and `conclusion`
            columns.
        table: One row per variable, indexed by variable name. Columns are
            `statistic`, `pvalue`, `lags`, `crit_1pct`, `crit_5pct`,
            `crit_10pct`, `reject`, `conclusion`. KPSS tables also carry
            `crit_2_5pct` (its table covers that level too) and
            `pvalue_bounded`, `True` when the reported p-value hit the edge
            of the published lookup table and is therefore a bound rather
            than an exact figure. ADF decisions come from the p-value; KPSS
            decisions compare the statistic against the critical value for
            `alpha`, because its p-value is clipped to `[0.01, 0.10]`.
    """

    test: Literal["adf", "kpss"]
    null_hypothesis: str
    regression: str
    alpha: float
    table: pd.DataFrame = Field(repr=False)

    def summary(self) -> pd.DataFrame:
        """Return the per-variable test table.

        Returns:
            DataFrame indexed by variable name.
        """
        return self.table

    @property
    def conclusions(self) -> dict[str, str]:
        """Per-variable verdict, `"stationary"` or `"non-stationary"`."""
        return dict(self.table["conclusion"])

    @property
    def pvalues(self) -> dict[str, float]:
        """Per-variable p-value."""
        return {k: float(v) for k, v in self.table["pvalue"].items()}


class CointegrationTestResult(ImpulsoBaseModel):
    """Result from the Johansen cointegration rank test.

    The Johansen procedure walks a sequence of nulls — rank is at most 0,
    at most 1, and so on — and stops at the first null it fails to reject.
    Both the trace and maximum-eigenvalue statistics are reported; they can
    disagree, and when they do the disagreement is information, not an error.

    Only critical values are available for this test, not p-values, so
    `alpha` is restricted to the levels tabulated by MacKinnon, Haug and
    Michelis (1996): 0.10, 0.05, and 0.01.

    Attributes:
        rank_trace: Cointegration rank selected by the trace statistic.
        rank_max_eigen: Cointegration rank selected by the maximum-eigenvalue
            statistic.
        det_order: Deterministic trend order passed to the test: -1 for no
            deterministic term, 0 for a constant, 1 for a linear trend.
        k_ar_diff: Number of lagged differences in the vector error-correction
            model (VECM), i.e. `p - 1` for a VAR(p) in levels.
        alpha: Significance level whose critical-value column was used.
        n_obs: Effective number of observations after differencing and
            lagging.
        eigenvalues: Estimated eigenvalues, descending.
        table: One row per null rank `r = 0, ..., n - 1`, indexed by `r`.
            Columns are `trace_stat`, `trace_crit`, `trace_reject`,
            `maxeig_stat`, `maxeig_crit`, `maxeig_reject`.
    """

    rank_trace: int
    rank_max_eigen: int
    det_order: int
    k_ar_diff: int
    alpha: float
    n_obs: int
    eigenvalues: np.ndarray = Field(repr=False)
    table: pd.DataFrame = Field(repr=False)

    def summary(self) -> pd.DataFrame:
        """Return the sequential rank-test table.

        Returns:
            DataFrame indexed by the null rank `r`.
        """
        return self.table

    @property
    def rank(self) -> int:
        """Cointegration rank, by convention the trace-statistic answer.

        The trace test is the conventional default because it is more robust
        in small samples. Compare against `rank_max_eigen` before relying on
        it.
        """
        return self.rank_trace


class IntegrationOrderResult(ImpulsoBaseModel):
    """Result from sequential integration-order determination.

    Each variable is tested at its level, then differenced and re-tested,
    until ADF rejects a unit root or `max_order` is reached. ADF drives the
    stopping rule; KPSS is recorded alongside it as a cross-check, and the
    two are combined into a `joint_status` per level:

    - `"stationary"`: ADF rejects, KPSS does not.
    - `"unit_root"`: ADF does not reject, KPSS does.
    - `"conflicting"`: both reject.
    - `"inconclusive"`: neither rejects.

    Attributes:
        order: Integration order `d` per variable.
        alpha: Significance level used for every test.
        max_order: Highest order searched.
        regression: Deterministic terms used for the level test. Differenced
            series are always tested with a constant only.
        inconclusive: Variables whose order should not be taken at face
            value, either because they were still non-stationary at
            `max_order` or because the two tests disagreed at the level where
            the search stopped.
        table: Long table indexed by `(variable, d)`. Columns are
            `adf_stat`, `adf_pvalue`, `adf_lags`, `adf_reject`, `kpss_stat`,
            `kpss_pvalue`, `kpss_lags`, `kpss_reject`, `kpss_pvalue_bounded`,
            `joint_status`.
    """

    order: dict[str, int]
    alpha: float
    max_order: int
    regression: str
    inconclusive: list[str]
    table: pd.DataFrame = Field(repr=False)

    def summary(self) -> pd.DataFrame:
        """Return the full level-by-level test table.

        Returns:
            DataFrame indexed by `(variable, d)`.
        """
        return self.table

    @property
    def d_max(self) -> int:
        """Highest integration order across the tested variables.

        Consult `inconclusive` first. A variable still non-stationary at
        `max_order` is recorded with `order = max_order`, which is a floor,
        not a finding — so whenever `inconclusive` is non-empty `d_max` may
        understate the true maximum. A Toda-Yamamoto consumer that augments
        by `d_max` would then under-augment.
        """
        return max(self.order.values(), default=0)


class GrangerCausalityResult(ImpulsoBaseModel):
    """Posterior Granger-causal strength for one ordered cause-effect pair.

    The headline quantity is the Euclidean norm of the tested lag
    coefficients of `cause` in the `effect` equation — `‖b‖ = sqrt(sum_k
    b_k^2)`, where `b_k` multiplies `cause_{t-k}` — evaluated draw by draw,
    so the result is a posterior for a *magnitude*. That separates "the
    effect is small" from "the effect is imprecisely estimated", which a
    Wald-style quadratic form (which divides through by the posterior
    covariance) deliberately conflates. Per-lag posteriors are reported
    alongside the norm, so a single dominant lag stays visible instead of
    being buried in it.

    Granger causality is conditional predictive precedence, not
    intervention: the statement is that the past of `cause` improves the
    prediction of `effect` beyond `effect`'s own past, *within this system
    of variables*. Omitted drivers, temporal aggregation, and simultaneous
    feedback each break the step from that statement to a mechanism.

    **What `p_rope` is, and what it is not.** `p_rope` is the posterior
    probability that the strength norm falls inside the region of practical
    equivalence (ROPE) the analyst supplied: `P(‖b‖ < rope | data)`. It is
    NOT `P(no causality)`, and it is not a Bayes factor. Under Impulso's
    continuous coefficient priors the event `b = 0` has probability zero
    both before and after seeing the data, so no dataset can raise it — a
    genuine posterior probability of exact non-causality needs a prior that
    puts point mass on the null (spike-and-slab / edge inclusion), which
    Impulso does not fit. What `p_rope` quantifies is the posterior
    probability that the tested coefficients are jointly *practically*
    negligible at the magnitude you declared. Choosing `rope` is the analyst's job and there is
    no default, because there is no data-free notion of "small enough";
    that the choice is explicit and recorded is the honesty of the
    statement. With `rope=None` the result reports the distribution only
    and `p_rope` is `None`.

    **Reporting units.** With `standardize=True` (the default) the draws are
    multiplied by `sd(cause) / sd(effect)`, both sample standard deviations
    of the estimation data, so a `rope` is read in standard deviations of
    the effect per standard deviation of the cause. The factor is recorded
    in `scale`. Under lag augmentation the model is fitted in levels, and
    the sample standard deviations of integrated series are inflated by
    their trends, so standardised magnitudes are most meaningful compared
    within one fitted model rather than across models.

    **Toda-Yamamoto metadata.** Under the lag-augmented procedure the model
    is fitted with `n_lags_fitted = n_lags_tested + augmentation` lags and
    only the first `n_lags_tested` are tested; the augmented lags are never
    reported and `n_lags_tested` is never silently changed to match the
    fit. `augmentation_source` records where the augmentation came from:
    `"none"` when there is none, `"user"` when it was passed explicitly,
    and `"integration_order"` when the integration-order diagnostics were
    consulted — including the case where they returned `d_max = 0`, so the
    record shows that they were consulted. Those diagnostics are attached
    as `integration_order_result` whenever they were consulted.
    `IntegrationOrderResult.d_max` is a floor rather than a finding
    whenever its `inconclusive` list is non-empty (a variable still
    integrated at `max_order` is recorded at `max_order`), which is why
    `toda_yamamoto` refuses to run in that case rather than
    under-augmenting silently.

    Attributes:
        cause: Name of the variable whose lags are tested.
        effect: Name of the variable whose equation they are tested in.
        n_lags_tested: Number of lags of `cause` entering the strength
            norm, counting from lag 1.
        n_lags_fitted: Lag order of the model that was fitted.
        augmentation: `n_lags_fitted - n_lags_tested`, the lags fitted but
            deliberately not tested.
        augmentation_source: `"none"`, `"user"`, or `"integration_order"`.
        standardize: Whether the draws are in standardised units.
        scale: The multiplier applied to the raw coefficient draws — the
            standardisation factor, or `1.0` when `standardize` is `False`.
        rope: The region of practical equivalence, in the reporting units,
            or `None` when none was supplied.
        coef_draws: Per-lag coefficient draws in the reporting units, shape
            `(chains, draws, n_lags_tested)`, lag 1 first.
        integration_order_result: The integration-order diagnostics that
            fixed the augmentation, when they were consulted.
    """

    cause: str
    effect: str
    n_lags_tested: int
    n_lags_fitted: int
    augmentation: int
    augmentation_source: Literal["none", "integration_order", "user"]
    standardize: bool
    scale: float
    rope: float | None = Field(default=None, gt=0)
    coef_draws: np.ndarray = Field(repr=False)
    integration_order_result: IntegrationOrderResult | None = Field(default=None, repr=False)

    @property
    def norm_draws(self) -> np.ndarray:
        """Per-draw strength norm `‖b‖`, shape `(chains, draws)`."""
        return np.linalg.norm(self.coef_draws, axis=-1)

    @property
    def lag_labels(self) -> list[str]:
        """Row labels for the tested lags, `["L1", ..., "Lp"]`."""
        return [f"L{lag}" for lag in range(1, self.n_lags_tested + 1)]

    @property
    def p_rope(self) -> float | None:
        """Posterior probability that `‖b‖` falls inside the ROPE.

        `None` when no `rope` was supplied. Read the class docstring before
        reporting it: this is not the probability of no causality.
        """
        if self.rope is None:
            return None
        return float((self.norm_draws < self.rope).mean())

    def _stacked(self) -> xr.DataArray:
        """Per-lag draws with the norm appended, as a `(chain, draw, term)` array.

        Stacking lets one HDI call cover the per-lag coefficients and the
        norm together. The dims are labelled rather than positional because
        the two ArviZ lines disagree about which axes of a bare ndarray are
        the sampling axes: ArviZ 0 reduces the leading `(chain, draw)` pair
        while ArviZ 1 reduces only the trailing axis. `hdi_bounds` requires
        the labels and reduces `chain`/`draw` explicitly.
        """
        stacked = np.concatenate([self.coef_draws, self.norm_draws[..., np.newaxis]], axis=-1)
        return xr.DataArray(stacked, dims=["chain", "draw", "term"], name="strength")

    def _stacked_hdi(self, prob: float) -> tuple[np.ndarray, np.ndarray]:
        """Lower/upper HDI bounds over the stacked terms, each shape `(p + 1,)`."""
        lower, upper = hdi_bounds(self._stacked(), prob)
        return lower.values, upper.values

    def median(self) -> float:
        """Posterior median of the strength norm.

        Returns:
            Median of `‖b‖` across all draws, in the reporting units.
        """
        return float(np.median(self.norm_draws))

    def hdi(self, prob: float = 0.89) -> tuple[float, float]:
        """Highest-density interval for the strength norm.

        Args:
            prob: Probability mass for the interval. Default 0.89.

        Returns:
            Tuple of `(lower, upper)` bounds, in the reporting units.
        """
        lower, upper = self._stacked_hdi(prob)
        return float(lower[-1]), float(upper[-1])

    def summary(self, prob: float = 0.89) -> pd.DataFrame:
        """Per-lag and overall posterior summary.

        Args:
            prob: Probability mass for the HDI columns. Default 0.89.

        Returns:
            DataFrame indexed by `["L1", ..., "Lp", "norm"]` with columns
            `median`, `hdi_lower`, `hdi_upper`. When a `rope` was supplied a
            `p_rope` column is added, filled only on the `norm` row — the
            ROPE is a statement about the joint magnitude, not about any one
            lag.
        """
        lower, upper = self._stacked_hdi(prob)
        frame = pd.DataFrame(
            {
                "median": np.median(self._stacked().values, axis=(0, 1)),
                "hdi_lower": lower,
                "hdi_upper": upper,
            },
            index=pd.Index([*self.lag_labels, "norm"], name="term"),
        )
        if self.rope is not None:
            frame["p_rope"] = [np.nan] * self.n_lags_tested + [self.p_rope]
        return frame


class VolatilityResult(VARResultBase):
    """Result from univariate SV fit — posterior of conditional SD.

    Conditional SD is sigma_t = exp(h_t / 2), where h_t is the
    posterior log-volatility path.

    Attributes:
        idata: InferenceData-schema container with 'h' in posterior.
        series_name: Name of the fitted series.
        index: DatetimeIndex aligned with the fitted series.
    """

    series_name: str
    index: pd.DatetimeIndex = Field(repr=False)

    def _sigma_da(self) -> xr.DataArray:
        """exp(h/2) DataArray over chains, draws, time."""
        return np.exp(0.5 * get_group_dataset(self.idata, "posterior")["h"]).rename("sigma")

    def median(self) -> pd.DataFrame:
        """Posterior median of the conditional SD path."""
        sigma = self._sigma_da()
        med = sigma.median(dim=("chain", "draw")).values
        return pd.DataFrame({self.series_name: med}, index=self.index)

    def hdi(self, prob: float = 0.89) -> HDIResult:
        """Highest-density interval for the conditional SD path."""
        lower_da, upper_da = hdi_bounds(self._sigma_da(), prob)
        lower = pd.DataFrame({self.series_name: lower_da.values}, index=self.index)
        upper = pd.DataFrame({self.series_name: upper_da.values}, index=self.index)
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
        idata: InferenceData-schema container with 'forecast' in posterior_predictive.
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
        forecast = self._pp()["forecast"]
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
        lower_da, upper_da = hdi_bounds(self._pp()["forecast"], prob)
        axis = self._axis()
        lower = pd.DataFrame({self.series_name: lower_da.values}, index=axis)
        upper = pd.DataFrame({self.series_name: upper_da.values}, index=axis)
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
