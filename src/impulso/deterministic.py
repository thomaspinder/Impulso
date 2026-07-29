"""Deterministic regressors — calendar-anchored exogenous design matrices.

Climate series arrive with structure that is not dynamics: an annual
cycle, a warming trend, a step change when an instrument was replaced.
Left in the endogenous block, a VAR spends its coefficients re-learning
the calendar. This module builds those features as *deterministic*
regressors — total functions of a timestamp — which enter the model
through `VARData`'s exogenous block and come back with posterior
coefficients you can read.

The pieces are `Trend`, `Fourier`, `SeasonalDummies` and `BreakDummy`,
composed by `DeterministicDesign`:

```python
design = DeterministicDesign(
    terms=[Trend(degree=1, scale=120.0), Fourier(period=12, order=2)]
)
frame = pd.concat([anomalies, design.build(anomalies.index)], axis=1)
data = VARData.from_df(frame, endog=list(anomalies.columns), exog=design.column_names)
```

Everything is anchored to the calendar rather than to row position:
elapsed time is counted in integer period ordinals from the first
timestamp of the *estimation* index. That is what makes the
continuation property hold —

    design.build(index[: T + h]).iloc[T:] == design.extend(index[:T], h)

— so `design.exog_future(fitted, h)` is exactly the block the estimated
coefficients were fitted against, in the column order the posterior
expects.

Out of scope (deliberately): holiday and business calendars; interaction
terms; slope breaks; inferring cycle lengths from the data; and any
regressor that is *data* rather than calendar arithmetic — solar
forcing, ENSO indices, CO2 concentrations belong in a dataset, not here.
"""

from __future__ import annotations

import datetime as _datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from impulso._base import ImpulsoBaseModel, ImpulsoModel
from impulso.data import VARData
from impulso.protocols import DeterministicTerm

if TYPE_CHECKING:
    from impulso.fitted import FittedVAR

_TREND_NAMES = ("trend", "trend_squared", "trend_cubed")

_SEASON_ATTR = {"month": "month", "quarter": "quarter", "dayofweek": "dayofweek"}
_SEASON_LEVELS = {
    "month": tuple(range(1, 13)),
    "quarter": tuple(range(1, 5)),
    "dayofweek": tuple(range(7)),
}
_SEASON_PREFIX = {"month": "month", "quarter": "quarter", "dayofweek": "dow"}
_SEASON_CYCLE = {"month": 12.0, "quarter": 4.0, "dayofweek": 7.0}


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #


def _as_datetime_index(index: pd.DatetimeIndex, label: str = "index") -> pd.DatetimeIndex:
    """Validate that `index` is a usable, strictly increasing DatetimeIndex."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{label} must be a pandas DatetimeIndex, got {type(index).__name__}")
    if len(index) == 0:
        raise ValueError(f"{label} must not be empty")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError(f"{label} must be strictly increasing with no duplicate timestamps")
    return index


def _resolve_offset(index: pd.DatetimeIndex, freq: str | None) -> pd.offsets.BaseOffset:
    """Resolve the sampling offset from an explicit alias, the index, or inference.

    Inference is *verified*: `pd.infer_freq` returns confident answers on
    short irregular indices, so the candidate offset must regenerate the
    index exactly before it is accepted.
    """
    if freq is not None:
        return pd.tseries.frequencies.to_offset(freq)
    if index.freq is not None:
        return index.freq
    inferred = pd.infer_freq(index) if len(index) >= 3 else None
    if inferred is None:
        raise ValueError(
            "Could not determine the sampling frequency of the index: it carries no "
            "`freq` and pandas could not infer one. Pass it explicitly, e.g. "
            'DeterministicDesign(terms=[...], freq="MS").'
        )
    offset = pd.tseries.frequencies.to_offset(inferred)
    regenerated = pd.date_range(index[0], periods=len(index), freq=offset)
    if not regenerated.equals(index):
        raise ValueError(
            f"pandas inferred freq={inferred!r} for this index, but regenerating the "
            "index from that frequency does not reproduce it — the inference is a "
            "false positive on an irregular index. Pass the true sampling frequency "
            'explicitly, e.g. DeterministicDesign(terms=[...], freq="MS").'
        )
    return offset


def _period_alias(offset: pd.offsets.BaseOffset) -> str:
    """Map a sampling offset to the pandas period alias used for ordinals.

    Period ordinals are the anchor for every elapsed-time count: they are
    exact integers, unit-agnostic, and immune to the `datetime64[us]` vs
    `datetime64[ns]` resolution traps of raw integer timestamp arithmetic.
    """
    if isinstance(offset, pd.offsets.BusinessDay | pd.offsets.CustomBusinessDay):
        raise ValueError(  # noqa: TRY004 — an unusable frequency, not a wrong type
            "Business-day frequencies have no period equivalent in pandas "
            "(PeriodDtype[B] is deprecated), so deterministic terms cannot be "
            'anchored to them. Use freq="D" — calendar-day ordinals count '
            "business days correctly as long as the index itself is business-daily."
        )
    try:
        return pd.date_range(pd.Timestamp("2000-01-03"), periods=1, freq=offset).to_period().freqstr
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Frequency {offset.freqstr!r} has no pandas period equivalent, so "
            "deterministic terms cannot be anchored to the calendar at this "
            "sampling rate. Use a period-compatible frequency (MS, ME, QS, QE, "
            "YS, YE, D, W, h, ...)."
        ) from exc


def _elapsed(index: pd.DatetimeIndex, origin: pd.Timestamp, alias: str) -> np.ndarray:
    """Sampling periods elapsed between `origin` and each timestamp, as float64.

    Calendar arithmetic, not row counting: a gap in the index produces a
    jump in the returned values, which is the correct behaviour for a
    trend (time really did pass) and for a harmonic (the phase really did
    advance).

    The unit is one *sampling period*, not one period-ordinal tick. Those
    differ whenever the frequency carries a multiplier: pandas stores
    `15D` ordinals in days and `2h` ordinals in hours, so the raw
    difference would count 15 (or 2) per observation. Dividing by the
    multiplier is exact — the origin is subtracted first, so an on-grid
    index yields exact integers — and it is what makes `Fourier.period`
    mean what its docstring says at every sampling rate.
    """
    anchor = pd.Period(origin, freq=alias)
    ticks = (index.to_period(alias).asi8 - anchor.ordinal).astype(np.float64)
    return ticks / anchor.freq.n


def _extend_index(index: pd.DatetimeIndex, offset: pd.offsets.BaseOffset, steps: int) -> pd.DatetimeIndex:
    """The `steps` timestamps following the last entry of `index`.

    `pd.date_range` rolls a start that is not on the offset's anchor
    forward to the next valid date, so the walk's first entry is *already*
    a future period whenever the sample ends off-anchor — an irregular
    index with an explicit `freq`, which this module supports. Dropping it
    unconditionally would skip a period (a sample ending 2000-03-15 under
    `MS` would forecast from May, silently losing April). Selecting the
    entries strictly after the last observation is correct either way.
    """
    walk = pd.date_range(index[-1], freq=offset, periods=steps + 1)
    return walk[walk > index[-1]][:steps]


def _format_period(period: float) -> str:
    """Render a cycle length for a column name: 12.0 -> "12", 365.25 -> "365.25"."""
    return str(int(period)) if float(period).is_integer() else str(float(period))


def _format_break_timestamp(timestamp: pd.Timestamp) -> str:
    """Render a break timestamp for a column name or an error message.

    Midnight — the overwhelmingly common case, and every timestamp on a
    daily-or-coarser index — renders as a bare ISO date, `2000-02-01`.
    Anything with an intraday component keeps it, ISO-8601 extended:
    `2000-02-01T12:00`, with seconds and a fractional part appended only
    when they are non-zero.

    Two properties matter. The rendering is *injective* — distinct
    timestamps never collide, so two intraday breaks on the same day
    stay distinct columns rather than tripping the duplicate-name check
    — and it round-trips, `pd.Timestamp("2000-02-01T12:00")` recovering
    the break date from its own column name.
    """
    date = timestamp.strftime("%Y-%m-%d")
    fraction = timestamp.microsecond * 1000 + timestamp.nanosecond
    if not (timestamp.hour or timestamp.minute or timestamp.second or fraction):
        return date
    rendered = f"{date}T{timestamp.hour:02d}:{timestamp.minute:02d}"
    if timestamp.second or fraction:
        rendered += f":{timestamp.second:02d}"
    if fraction:
        rendered += f".{fraction:09d}".rstrip("0")
    return rendered


# --------------------------------------------------------------------------- #
# Terms
# --------------------------------------------------------------------------- #


class Trend(ImpulsoModel):
    """Polynomial time trend anchored to the start of the estimation sample.

    Column `trend` is the number of periods elapsed since the first
    observation, divided by `scale`; higher degrees are integer powers of
    that same column.

    The prior on the exogenous coefficients adapts to each regressor's
    sample spread (see `VAR.exog_prior_scale`), so an unscaled trend is
    no longer fought by the prior. `scale` still matters for two other
    reasons: the coefficient's units, and the sampler's geometry. Divide
    by a fixed, interpretable constant — periods per decade, say — so
    the coefficient reads as "change per decade", and so the design
    column stays O(1) rather than reaching 539 by the end of a 540-month
    sample.

    Warning:
        `scale` must not depend on the sample length. A `T`-dependent
        scale (`scale=len(index)`) breaks the continuation property: the
        design you extend with would no longer be the design you fitted.

    Attributes:
        degree: Highest power of elapsed time, 1 to 3.
        scale: Divisor applied to elapsed time before exponentiation.
    """

    degree: int = Field(default=1, ge=1, le=3)
    scale: float = Field(default=1.0, gt=0)

    @property
    def column_names(self) -> list[str]:
        """Column names contributed by this term."""
        return list(_TREND_NAMES[: self.degree])

    def build(self, index: pd.DatetimeIndex, origin: pd.Timestamp, alias: str) -> np.ndarray:
        """Evaluate the trend powers on `index`, anchored at `origin`."""
        t = _elapsed(index, origin, alias) / self.scale
        return np.column_stack([t**k for k in range(1, self.degree + 1)])


class Fourier(ImpulsoModel):
    """Harmonic pair(s) representing a smooth cycle of known length.

    Order `k` contributes `sin(2πk·t/period)` and `cos(2πk·t/period)`,
    where `t` is periods elapsed since the start of the estimation
    sample. Two harmonics on a monthly index (`period=12, order=2`) buy a
    seasonal shape at four coefficients per variable instead of the
    eleven that month dummies cost.

    The cycle length is always explicit — nothing here infers a period
    from the data.

    Note:
        On a daily index, `period=365.25` is an approximation: elapsed
        time is counted in whole days, so the harmonic drifts against the
        calendar within a leap cycle. It is fine for multi-year samples
        and exact for monthly, quarterly and annual sampling, where a
        period ordinal *is* the calendar unit.

    Attributes:
        period: Cycle length in sampling periods (12 for an annual cycle
            on monthly data, 4 on quarterly data).
        order: Number of harmonic pairs. Must satisfy `2 * order <=
            period` — beyond the Nyquist limit the extra harmonics are
            not identified from the sampled points.
    """

    period: float = Field(..., gt=1)
    order: int = Field(..., ge=1)

    @model_validator(mode="after")
    def _validate_nyquist(self) -> Fourier:
        if 2 * self.order > self.period:
            raise ValueError(
                f"Fourier(period={self.period}, order={self.order}) exceeds the Nyquist "
                f"limit: at most {int(self.period // 2)} harmonic pairs are identified "
                f"from a cycle sampled {self.period} times."
            )
        return self

    @property
    def column_names(self) -> list[str]:
        """Column names contributed by this term, sine before cosine per order."""
        label = _format_period(self.period)
        names: list[str] = []
        for k in range(1, self.order + 1):
            names.append(f"sin({k},{label})")
            names.append(f"cos({k},{label})")
        return names

    def build(self, index: pd.DatetimeIndex, origin: pd.Timestamp, alias: str) -> np.ndarray:
        """Evaluate the harmonics on `index`, with phase anchored at `origin`."""
        t = _elapsed(index, origin, alias)
        columns: list[np.ndarray] = []
        for k in range(1, self.order + 1):
            angle = 2.0 * np.pi * k * t / self.period
            columns.append(np.sin(angle))
            columns.append(np.cos(angle))
        return np.column_stack(columns)


class SeasonalDummies(ImpulsoModel):
    """Indicator columns for a calendar season, one level dropped.

    Unlike trends and harmonics this term reads the calendar attribute
    directly, so it needs no origin and is invariant to where the sample
    starts.

    One level must be dropped: `VAR` and `ConjugateVAR` both fit an
    unconditional intercept, and a full set of indicators sums to it.

    Attributes:
        season: `"month"` (levels 1-12), `"quarter"` (1-4) or
            `"dayofweek"` (0-6, Monday = 0).
        drop_first: Drop one level to keep the design full rank against
            the intercept. Leave `True` unless the design matrix is bound
            for an estimator that fits no intercept.
        reference: The level to drop. Defaults to the first level of the
            season. Only meaningful when `drop_first` is `True`.
    """

    season: Literal["month", "quarter", "dayofweek"]
    drop_first: bool = True
    reference: int | None = None

    @model_validator(mode="after")
    def _validate_reference(self) -> SeasonalDummies:
        if self.reference is None:
            return self
        if not self.drop_first:
            raise ValueError("reference is only meaningful with drop_first=True — nothing is dropped otherwise.")
        levels = _SEASON_LEVELS[self.season]
        if self.reference not in levels:
            raise ValueError(
                f"reference={self.reference} is not a valid {self.season} level; expected one of {list(levels)}."
            )
        return self

    @property
    def _kept_levels(self) -> tuple[int, ...]:
        levels = _SEASON_LEVELS[self.season]
        if not self.drop_first:
            return levels
        dropped = levels[0] if self.reference is None else self.reference
        return tuple(level for level in levels if level != dropped)

    @property
    def column_names(self) -> list[str]:
        """Column names contributed by this term, in ascending level order."""
        prefix = _SEASON_PREFIX[self.season]
        return [f"{prefix}_{level}" for level in self._kept_levels]

    def build(self, index: pd.DatetimeIndex, origin: pd.Timestamp, alias: str) -> np.ndarray:
        """Evaluate the indicators on `index`. `origin` and `alias` are unused."""
        del origin, alias
        values = np.asarray(getattr(index, _SEASON_ATTR[self.season]))
        return np.column_stack([(values == level).astype(np.float64) for level in self._kept_levels])


class BreakDummy(ImpulsoBaseModel):
    """Indicator for a known, dated discontinuity.

    A `"level"` break is 1 from `date` onward — an instrument change, a
    station move, a regime shift. Its coefficient is the permanent offset,
    and `FittedVAR.dynamic_multiplier(cumulative=True)` propagates it
    through the lag dynamics into the full adjustment path.

    A `"pulse"` break is 1 on `date` alone — a one-off event whose
    influence you want held out of the dynamics.

    The column is named `{kind}_{date}`, with the date rendered as
    `YYYY-MM-DD` at midnight and `YYYY-MM-DDTHH:MM` (seconds and
    fractional seconds appended when non-zero) at any other time of day.
    Intraday sampling therefore keeps two breaks on the same day apart,
    and a `"pulse"` that missed the index says so with the time
    included — the dropped time being the usual reason it missed.

    Attributes:
        date: Timestamp of the break. Strings, `datetime` objects and
            `numpy.datetime64` are coerced.
        kind: `"level"` (step, the default) or `"pulse"` (one period).
    """

    date: pd.Timestamp
    kind: Literal["level", "pulse"] = "level"

    @field_validator("date", mode="before")
    @classmethod
    def _coerce_timestamps(cls, value: object) -> object:
        if isinstance(value, pd.Timestamp):
            return value
        if isinstance(value, str | _datetime.date | np.datetime64):
            return pd.Timestamp(value)
        return value

    @property
    def column_names(self) -> list[str]:
        """The single column name contributed by this term."""
        return [f"{self.kind}_{_format_break_timestamp(self.date)}"]

    def build(self, index: pd.DatetimeIndex, origin: pd.Timestamp, alias: str) -> np.ndarray:
        """Evaluate the indicator on `index`. `origin` and `alias` are unused."""
        del origin, alias
        hit = index >= self.date if self.kind == "level" else index == self.date
        return np.asarray(hit, dtype=np.float64).reshape(-1, 1)

    def _check_in_sample(self, index: pd.DatetimeIndex) -> None:
        """Raise if the break is not identified from the estimation sample."""
        stamp = _format_break_timestamp(self.date)
        if self.kind == "pulse":
            if self.date not in index:
                position = int(index.searchsorted(self.date))
                before = index[position - 1] if position > 0 else None
                after = index[position] if position < len(index) else None
                raise ValueError(
                    f"BreakDummy(kind='pulse', date={stamp}) does not fall on "
                    f"the sampled index, so the column is zero everywhere and its "
                    f"coefficient is not identified. Nearest sampled timestamps: "
                    f"{_format_break_timestamp(before) if before is not None else None} (before) and "
                    f"{_format_break_timestamp(after) if after is not None else None} (after)."
                )
            return
        if self.date <= index[0]:
            raise ValueError(
                f"BreakDummy(kind='level', date={stamp}) is at or before the "
                f"start of the sample ({_format_break_timestamp(index[0])}), so the column "
                f"is 1 everywhere and is collinear with the intercept every estimator fits."
            )
        if self.date > index[-1]:
            raise ValueError(
                f"BreakDummy(kind='level', date={stamp}) is after the end of "
                f"the sample ({_format_break_timestamp(index[-1])}), so the shift never "
                f"occurs in-sample and its coefficient is not identified."
            )


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #


class DeterministicDesign(ImpulsoBaseModel):
    """A composed set of deterministic terms, built and extended as one block.

    The design owns the calendar: it resolves a single `(origin, alias)`
    pair from the estimation index and passes it to every term, in-sample
    and out. That is what makes the continuation property hold —

        design.build(index[: T + h]).iloc[T:] == design.extend(index[:T], h)

    — and it is why `exog_future` can hand `FittedVAR.forecast` a block
    that matches, column for column, what the model was fitted against.

    Attributes:
        terms: The `DeterministicTerm` instances to concatenate, in
            column order. At least one is required.
        freq: Explicit pandas frequency alias for the sampling rate (e.g.
            `"MS"`, `"QS"`, `"D"`). Optional: the design falls back to the
            index's own `freq`, then to *verified* inference. Pass it
            whenever the index has gaps — calendar anchoring handles gaps
            correctly, but pandas cannot infer a frequency through them.
    """

    terms: tuple[DeterministicTerm, ...]
    freq: str | None = None

    @model_validator(mode="after")
    def _validate_terms(self) -> DeterministicDesign:
        if not self.terms:
            raise ValueError("DeterministicDesign requires at least one term.")
        seen: dict[str, int] = {}
        for position, term in enumerate(self.terms):
            for name in term.column_names:
                if name in seen:
                    raise ValueError(
                        f"Duplicate column name {name!r}: terms {seen[name]} "
                        f"({type(self.terms[seen[name]]).__name__}) and {position} "
                        f"({type(term).__name__}) both emit it. Deterministic column "
                        f"names are the contract that keeps forecast blocks aligned, so "
                        f"they must be unique."
                    )
                seen[name] = position
        return self

    @property
    def column_names(self) -> list[str]:
        """Every column the design emits, in build order."""
        return [name for term in self.terms for name in term.column_names]

    # -- calendar resolution ------------------------------------------------ #

    def _resolve_calendar(self, index: pd.DatetimeIndex) -> tuple[pd.Timestamp, str, pd.offsets.BaseOffset]:
        offset = _resolve_offset(index, self.freq)
        return index[0], _period_alias(offset), offset

    def _assemble(self, index: pd.DatetimeIndex, origin: pd.Timestamp, alias: str) -> pd.DataFrame:
        blocks: list[np.ndarray] = []
        for term in self.terms:
            block = np.asarray(term.build(index, origin, alias), dtype=np.float64)
            expected = (len(index), len(term.column_names))
            if block.shape != expected:
                raise ValueError(
                    f"{type(term).__name__}.build returned shape {block.shape}, but its column_names imply {expected}."
                )
            blocks.append(block)
        return pd.DataFrame(np.hstack(blocks), index=index, columns=self.column_names)

    # -- public surface ----------------------------------------------------- #

    def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Construct the in-sample design matrix.

        The result is checked for rank deficiency against an intercept
        column, because every Impulso estimator fits one and a deficient
        exogenous block is not identified.

        Args:
            index: The estimation index — strictly increasing, no
                duplicates. Its first timestamp becomes the origin for
                every elapsed-time count.

        Returns:
            A `float64` DataFrame indexed by `index`, with columns
            `self.column_names` in that order and no missing values.

        Raises:
            ValueError: If the frequency cannot be resolved, a break
                dummy is not identified in-sample, there are fewer
                observations than columns, or the design is collinear.
        """
        index = _as_datetime_index(index)
        origin, alias, _ = self._resolve_calendar(index)
        for term in self.terms:
            if isinstance(term, BreakDummy):
                term._check_in_sample(index)
        frame = self._assemble(index, origin, alias)
        self._check_rank(frame)
        return frame

    def extend(
        self,
        index: pd.DatetimeIndex,
        steps: int,
        future_index: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """Construct the design matrix for `steps` periods past `index`.

        Takes the *estimation* index, not the future one, so the origin
        and frequency resolve exactly as they did in `build` (the
        statsmodels `out_of_sample` contract). No rank check is applied:
        short horizons are legitimately rank-deficient — a 3-step block
        cannot span twelve month dummies — and nothing is being estimated
        from these rows.

        Args:
            index: The estimation index the design was built on.
            steps: Number of future periods, at least 1.
            future_index: Optional explicit future timestamps, overriding
                the frequency walk. Must have length `steps`.

        Returns:
            A `float64` DataFrame of `steps` rows with the same columns,
            in the same order, as `build`.
        """
        index = _as_datetime_index(index)
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        origin, alias, offset = self._resolve_calendar(index)
        if future_index is None:
            resolved = _extend_index(index, offset, steps)
        else:
            resolved = _as_datetime_index(future_index, label="future_index")
            if len(resolved) != steps:
                raise ValueError(f"future_index has length {len(resolved)}, but steps={steps}.")
        return self._assemble(resolved, origin, alias)

    def future_index(self, index: pd.DatetimeIndex, steps: int) -> pd.DatetimeIndex:
        """The timestamps `extend` would use for `steps` periods past `index`.

        Args:
            index: The estimation index.
            steps: Number of future periods, at least 1.

        Returns:
            A `DatetimeIndex` of length `steps`.
        """
        index = _as_datetime_index(index)
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        _, _, offset = self._resolve_calendar(index)
        return _extend_index(index, offset, steps)

    def exog_future(
        self,
        fitted: FittedVAR | VARData,
        steps: int,
        future_index: pd.DatetimeIndex | None = None,
    ) -> np.ndarray:
        """The future exogenous block for `FittedVAR.forecast`, column-aligned.

        `forecast`, `conditional_forecast` and `structural_scenario` all
        index `exog_future` positionally, so a permuted column order is a
        silently wrong forecast rather than an error. This method removes
        that hazard: it extends the design, then reorders the columns by
        *name* to match `exog_names` on the data the model was fitted
        with.

        Args:
            fitted: A `FittedVAR`, or the `VARData` it was fitted on.
            steps: Forecast horizon, at least 1.
            future_index: Optional explicit future timestamps, of length
                `steps`.

        Returns:
            A `(steps, k)` float64 array whose columns are ordered as
            `exog_names`.

        Raises:
            ValueError: If the data carries no exogenous block, or if the
                design's columns and `exog_names` are not the same set.
        """
        data = fitted if isinstance(fitted, VARData) else fitted.data
        if data.exog_names is None:
            raise ValueError(
                "This model was fitted without exogenous regressors, so there is no "
                "future exogenous block to build. Refit with the design included in "
                "VARData before forecasting with it."
            )
        frame = self.extend(data.index, steps, future_index=future_index)
        missing = [name for name in data.exog_names if name not in frame.columns]
        extra = [name for name in frame.columns if name not in data.exog_names]
        if missing or extra:
            raise ValueError(
                f"The design does not match the fitted exogenous block. Fitted "
                f"exog_names: {list(data.exog_names)}; design columns: "
                f"{list(frame.columns)}. Missing from the design: {missing}; not fitted: "
                f"{extra}. Build the forecast block with the same design the model was "
                f"fitted with."
            )
        return np.ascontiguousarray(frame[list(data.exog_names)].to_numpy(dtype=np.float64))

    # -- rank diagnostics --------------------------------------------------- #

    def _check_rank(self, frame: pd.DataFrame) -> None:
        design = frame.to_numpy(dtype=np.float64)
        augmented = np.column_stack([np.ones(len(frame)), design])
        n_rows, n_cols = augmented.shape
        if n_rows < n_cols:
            raise ValueError(
                f"Too few observations for this design: {len(frame)} rows against "
                f"{design.shape[1]} deterministic columns plus the intercept every "
                f"estimator fits. Shorten the design or lengthen the sample."
            )
        if int(np.linalg.matrix_rank(augmented)) == n_cols:
            return
        hints = self._collinearity_hints()
        detail = ("\n  - " + "\n  - ".join(hints)) if hints else ""
        raise ValueError(
            "This deterministic design has collinear columns once the intercept every "
            f"estimator fits is included ({n_cols} columns, rank "
            f"{int(np.linalg.matrix_rank(augmented))}), so its coefficients are not "
            f"identified.{detail}"
        )

    def _collinearity_hints(self) -> list[str]:
        hints: list[str] = []
        seasons = [t for t in self.terms if isinstance(t, SeasonalDummies)]
        fouriers = [t for t in self.terms if isinstance(t, Fourier)]
        for term in seasons:
            if not term.drop_first:
                hints.append(
                    f"SeasonalDummies(season={term.season!r}, drop_first=False) emits every "
                    f"level, and they sum to the intercept column. Set drop_first=True."
                )
        for term in fouriers:
            if 2 * term.order == term.period:
                hints.append(
                    f"Fourier(period={term.period}, order={term.order}) sits exactly at the "
                    f"Nyquist limit: sin(2*pi*{term.order}*t/{term.period}) is identically "
                    f"zero at integer t. Drop the last harmonic pair (order="
                    f"{term.order - 1})."
                )
        for season in seasons:
            for fourier in fouriers:
                if fourier.period == _SEASON_CYCLE[season.season]:
                    hints.append(
                        f"SeasonalDummies(season={season.season!r}) and Fourier(period="
                        f"{fourier.period}) describe the same cycle; harmonics of a cycle "
                        f"sampled that many times are linear combinations of its level "
                        f"indicators. Keep one or the other."
                    )
        return hints
