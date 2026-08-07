"""Stationarity, unit-root, and cointegration pretests.

These are classical frequentist diagnostics, run before a VAR is specified,
to answer a modelling question: should the model be fitted in levels or in
differences, and is there a long-run relationship worth preserving?

They depend on `statsmodels`, which Impulso does not require by default.
Install the extra to use anything in this module:

```
pip install "impulso[diagnostics]"
```

Nothing here decides anything on the user's behalf. Unit-root tests have low
power against persistent alternatives and are sensitive to deterministic
terms and structural breaks, so the results are reported in full — including
the cases where the Augmented Dickey-Fuller (ADF) and
Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests disagree — and the modelling
call is left to the analyst.
"""

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from impulso._optional import require
from impulso.data import VARData
from impulso.results import (
    CointegrationTestResult,
    IntegrationOrderResult,
    StationarityTestResult,
)

_ADF_REGRESSIONS = ("n", "c", "ct")
_KPSS_REGRESSIONS = ("c", "ct")
_LAG_SELECTIONS = ("aic", "bic", "t-stat")
# MacKinnon-Haug-Michelis (1996) tabulates the Johansen critical values at
# these three levels only, and the test has no p-value to interpolate from.
_JOHANSEN_CRIT_COLUMN = {0.10: 0, 0.05: 1, 0.01: 2}
# Kwiatkowski et al. (1992), Table 1 tabulates KPSS at these four levels.
# Decisions compare the statistic against the critical value directly: the
# reported p-value is interpolated and clipped to [0.01, 0.10], so a
# `pvalue < alpha` rule silently fails outside that range.
_KPSS_CRIT_KEY = {0.10: "10%", 0.05: "5%", 0.025: "2.5%", 0.01: "1%"}

_ADF_NULL = "the series has a unit root (non-stationary)"
_KPSS_NULL = "the series is stationary around a constant or trend"


def _adfuller():
    """Return `statsmodels.tsa.stattools.adfuller`, or raise an install hint."""
    require("statsmodels", extra="diagnostics")
    from statsmodels.tsa.stattools import adfuller

    return adfuller


def _kpss():
    """Return `statsmodels`' KPSS entry point and its interpolation warning."""
    require("statsmodels", extra="diagnostics")
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    from statsmodels.tsa.stattools import kpss

    return kpss, InterpolationWarning


def _coint_johansen():
    """Return `statsmodels`' Johansen test, or raise an install hint."""
    require("statsmodels", extra="diagnostics")
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    return coint_johansen


def _to_frame(data: VARData | pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Normalise the accepted input types to a DataFrame of series to test.

    A `VARData` contributes its endogenous block only; exogenous regressors
    are not integration-order candidates for the VAR being specified.

    Args:
        data: VARData, DataFrame, or Series.

    Returns:
        DataFrame with one column per series.

    Raises:
        TypeError: If `data` is none of the accepted types.
    """
    if isinstance(data, VARData):
        return pd.DataFrame(data.endog, columns=data.endog_names, index=data.index)
    if isinstance(data, pd.Series):
        return data.to_frame(name=data.name if data.name is not None else "series")
    if isinstance(data, pd.DataFrame):
        return data
    raise TypeError(f"data must be VARData, DataFrame, or Series, got {type(data).__name__}")


def _select(frame: pd.DataFrame, variables: Sequence[str] | None) -> pd.DataFrame:
    """Subset `frame` to `variables`, preserving the requested order."""
    if variables is None:
        return frame
    missing = [v for v in variables if v not in frame.columns]
    if missing:
        raise ValueError(f"variables not found in data: {missing}")
    return frame[list(variables)]


def _check_finite(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject non-finite values, naming the offending columns.

    Runs *after* subsetting, so a NaN in a column the caller excluded is not
    the caller's problem.
    """
    finite = np.isfinite(frame.to_numpy(dtype=np.float64)).all(axis=0)
    if not finite.all():
        bad = [str(c) for c in frame.columns[~finite]]
        raise ValueError(f"columns contain NaN or Inf and cannot be tested: {bad}")
    return frame


def _prepare(data: VARData | pd.DataFrame | pd.Series, variables: Sequence[str] | None = None) -> pd.DataFrame:
    """Normalise, subset, then validate — in that order."""
    return _check_finite(_select(_to_frame(data), variables))


def _check_alpha(alpha: float) -> None:
    """Reject significance levels outside the open unit interval."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")


def _check_kpss_alpha(alpha: float) -> None:
    """Restrict KPSS to the levels its published table covers."""
    if alpha not in _KPSS_CRIT_KEY:
        raise ValueError(
            f"alpha must be one of {sorted(_KPSS_CRIT_KEY)} "
            f"(KPSS critical values are tabulated only at these levels), got {alpha}"
        )


def _adf_single(
    x: np.ndarray,
    *,
    regression: str,
    max_lags: int | None,
    lag_selection: str | None,
    alpha: float,
) -> dict:
    """Run ADF on one series and return a flat row of results."""
    adfuller = _adfuller()
    stat, pvalue, used_lag, _nobs, crit, *_ = adfuller(
        x,
        maxlag=max_lags,
        regression=regression,
        autolag=lag_selection,
    )
    reject = bool(pvalue < alpha)
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "lags": int(used_lag),
        "crit_1pct": float(crit["1%"]),
        "crit_5pct": float(crit["5%"]),
        "crit_10pct": float(crit["10%"]),
        "reject": reject,
        # ADF's null is a unit root, so rejecting it argues for stationarity.
        "conclusion": "stationary" if reject else "non-stationary",
    }


def _kpss_single(
    x: np.ndarray,
    *,
    regression: str,
    nlags: int | str,
    alpha: float,
) -> dict:
    """Run KPSS on one series and return a flat row of results.

    The decision compares the statistic against the critical value for
    `alpha`, not the p-value against `alpha`. statsmodels clips the reported
    p-value to `[0.01, 0.10]`, so a p-value rule can never reject at
    alpha = 0.01 and always rejects above 0.10. Comparing against the
    critical value is the published test, and agrees with the p-value rule
    everywhere inside that range.

    The clip is detected by catching statsmodels' `InterpolationWarning`, which
    is absorbed into `pvalue_bounded`. Every other warning raised inside the
    call is re-emitted unchanged, so nothing else is hidden from the caller.
    """
    kpss, interpolation_warning = _kpss()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stat, pvalue, used_lag, crit = kpss(x, regression=regression, nlags=nlags)
    bounded = any(issubclass(w.category, interpolation_warning) for w in caught)
    # The interpolation warning is the only one we absorb — it is reported as
    # `pvalue_bounded` instead. Anything else statsmodels raised is the
    # caller's business, so re-emit it outside the recorder with its original
    # category, message, and provenance (the statsmodels file and line that
    # raised it), which keeps module-scoped warning filters working.
    for w in caught:
        if not issubclass(w.category, interpolation_warning):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)
    reject = bool(stat > crit[_KPSS_CRIT_KEY[alpha]])
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "lags": int(used_lag),
        "crit_1pct": float(crit["1%"]),
        "crit_2_5pct": float(crit["2.5%"]),
        "crit_5pct": float(crit["5%"]),
        "crit_10pct": float(crit["10%"]),
        "pvalue_bounded": bounded,
        "reject": reject,
        # KPSS's null is stationarity, so rejecting it argues the other way.
        "conclusion": "non-stationary" if reject else "stationary",
    }


def adf_test(
    data: VARData | pd.DataFrame | pd.Series,
    variables: Sequence[str] | None = None,
    *,
    regression: Literal["n", "c", "ct"] = "c",
    max_lags: int | None = None,
    lag_selection: Literal["aic", "bic", "t-stat"] | None = "aic",
    alpha: float = 0.05,
) -> StationarityTestResult:
    """Augmented Dickey-Fuller (ADF) unit-root test, one series at a time.

    The null hypothesis is that the series has a unit root. A small p-value
    therefore argues *against* a unit root, i.e. for stationarity — the
    opposite orientation to `kpss_test`. Running both is the usual practice,
    because ADF has low power against near-unit-root alternatives.

    Args:
        data: VARData (endogenous block only), DataFrame, or Series.
        variables: Subset of column names to test. Defaults to all.
        regression: Deterministic terms in the test regression. `"n"` for
            none, `"c"` for a constant, `"ct"` for a constant and linear
            trend. Use `"ct"` when the series has a visible trend, otherwise
            the test confuses trend with a unit root.
        max_lags: Maximum lag length considered. Defaults to the statsmodels
            rule, `12 * (T / 100) ** 0.25`.
        lag_selection: Criterion used to pick the lag length up to
            `max_lags`. Pass `None` to use `max_lags` itself.
        alpha: Significance level for the reported conclusion.

    Returns:
        StationarityTestResult with one row per variable.

    Raises:
        ValueError: If `regression`, `lag_selection`, or `alpha` is invalid.
    """
    if regression not in _ADF_REGRESSIONS:
        raise ValueError(f"regression must be one of {_ADF_REGRESSIONS}, got {regression!r}")
    if lag_selection is not None and lag_selection not in _LAG_SELECTIONS:
        raise ValueError(f"lag_selection must be one of {_LAG_SELECTIONS} or None, got {lag_selection!r}")
    _check_alpha(alpha)

    frame = _prepare(data, variables)
    rows = {
        str(name): _adf_single(
            frame[name].to_numpy(dtype=np.float64),
            regression=regression,
            max_lags=max_lags,
            lag_selection=lag_selection,
            alpha=alpha,
        )
        for name in frame.columns
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "variable"
    return StationarityTestResult(
        test="adf",
        null_hypothesis=_ADF_NULL,
        regression=regression,
        alpha=alpha,
        table=table,
    )


def kpss_test(
    data: VARData | pd.DataFrame | pd.Series,
    variables: Sequence[str] | None = None,
    *,
    regression: Literal["c", "ct"] = "c",
    nlags: int | Literal["auto"] = "auto",
    alpha: float = 0.05,
) -> StationarityTestResult:
    """Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity test.

    Runs one series at a time. The null hypothesis is that the series is
    stationary, so rejecting argues *for* a unit root — the reverse of
    `adf_test`.

    The reject/no-reject decision compares the statistic against the critical
    value for `alpha`, taken from Table 1 of Kwiatkowski et al. (1992). The
    p-value is reported too, but is interpolated from that same table and
    clipped to `[0.01, 0.10]`; when the clip binds, `pvalue_bounded` is
    `True` and the figure should be read as a bound. Because the p-value is
    clipped, `alpha` is restricted to the four levels the table covers —
    comparing a clipped p-value against, say, 0.01 could never reject.

    Args:
        data: VARData (endogenous block only), DataFrame, or Series.
        variables: Subset of column names to test. Defaults to all.
        regression: `"c"` to test stationarity around a constant, `"ct"` to
            test trend stationarity.
        nlags: Newey-West bandwidth for the long-run variance, or `"auto"`
            for the data-dependent rule.
        alpha: Significance level. Restricted to 0.10, 0.05, 0.025, or 0.01,
            the levels for which critical values are tabulated.

    Returns:
        StationarityTestResult with one row per variable.

    Raises:
        ValueError: If `regression` is invalid, or `alpha` is not a tabulated
            level.
    """
    if regression not in _KPSS_REGRESSIONS:
        raise ValueError(f"regression must be one of {_KPSS_REGRESSIONS}, got {regression!r}")
    _check_kpss_alpha(alpha)

    frame = _prepare(data, variables)
    rows = {
        str(name): _kpss_single(
            frame[name].to_numpy(dtype=np.float64),
            regression=regression,
            nlags=nlags,
            alpha=alpha,
        )
        for name in frame.columns
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "variable"
    return StationarityTestResult(
        test="kpss",
        null_hypothesis=_KPSS_NULL,
        regression=regression,
        alpha=alpha,
        table=table,
    )


def _sequential_rank(stats: np.ndarray, crits: np.ndarray) -> tuple[int, np.ndarray]:
    """Walk the Johansen null sequence and return the selected rank.

    The nulls are "rank is at most r" for r = 0, 1, ... Each rejection moves
    to the next null; the first non-rejection fixes the rank. Rejecting every
    null means the system is full rank, i.e. stationary in levels.

    Args:
        stats: Test statistics ordered by r.
        crits: Matching critical values.

    Returns:
        Tuple of the selected rank and the per-r rejection flags.
    """
    rejects = stats > crits
    for r, rejected in enumerate(rejects):
        if not rejected:
            return r, rejects
    return len(rejects), rejects


def johansen_test(
    data: VARData | pd.DataFrame,
    *,
    det_order: Literal[-1, 0, 1] = 0,
    k_ar_diff: int = 1,
    alpha: float = 0.05,
) -> CointegrationTestResult:
    """Johansen cointegration rank test.

    Reports both the trace and maximum-eigenvalue sequential tests. A rank of
    0 means no cointegration; a rank equal to the number of series means the
    system is stationary in levels; anything in between means the levels
    share common stochastic trends. Any rank of 1 or more therefore carries a
    modelling consequence rather than just a verdict: differencing every
    series discards the long-run relationship those series share, throwing
    away the cointegrating restrictions along with the unit roots.

    Impulso does not implement a vector error-correction model, and that is a
    deliberate scope boundary rather than an omission. The recommended
    response to a non-zero rank is a VAR in levels — the Sims-Stock-Watson
    stance, under which a levels VAR stays consistent when the series are
    cointegrated and avoids imposing a rank the test only estimates — and the
    Minnesota prior already shrinks toward random walks, so a levels fit is
    not fighting the unit roots it contains.

    The test is conditioned on a lag order. `k_ar_diff` counts lagged
    *differences*, so it is `p - 1` for a VAR(p) in levels — pick `p` with
    `select_lag_order` first, then subtract one.

    Critical values are MacKinnon-Haug-Michelis (1996); there are no
    p-values, so `alpha` is restricted to the tabulated levels.

    Args:
        data: VARData (endogenous block only) or DataFrame, two or more
            columns.
        det_order: Deterministic term. `-1` for none, `0` for a constant,
            `1` for a linear trend.
        k_ar_diff: Number of lagged differences in the vector error-correction
            model (VECM), `p - 1`.
        alpha: Significance level. Restricted to 0.10, 0.05, or 0.01, the
            levels for which critical values are tabulated.

    Returns:
        CointegrationTestResult with both rank decisions and the full table.

    Raises:
        ValueError: If `alpha` is not a tabulated level, if `det_order` is
            not -1, 0, or 1, if `k_ar_diff` is negative, or if fewer than two
            series are supplied.
    """
    if alpha not in _JOHANSEN_CRIT_COLUMN:
        raise ValueError(
            f"alpha must be one of {sorted(_JOHANSEN_CRIT_COLUMN)} (critical values are tabulated only at these levels), got {alpha}"
        )
    if det_order not in (-1, 0, 1):
        raise ValueError(f"det_order must be -1, 0, or 1, got {det_order}")
    if k_ar_diff < 0:
        raise ValueError(f"k_ar_diff must be non-negative, got {k_ar_diff}")

    frame = _prepare(data)
    if frame.shape[1] < 2:
        raise ValueError(f"johansen_test needs at least two series, got {frame.shape[1]}")

    coint_johansen = _coint_johansen()
    res = coint_johansen(frame.to_numpy(dtype=np.float64), det_order, k_ar_diff)

    column = _JOHANSEN_CRIT_COLUMN[alpha]
    trace_stat = np.asarray(res.lr1, dtype=np.float64)
    trace_crit = np.asarray(res.cvt, dtype=np.float64)[:, column]
    maxeig_stat = np.asarray(res.lr2, dtype=np.float64)
    maxeig_crit = np.asarray(res.cvm, dtype=np.float64)[:, column]

    rank_trace, trace_reject = _sequential_rank(trace_stat, trace_crit)
    rank_max_eigen, maxeig_reject = _sequential_rank(maxeig_stat, maxeig_crit)

    table = pd.DataFrame(
        {
            "trace_stat": trace_stat,
            "trace_crit": trace_crit,
            "trace_reject": trace_reject,
            "maxeig_stat": maxeig_stat,
            "maxeig_crit": maxeig_crit,
            "maxeig_reject": maxeig_reject,
        },
        index=pd.Index(range(len(trace_stat)), name="r"),
    )

    return CointegrationTestResult(
        rank_trace=rank_trace,
        rank_max_eigen=rank_max_eigen,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
        alpha=alpha,
        n_obs=int(np.asarray(res.r0t).shape[0]),
        eigenvalues=np.asarray(res.eig, dtype=np.float64),
        table=table,
    )


def _joint_status(adf_reject: bool, kpss_reject: bool) -> str:
    """Combine the two tests' verdicts at one differencing level."""
    if adf_reject and not kpss_reject:
        return "stationary"
    if kpss_reject and not adf_reject:
        return "unit_root"
    if adf_reject and kpss_reject:
        return "conflicting"
    return "inconclusive"


def integration_order(
    data: VARData | pd.DataFrame | pd.Series,
    variables: Sequence[str] | None = None,
    *,
    max_order: int = 2,
    alpha: float = 0.05,
    regression: Literal["c", "ct"] = "c",
) -> IntegrationOrderResult:
    """Determine each series' integration order by repeated differencing.

    For every variable the series is tested at its level, then differenced
    and re-tested, until ADF rejects a unit root or `max_order` is reached.
    ADF drives the stopping rule. KPSS is run at every level as a cross-check
    and recorded in a `joint_status` column; where the two disagree, or where
    a series is still non-stationary at `max_order`, the variable is listed
    in `inconclusive` and the reported order should not be used without
    looking at the table.

    The returned `d_max` is the augmentation term a Toda-Yamamoto style
    procedure needs. Check `inconclusive` before using it: where a variable
    is listed there, its order — and therefore `d_max` — is a placeholder.

    Args:
        data: VARData (endogenous block only), DataFrame, or Series.
        variables: Subset of column names to test. Defaults to all.
        max_order: Highest order to search.
        alpha: Significance level for both tests. Restricted to the levels
            KPSS tabulates: 0.10, 0.05, 0.025, or 0.01.
        regression: Deterministic terms for the *level* test only. Pass
            `"ct"` when the levels trend. Differenced series are always
            tested with a constant, since differencing removes a linear
            trend.

    Returns:
        IntegrationOrderResult with per-variable orders and the full table.

    Raises:
        ValueError: If `max_order` is negative, `regression` is invalid, or
            `alpha` is not a level KPSS tabulates.
    """
    if max_order < 0:
        raise ValueError(f"max_order must be non-negative, got {max_order}")
    if regression not in _KPSS_REGRESSIONS:
        raise ValueError(f"regression must be one of {_KPSS_REGRESSIONS}, got {regression!r}")
    # Both tests share this alpha, and the KPSS cross-check needs a tabulated
    # level to compare its statistic against.
    _check_kpss_alpha(alpha)

    frame = _prepare(data, variables)

    order: dict[str, int] = {}
    inconclusive: list[str] = []
    keys: list[tuple[str, int]] = []
    rows: list[dict] = []

    for column in frame.columns:
        name = str(column)
        series = frame[column].to_numpy(dtype=np.float64)
        stopped_at: int | None = None
        status = "inconclusive"

        for d in range(max_order + 1):
            x = np.diff(series, n=d) if d else series
            # Differencing removes a linear trend, so only the level test
            # carries the caller's deterministic specification.
            reg = regression if d == 0 else "c"
            adf = _adf_single(x, regression=reg, max_lags=None, lag_selection="aic", alpha=alpha)
            kp = _kpss_single(x, regression=reg, nlags="auto", alpha=alpha)
            status = _joint_status(adf["reject"], kp["reject"])

            keys.append((name, d))
            rows.append({
                "adf_stat": adf["statistic"],
                "adf_pvalue": adf["pvalue"],
                "adf_lags": adf["lags"],
                "adf_reject": adf["reject"],
                "kpss_stat": kp["statistic"],
                "kpss_pvalue": kp["pvalue"],
                "kpss_lags": kp["lags"],
                "kpss_reject": kp["reject"],
                "kpss_pvalue_bounded": kp["pvalue_bounded"],
                "joint_status": status,
            })

            if adf["reject"]:
                stopped_at = d
                break

        if stopped_at is None:
            # Still non-stationary after max_order differences.
            order[name] = max_order
            inconclusive.append(name)
        else:
            order[name] = stopped_at
            if status == "conflicting":
                inconclusive.append(name)

    table = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(keys, names=["variable", "d"]))

    return IntegrationOrderResult(
        order=order,
        alpha=alpha,
        max_order=max_order,
        regression=regression,
        inconclusive=inconclusive,
        table=table,
    )
