"""Entropic tilting — post-hoc reweighting of forecast draws (ADR-0009).

Solves the minimum-relative-entropy problem of Robertson, Tallman and
Whiteman (2005),

    min_w  sum_i w_i log(N w_i)
    s.t.   sum_i w_i g_k(y_i) = t_k  (k = 1..K),  sum_i w_i = 1,  w >= 0,

over the draws of a forecast that has already been produced. The draws
never move — only their weights — so anything every draw already
satisfies (a hard pin from `conditional_forecast`, say) survives any
reweighting untouched.

Two solvers share one entry point. A single `ProbabilityTarget` reduces
to a two-mass problem with a closed form (`p / N_A` inside the event,
`(1 - p) / (N - N_A)` outside), so no optimiser runs. Anything else goes
through the convex dual: `lambda* = argmin log((1/N) sum_i
exp(lambda'(g_i - t)))`, minimised by BFGS with the analytic gradient
`E_lambda[g] - t` and a log-sum-exp-stabilised objective.

This module is pure numpy/scipy: impulso types are imported lazily or
under `TYPE_CHECKING` only.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from impulso.results import ConditionalForecastResult, ForecastResult, TiltedForecastResult
    from impulso.scenario import MomentTarget, ProbabilityTarget

    Target = ProbabilityTarget | MomentTarget

# Default warning threshold on ESS as a fraction of the draw count.
ESS_WARN_FRACTION = 0.1
# Post-solve tolerance on |achieved - requested| and the dual's norm guard.
ACHIEVED_TOL = 1e-6
LAMBDA_GUARD = 1e6


# --- moment construction -------------------------------------------------


def target_label(target: Target) -> str:
    """Human-readable label for a target, used as the `target` coordinate.

    The label names the *quantity* a target constrains — variable and
    horizon, plus threshold and direction for a `ProbabilityTarget` — but
    not the value requested of it, so two targets share a label exactly
    when they speak about the same event or the same mean. `build_moments`
    keys its duplicate handling on that: identical targets collapse to one
    column, and same-label targets asking for different values are
    rejected as contradictory.

    Args:
        target: The target to label.

    Returns:
        The label, e.g. `P(y1[h=4] < 0)` or `E[y2[h=2]]`.
    """
    from impulso.scenario import ProbabilityTarget

    if isinstance(target, ProbabilityTarget):
        sign = "<" if target.direction == "below" else ">"
        return f"P({target.variable}[h={target.horizon}] {sign} {target.threshold:g})"
    return f"E[{target.variable}[h={target.horizon}]]"


def build_moments(
    forecast: np.ndarray,
    targets: list[Target],
    var_names: list[str],
    steps: int,
) -> tuple[np.ndarray, np.ndarray, list[Target]]:
    """Build the moment matrix `G`, requested vector `t`, and target list.

    Column `k` holds the moment function of target `k` evaluated on every
    draw: the event indicator for a `ProbabilityTarget`, the level itself
    for a `MomentTarget`. Draws are flattened over `(chain, draw)` in C
    order, so `G[i]` lines up with `forecast.reshape(N, steps, n)[i]`.

    The targets are deduplicated first (see `dedupe_targets`), so the
    returned list — not the one passed in — is what the columns
    correspond to and what the caller must label the `target` coordinate
    with.

    Feasibility that does not depend on the solver is checked here: an
    event no draw satisfies (or, for `p < 1`, one every draw satisfies)
    and a requested mean outside the draws' range are unachievable by
    reweighting, whatever the optimiser does.

    Args:
        forecast: Forecast draws of shape `(C, D, steps, n)`.
        targets: The targets to impose (must be non-empty).
        var_names: Endogenous variable names.
        steps: Forecast horizon.

    Returns:
        Tuple `(G, t, targets)` with `G` of shape `(N, K)`, `t` of shape
        `(K,)`, and the `K` deduplicated targets in their original order.

    Raises:
        TypeError: If a target is neither a `ProbabilityTarget` nor a
            `MomentTarget`.
        ValueError: On an empty target list, two targets on the same
            quantity requesting different values, an unknown variable, a
            horizon beyond the forecast, or an unachievable target.
    """
    from impulso.scenario import ProbabilityTarget

    if not targets:
        raise ValueError("Tilting requires at least one target; an empty target list would leave the weights uniform.")
    targets = dedupe_targets(targets)
    n_chains, n_draws, _, n_vars = forecast.shape
    n_total = n_chains * n_draws
    flat = forecast.reshape(n_total, steps, n_vars)

    G = np.empty((n_total, len(targets)))
    t = np.empty(len(targets))
    for k, target in enumerate(targets):
        i, h = _resolve_target(target, var_names, steps)
        y = flat[:, h, i]
        if isinstance(target, ProbabilityTarget):
            event = (y < target.threshold) if target.direction == "below" else (y > target.threshold)
            _check_event_support(target, int(event.sum()), n_total)
            G[:, k] = event.astype(float)
            t[k] = target.probability
        else:
            _check_moment_support(target, y, n_total)
            G[:, k] = y
            t[k] = target.mean
    return G, t, targets


def dedupe_targets(targets: list[Target]) -> list[Target]:
    """Drop repeated targets and reject same-quantity targets that disagree.

    Labels key the `target` coordinate of a tilted result, so a repeated
    label would make `.sel(target=...)` return several rows. A target
    passed twice verbatim is harmless redundancy — one moment column
    achieves it exactly as the two identical ones would — so the repeat
    is dropped silently and the first occurrence kept, preserving order.

    Two *different* targets sharing a label are another matter: they ask
    the same event to carry two probabilities (or the same mean to take
    two values), which no reweighting can do. Without this check they
    surface late, as a joint-infeasibility failure from the dual, naming
    neither the label nor the values that clash.

    Args:
        targets: The targets as passed by the caller.

    Returns:
        The deduplicated targets, first occurrence first.

    Raises:
        TypeError: If a target is neither a `ProbabilityTarget` nor a
            `MomentTarget`.
        ValueError: If two targets share a label but are not identical.
    """
    from impulso.scenario import MomentTarget, ProbabilityTarget

    first_seen: dict[str, Target] = {}
    unique: list[Target] = []
    for target in targets:
        if not isinstance(target, ProbabilityTarget | MomentTarget):
            raise TypeError(f"tilt() accepts ProbabilityTarget or MomentTarget, got {type(target).__name__}")
        label = target_label(target)
        first = first_seen.get(label)
        if first is None:
            first_seen[label] = target
            unique.append(target)
        elif first != target:
            raise ValueError(
                f"Conflicting targets for {label}: {_requested_value(first)} and "
                f"{_requested_value(target)}. A target label pins the quantity, not the value "
                "asked of it, so these are two contradictory constraints on the same "
                "quantity — keep one."
            )
    return unique


def _requested_value(target: Target) -> str:
    """The value a target asks for, for the conflicting-targets message."""
    from impulso.scenario import ProbabilityTarget

    if isinstance(target, ProbabilityTarget):
        return f"probability={target.probability:g}"
    return f"mean={target.mean:g}"


def _resolve_target(target: Target, var_names: list[str], steps: int) -> tuple[int, int]:
    """Resolve a target to `(variable_index, 0-based step index)`."""
    if target.variable not in var_names:
        raise ValueError(f"Unknown variable {target.variable!r}; available variables: {var_names}")
    if target.horizon > steps:
        raise ValueError(
            f"Target for {target.variable!r} refers to horizon {target.horizon} "
            f"but the forecast has only {steps} steps."
        )
    return var_names.index(target.variable), target.horizon - 1


def _check_event_support(target: ProbabilityTarget, n_event: int, n_total: int) -> None:
    """Reject probability targets no reweighting of these draws can hit."""
    label = target_label(target)
    if n_event == 0:
        raise ValueError(
            f"0 of {n_total} draws satisfy {label}; the target is unachievable by reweighting "
            "— widen the threshold or increase the number of draws."
        )
    if n_event == n_total and target.probability < 1.0:
        raise ValueError(
            f"All {n_total} of {n_total} draws satisfy {label}, so the only achievable "
            f"probability is 1.0, not {target.probability:g}; tighten the threshold or "
            "increase the number of draws."
        )


def _check_moment_support(target: MomentTarget, y: np.ndarray, n_total: int) -> None:
    """Reject mean targets outside the convex hull of the draws."""
    lo, hi = float(y.min()), float(y.max())
    if not lo < target.mean < hi:
        raise ValueError(
            f"{target_label(target)} = {target.mean:g} lies outside the range spanned by the "
            f"{n_total} draws ([{lo:g}, {hi:g}]); reweighting cannot move mass where there is "
            "none — relax the target or increase the number of draws."
        )


# --- solvers -------------------------------------------------------------


def solve_tilt(G: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve for the minimum-relative-entropy weights.

    A single binary moment column (one `ProbabilityTarget`) takes the
    closed-form two-mass solution; everything else takes the convex dual.

    Args:
        G: Moment matrix `(N, K)` from `build_moments`.
        t: Requested moments `(K,)`.

    Returns:
        Tuple `(weights, achieved)`: weights `(N,)` summing to 1 and the
        achieved moments `G' w` `(K,)`.

    Raises:
        ValueError: If the dual fails to reproduce the requested moments
            (jointly infeasible targets).
    """
    if G.shape[1] == 1 and _is_binary(G[:, 0]):
        return _closed_form_two_mass(G[:, 0], float(t[0]))
    return _dual_solve(G, t)


def _is_binary(column: np.ndarray) -> bool:
    """Whether a moment column is a 0/1 indicator."""
    return bool(np.all((column == 0.0) | (column == 1.0)))


def _closed_form_two_mass(indicator: np.ndarray, probability: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form tilt for a single event probability.

    Relative entropy is minimised by keeping the weights uniform *within*
    the event and within its complement, so the whole problem collapses
    to splitting mass `p` over `N_A` draws and `1 - p` over the rest.
    At `p = 1` this is exact conditioning on the event.
    """
    in_event = indicator > 0.5
    n_total = indicator.size
    n_event = int(in_event.sum())
    weights = np.empty(n_total)
    if probability >= 1.0:
        weights[in_event] = 1.0 / n_event
        weights[~in_event] = 0.0
    else:
        weights[in_event] = probability / n_event
        weights[~in_event] = (1.0 - probability) / (n_total - n_event)
    return weights, np.array([float(weights[in_event].sum())])


def _dual_solve(G: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimise the convex dual with BFGS, then polish with Newton steps.

    BFGS on the log-sum-exp objective with the analytic gradient gets
    close; its default gradient tolerance is looser than the moment
    tolerance we then check against, so a handful of Newton steps on the
    same objective (whose Hessian is the tilted covariance of `g`) take
    the residual to machine precision. Both stages are on the same convex
    dual, so the polish cannot move to a different optimum.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    n_total, n_targets = G.shape
    centred = G - t

    def objective(lam: np.ndarray) -> tuple[float, np.ndarray]:
        z = centred @ lam
        normaliser = logsumexp(z)
        w = np.exp(z - normaliser)
        return float(normaliser - np.log(n_total)), centred.T @ w

    result = minimize(objective, np.zeros(n_targets), jac=True, method="BFGS", options={"gtol": 1e-12})
    lam = _newton_polish(centred, np.asarray(result.x, dtype=float))
    weights = _weights_from_dual(centred, lam)
    achieved = G.T @ weights
    _check_dual_solution(achieved, t, lam)
    return weights, achieved


def _newton_polish(centred: np.ndarray, lam: np.ndarray, max_steps: int = 50) -> np.ndarray:
    """Newton iterations on the dual until the moment residual is machine-small."""
    for _ in range(max_steps):
        w = _weights_from_dual(centred, lam)
        gradient = centred.T @ w
        if np.max(np.abs(gradient)) <= 1e-14:
            break
        centred_mean = centred - gradient
        hessian = (centred_mean * w[:, np.newaxis]).T @ centred_mean
        step, *_ = np.linalg.lstsq(hessian, gradient, rcond=None)
        if not np.all(np.isfinite(step)):
            break
        lam = lam - step
    return lam


def _weights_from_dual(centred: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Normalised tilting weights `w_i ∝ exp(lambda'(g_i - t))`."""
    from scipy.special import logsumexp

    z = centred @ lam
    weights = np.exp(z - logsumexp(z))
    return weights / weights.sum()


def _check_dual_solution(achieved: np.ndarray, t: np.ndarray, lam: np.ndarray) -> None:
    """Reject a dual solution that misses the requested moments."""
    gap = float(np.max(np.abs(achieved - t)))
    norm = float(np.linalg.norm(lam))
    if gap > ACHIEVED_TOL or norm > LAMBDA_GUARD:
        raise ValueError(
            "The requested targets are not jointly achievable by reweighting these draws: the "
            f"entropic dual reached requested={np.array2string(t, precision=6)} vs "
            f"achieved={np.array2string(achieved, precision=6)} (max gap {gap:.3g}, "
            f"|lambda| = {norm:.3g}). Relax a target or increase the number of draws."
        )


# --- diagnostics ---------------------------------------------------------


def ess(weights: np.ndarray) -> float:
    """Kish effective sample size `1 / sum_i w_i^2`.

    Equals `N` for uniform weights and `1` when all mass sits on one
    draw, so `ess / N` reads directly as the fraction of the sample the
    tilt actually uses.
    """
    return float(1.0 / np.sum(weights**2))


def kl_divergence(weights: np.ndarray) -> float:
    """Relative entropy `sum_i w_i log(N w_i)` of the tilt from uniform.

    This is the objective the tilt minimises, and (unlike the hard-pin
    case of ADR-0005) a genuinely finite divergence — soft conditioning
    keeps the tilted law absolutely continuous with respect to the
    untilted one. Zero-weight draws contribute nothing.
    """
    n_total = weights.size
    positive = weights[weights > 0.0]
    return float(np.sum(positive * np.log(n_total * positive)))


# --- weighted summaries --------------------------------------------------


def _sorted_columns(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Sort each column of `x` and carry the weights along.

    Returns `(sorted values, normalised sorted weights, trailing shape)`
    with both arrays shaped `(N, M)` where `M` is the product of the
    trailing dimensions of `x`.
    """
    x = np.asarray(x, dtype=float)
    n_total = weights.size
    if x.shape[0] != n_total:
        raise ValueError(f"x must have {n_total} draws on its leading axis, got {x.shape[0]}")
    trailing = x.shape[1:]
    flat = x.reshape(n_total, -1)
    order = np.argsort(flat, axis=0, kind="stable")
    values = np.take_along_axis(flat, order, axis=0)
    w = weights[order]
    return values, w / w.sum(axis=0, keepdims=True), trailing


def weighted_quantile(x: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray | float:
    """Weighted quantile with the mid-cumulative-weight interpolation.

    Each sorted draw `j` is placed at `c_j = W_j - w_j / 2` (its
    cumulative weight less half its own mass) and the quantile is read
    off by linear interpolation between those knots, clamped at the
    extremes. Under uniform weights this reproduces
    `np.quantile(..., method="hazen")` exactly.

    Args:
        x: Values with draws on the leading axis, shape `(N, ...)`.
        weights: Non-negative weights `(N,)`; normalised internally.
        q: Quantile level in `[0, 1]`.

    Returns:
        Array shaped like `x.shape[1:]`, or a float when `x` is 1-D.
    """
    values, w, trailing = _sorted_columns(x, weights)
    knots = np.cumsum(w, axis=0) - w / 2.0
    out = np.array([np.interp(q, knots[:, m], values[:, m]) for m in range(values.shape[1])])
    return float(out[0]) if not trailing else out.reshape(trailing)


def weighted_hdi(x: np.ndarray, weights: np.ndarray, prob: float = 0.89) -> tuple[np.ndarray, np.ndarray]:
    """Weighted highest-density interval by minimum width.

    Scans every sorted draw as a candidate left endpoint, takes the
    first right endpoint carrying at least `prob` of the weight, and
    keeps the narrowest such interval. Under uniform weights and a
    non-integer `prob * N` this reproduces `arviz.hdi`.

    Args:
        x: Values with draws on the leading axis, shape `(N, ...)`.
        weights: Non-negative weights `(N,)`; normalised internally.
        prob: Probability mass the interval must carry.

    Returns:
        Tuple `(lower, upper)`, each shaped like `x.shape[1:]` (0-D
        arrays when `x` is 1-D).

    Raises:
        ValueError: If `prob` is not in `(0, 1]`.
    """
    if not 0.0 < prob <= 1.0:
        raise ValueError(f"prob must satisfy 0 < prob <= 1, got {prob}")
    values, w, trailing = _sorted_columns(x, weights)
    cumulative = np.cumsum(w, axis=0)
    below = cumulative - w  # weight strictly left of each candidate start
    n_cols = values.shape[1]
    lower = np.empty(n_cols)
    upper = np.empty(n_cols)
    for m in range(n_cols):
        lower[m], upper[m] = _min_width_interval(values[:, m], cumulative[:, m], below[:, m], prob)
    return lower.reshape(trailing), upper.reshape(trailing)


def _min_width_interval(
    values: np.ndarray,
    cumulative: np.ndarray,
    below: np.ndarray,
    prob: float,
) -> tuple[float, float]:
    """Narrowest interval of sorted draws carrying at least `prob` mass."""
    # searchsorted on the cumulative weights gives, for each start i, the
    # first index j with cumulative[j] >= below[i] + prob (the two-pointer
    # sweep, vectorised — cumulative is non-decreasing).
    ends = np.searchsorted(cumulative, below + prob - 1e-12, side="left")
    feasible = ends < values.size
    if not feasible.any():
        return float(values[0]), float(values[-1])
    starts = np.flatnonzero(feasible)
    ends = ends[feasible]
    widths = values[ends] - values[starts]
    best = int(np.argmin(widths))
    return float(values[starts[best]]), float(values[ends[best]])


# --- result assembly -----------------------------------------------------


def tilt_result(
    result: ForecastResult | ConditionalForecastResult,
    targets: list[Target],
    ess_warn_fraction: float = ESS_WARN_FRACTION,
) -> TiltedForecastResult:
    """Tilt an existing forecast result onto the requested targets.

    The parent's `"forecast"` DataArray is carried into the new result by
    reference — tilting adds weights, it does not copy or move draws.

    Args:
        result: A density-mode `ForecastResult` or
            `ConditionalForecastResult` (`ScenarioResult` included).
        targets: `ProbabilityTarget` / `MomentTarget` list. Repeats are
            dropped, so the result's `targets` and its `target`
            coordinate carry one entry per distinct target.
        ess_warn_fraction: Warn when the effective sample size falls
            below this fraction of the draw count.

    Returns:
        A frozen `TiltedForecastResult`.

    Raises:
        ValueError: If the parent result is a mean forecast, if
            `ess_warn_fraction` is outside `[0, 1]`, if two targets
            constrain the same quantity with different values, or if the
            targets are unachievable.
    """
    import arviz as az
    import xarray as xr

    from impulso.results import TiltedForecastResult
    from impulso.scenario import ProbabilityTarget

    if result.mode != "density":
        raise ValueError(
            f"Entropic tilting needs a density forecast, but this result is a {result.mode!r} "
            "forecast: mean-mode draws carry parameter uncertainty only, so probabilities read "
            "off them are not predictive probabilities. Re-run with "
            "include_shock_uncertainty=True."
        )
    if not 0.0 <= ess_warn_fraction <= 1.0:
        raise ValueError(f"ess_warn_fraction must lie in [0, 1], got {ess_warn_fraction}")

    da = result.idata.posterior_predictive["forecast"]
    # build_moments returns the deduplicated targets: one column, one
    # label, one row of achieved/requested per distinct target.
    G, t, targets = build_moments(da.values, targets, result.var_names, result.steps)
    weights, achieved = solve_tilt(G, t)

    n_chains, n_draws = da.shape[:2]
    labels = [target_label(target) for target in targets]
    counts = np.array([
        float(G[:, k].sum()) if isinstance(target, ProbabilityTarget) else np.nan for k, target in enumerate(targets)
    ])
    diagnostics = tilt_diagnostics(weights, ess_warn_fraction, stacklevel=4)

    ds = xr.Dataset({
        "forecast": da,
        "tilting_weights": xr.DataArray(weights.reshape(n_chains, n_draws), dims=["chain", "draw"]),
        "achieved": xr.DataArray(achieved, dims=["target"], coords={"target": labels}),
        "requested": xr.DataArray(t, dims=["target"], coords={"target": labels}),
        "event_draws": xr.DataArray(counts, dims=["target"], coords={"target": labels}),
    })
    ds.attrs.update(diagnostics)
    return TiltedForecastResult(
        idata=az.InferenceData(posterior_predictive=ds),
        steps=result.steps,
        var_names=list(result.var_names),
        targets=targets,
    )


def tilt_diagnostics(
    weights: np.ndarray,
    ess_warn_fraction: float = ESS_WARN_FRACTION,
    stacklevel: int = 3,
) -> dict[str, float]:
    """Effective sample size and relative entropy, with the degeneracy warning.

    Args:
        weights: Normalised tilting weights `(N,)`.
        ess_warn_fraction: Warn below this ESS fraction.
        stacklevel: `warnings.warn` stacklevel targeting the public caller.

    Returns:
        Dict with `ess`, `ess_fraction`, and `kl_divergence`.
    """
    n_total = weights.size
    ess_value = ess(weights)
    fraction = ess_value / n_total
    if fraction < ess_warn_fraction:
        warnings.warn(
            f"The tilt is concentrated: effective sample size {ess_value:.1f} of {n_total} draws "
            f"({fraction:.1%}, below the {ess_warn_fraction:.0%} threshold). Tilted summaries rest "
            "on few draws and are noisy — relax the targets or draw a larger forecast sample.",
            UserWarning,
            stacklevel=stacklevel,
        )
    return {"ess": ess_value, "ess_fraction": fraction, "kl_divergence": kl_divergence(weights)}
