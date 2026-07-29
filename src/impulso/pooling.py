"""Static predictive-density pooling.

Score several fitted models' predictive densities on a held-out window, turn
those scores into weights, and combine the models into a single predictive
distribution. `pool_forecasts` estimates the weights; `PredictivePool.combine`
applies them to new forecasts. See ADR-0006 for the design and its limits.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import arviz as az
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pydantic import Field, model_validator

from impulso._base import ImpulsoBaseModel
from impulso.data import VARData
from impulso.results import ForecastResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    from impulso.fitted import FittedVAR

WeightMethod = Literal["stacking", "log_score"]
"""How held-out log scores become weights."""

DensityKind = Literal["gaussian", "diagonal"]
"""How forecast draws become a predictive density."""

_COV_JITTER = 1e-10
_LOG_2PI = float(np.log(2.0 * np.pi))
_NO_OPTIMISER = "log-score weights are closed-form; no optimiser was run."


# --------------------------------------------------------------------------
# Predictive densities: forecast draws -> (H,) log scores
# --------------------------------------------------------------------------


def _degenerate_density_message(label: str, h: int, reason: str) -> str:
    return (
        f"Model {label!r} has a degenerate joint predictive density at holdout step {h + 1}: "
        f"{reason}. Pass density='diagonal' to score each variable on its own marginal, "
        "or draw more posterior samples."
    )


def _joint_log_scores(draws: np.ndarray, y: np.ndarray, label: str) -> np.ndarray:
    """Moment-matched joint Gaussian log density, one score per horizon."""
    from scipy.linalg import solve_triangular

    _, n_steps, n_vars = draws.shape
    scores = np.empty(n_steps)
    eye = np.eye(n_vars)
    for h in range(n_steps):
        block = draws[:, h, :]
        mean = block.mean(axis=0)
        cov = np.cov(block, rowvar=False, ddof=1)
        mean_var = float(np.mean(np.diag(cov)))
        if mean_var <= 0.0:
            raise ValueError(_degenerate_density_message(label, h, "every forecast draw is identical"))
        # The sample covariance is positive semi-definite by construction, so
        # proving the smallest eigenvalue clears the jitter both rejects
        # singular densities (where the jitter, not the data, would be doing
        # the work) and bounds the jittered condition number at ~n/_COV_JITTER,
        # which Cholesky handles comfortably in float64.
        if float(np.linalg.eigvalsh(cov)[0]) <= _COV_JITTER * mean_var:
            raise ValueError(_degenerate_density_message(label, h, "the predictive covariance is numerically singular"))
        chol = np.linalg.cholesky(cov + _COV_JITTER * mean_var * eye)
        z = solve_triangular(chol, y[h] - mean, lower=True)
        scores[h] = -0.5 * (n_vars * _LOG_2PI + 2.0 * float(np.sum(np.log(np.diag(chol)))) + float(z @ z))
    return scores


def _diagonal_log_scores(draws: np.ndarray, y: np.ndarray, label: str) -> np.ndarray:
    """Per-variable normal log density summed over variables, one score per horizon."""
    n_vars = draws.shape[2]
    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)
    if not bool((var > 0.0).all()):
        h, v = (int(i) for i in np.argwhere(var <= 0.0)[0])
        raise ValueError(
            f"Model {label!r} has zero forecast variance for variable index {v} at holdout step {h + 1}, "
            "so its marginal predictive density is degenerate. Draw more posterior samples, or forecast "
            "in density mode (include_shock_uncertainty=True)."
        )
    z = (y - mean) / np.sqrt(var)
    return -0.5 * (n_vars * _LOG_2PI + np.sum(np.log(var), axis=1) + np.sum(z**2, axis=1))


def _gaussian_log_scores(
    draws: np.ndarray,
    y: np.ndarray,
    density: DensityKind = "gaussian",
    label: str = "",
) -> np.ndarray:
    """Log predictive density of each held-out row under a draw-based Gaussian.

    Args:
        draws: Flattened forecast draws of shape `(S, H, n)` — posterior
            draws by horizon by variable.
        y: Held-out realisations of shape `(H, n)`.
        density: `"gaussian"` for a joint density across variables (moments
            matched to the draws), `"diagonal"` for the sum of per-variable
            marginals.
        label: Model label, used only in error messages.

    Returns:
        Array of `H` log scores.
    """
    n_sims, n_steps, n_vars = draws.shape
    if y.shape != (n_steps, n_vars):
        raise ValueError(f"Held-out array has shape {y.shape} but the draws imply {(n_steps, n_vars)}.")
    if not bool(np.isfinite(draws).all()):
        raise ValueError(
            f"Model {label!r} produced non-finite forecast draws (NaN or Inf); its predictive density "
            "cannot be scored. This usually means an explosive posterior draw — check the fit."
        )
    if n_sims < 2:
        raise ValueError(f"Model {label!r} has {n_sims} posterior draw(s); scoring needs at least two.")
    if density == "diagonal":
        return _diagonal_log_scores(draws, y, label)
    if n_sims <= n_vars:
        raise ValueError(
            f"Model {label!r} has {n_sims} posterior draws for {n_vars} variables; with S <= n the joint "
            "predictive covariance is rank-deficient. Sample more draws, or pass density='diagonal'."
        )
    return _joint_log_scores(draws, y, label)


# --------------------------------------------------------------------------
# Weight solvers: (H, M) log-score matrix -> (M,) weights
# --------------------------------------------------------------------------


def _check_score_matrix(log_scores: np.ndarray, index: pd.Index | None = None) -> None:
    """Reject score matrices no weight vector can be fitted to."""
    if log_scores.ndim != 2:
        raise ValueError(f"The log-score matrix must be 2-D (horizons by models), got {log_scores.ndim}-D.")
    if log_scores.shape[1] < 2:
        raise ValueError(f"Pooling requires at least two fitted models, got {log_scores.shape[1]}.")
    if bool(np.isnan(log_scores).any()) or bool(np.isposinf(log_scores).any()):
        raise ValueError("The log-score matrix contains NaN or +inf entries; log densities must be finite or -inf.")
    dead = ~np.isfinite(log_scores).any(axis=1)
    if bool(dead.any()):
        pos = int(np.flatnonzero(dead)[0])
        where = f"{index[pos]}" if index is not None else f"holdout step {pos + 1}"
        raise ValueError(
            f"No model assigns positive predictive density at {where}, so the pooled score is -inf for every "
            "weight vector. Check that point for an outlier or level shift, or widen the predictive densities "
            "(more posterior draws, or density='diagonal')."
        )


def _log_score_weights(log_scores: np.ndarray, index: pd.Index | None = None) -> np.ndarray:
    """Softmax of each model's total held-out log score (pseudo-BMA weights).

    Shifting by the maximum total before exponentiating keeps the weights
    finite at log scores where an unshifted implementation overflows.
    """
    _check_score_matrix(log_scores, index)
    totals = log_scores.sum(axis=0)
    if not bool(np.isfinite(totals).any()):
        raise ValueError(
            "Every model scores -inf somewhere in the holdout, so every total log score is -inf and "
            "log-score weights are undefined. Use method='stacking', which scores the pooled density "
            "row by row rather than model by model."
        )
    weights = np.exp(totals - totals.max())
    return weights / weights.sum()


def _stacking_weights(log_scores: np.ndarray, index: pd.Index | None = None) -> tuple[np.ndarray, bool, str]:
    """Weights maximising the log score of the pooled predictive.

    The objective is convex on the simplex, so the SLSQP solution is the
    global optimum. Densities are exponentiated after a per-row maximum
    shift, which changes the objective by a `w`-independent constant and
    therefore leaves the optimum untouched while keeping the pool finite at
    log scores that would otherwise underflow to zero.

    Returns:
        Tuple of `(weights, converged, optimiser_message)`.
    """
    from scipy.optimize import Bounds, LinearConstraint, minimize

    _check_score_matrix(log_scores, index)
    n_models = log_scores.shape[1]
    densities = np.exp(log_scores - log_scores.max(axis=1, keepdims=True))

    def objective(w: np.ndarray) -> float:
        return -float(np.sum(np.log(densities @ w)))

    def gradient(w: np.ndarray) -> np.ndarray:
        return -np.sum(densities / (densities @ w)[:, None], axis=0)

    solver_kwargs = {
        "fun": objective,
        "jac": gradient,
        "method": "SLSQP",
        "bounds": Bounds(0.0, 1.0),
        "constraints": LinearConstraint(np.ones((1, n_models)), 1.0, 1.0),
        "options": {"ftol": 1e-12, "maxiter": 1000},
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        result = minimize(x0=np.full(n_models, 1.0 / n_models), **solver_kwargs)
        if not result.success:
            # Restart from a data-driven point on the simplex; unlike the
            # log-score weights this is finite even when every model scores
            # -inf somewhere.
            column_mass = densities.sum(axis=0)
            result = minimize(x0=column_mass / column_mass.sum(), **solver_kwargs)
    if not result.success:
        raise RuntimeError(
            f"Stacking weights failed to converge: {result.message}. The objective is convex on the "
            "simplex, so this points at a pathological score matrix — inspect the log scores, or fall "
            "back to method='log_score'."
        )
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    return weights / weights.sum(), bool(result.success), str(result.message)


def _pooled_row_scores(log_scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Log score of the pooled predictive at each held-out point."""
    from scipy.special import logsumexp

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(logsumexp(log_scores + np.log(weights), axis=1), dtype=float)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def _resolve_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    try:
        return np.random.default_rng(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seed must be an int, None, or a numpy.random.Generator, got {type(seed).__name__}.") from exc


def _spawn(rng: np.random.Generator, count: int) -> list[np.random.Generator]:
    try:
        return list(rng.spawn(count))
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            "seed must be an int, None, or a numpy.random.Generator that supports spawning "
            "(one built by numpy.random.default_rng does)."
        ) from exc


def _index_freq(index: pd.DatetimeIndex) -> pd.offsets.BaseOffset | None:
    """Best-effort frequency for a DatetimeIndex, or None."""
    if index.freq is not None:
        return index.freq
    if len(index) >= 3:
        inferred = pd.infer_freq(index)
        if inferred is not None:
            return pd.tseries.frequencies.to_offset(inferred)
    return None


def _check_variables(label: str, fit_names: list[str], holdout_names: list[str]) -> None:
    if fit_names == holdout_names:
        return
    message = (
        f"Model {label!r} was fitted on variables {fit_names} but the holdout carries "
        f"{holdout_names}; pooled models must share the holdout's variables"
    )
    if set(fit_names) == set(holdout_names):
        raise ValueError(f"{message} in the same order — reorder the holdout columns to {fit_names}.")
    raise ValueError(f"{message}.")


def _check_exog(label: str, fit: FittedVAR, holdout: VARData) -> None:
    if not fit.has_exog:
        return
    if holdout.exog is None:
        raise ValueError(
            f"Model {label!r} was fitted with exogenous regressors, so the holdout must carry their "
            "future values too; rebuild it with VARData(..., exog=..., exog_names=...)."
        )
    fit_names = list(fit.data.exog_names or [])
    holdout_names = list(holdout.exog_names or [])
    if fit_names != holdout_names:
        raise ValueError(
            f"Model {label!r} was fitted with exogenous regressors named {fit_names} but the holdout "
            f"carries {holdout_names}; they must match in name and order."
        )


def _check_alignment(train_index: pd.DatetimeIndex, holdout: VARData, origin: pd.Timestamp) -> None:
    """Require the holdout to continue the estimation sample without a gap."""
    freq = _index_freq(train_index) or _index_freq(holdout.index)
    if freq is None:
        warnings.warn(
            "Could not infer a frequency for the estimation sample or the holdout, so the held-out "
            "dates are assumed to line up positionally with forecast steps 1..H. Pass data with a "
            "regular DatetimeIndex if you want that checked.",
            UserWarning,
            stacklevel=4,
        )
        return
    expected = pd.date_range(origin, periods=len(holdout.index) + 1, freq=freq)[1:]
    mismatch = np.flatnonzero(expected.to_numpy() != holdout.index.to_numpy())
    if mismatch.size:
        first = int(mismatch[0])
        raise ValueError(
            f"The holdout does not continue the estimation sample at step {first + 1}: expected "
            f"{expected[first].date()} at frequency {freq.freqstr}, got {holdout.index[first].date()}. "
            "Pooled scores are indexed by forecast step, so the holdout must be the H periods "
            "immediately after the forecast origin."
        )


def _validate_pool_inputs(fits: Mapping[str, FittedVAR], holdout: VARData) -> pd.Timestamp:
    """Check a pool's inputs and return the shared forecast origin."""
    if len(fits) < 2:
        raise ValueError(f"Pooling requires at least two fitted models, got {len(fits)}.")
    if len(holdout.index) < 1:
        raise ValueError("Pooling needs at least one held-out observation to score; the holdout is empty.")

    ends = {}
    for label, fit in fits.items():
        _check_variables(label, list(fit.var_names), list(holdout.endog_names))
        _check_exog(label, fit, holdout)
        ends[label] = fit.data.index[-1]
    if len(set(ends.values())) > 1:
        stamps = ", ".join(f"{label}={end.date()}" for label, end in ends.items())
        raise ValueError(
            f"Pooled models were estimated to different sample ends: {stamps}; pooling compares "
            "forecasts made from a single origin, so every model must end on the same date."
        )

    origin = next(iter(ends.values()))
    if holdout.index[0] <= origin:
        raise ValueError(
            f"The holdout starts at {holdout.index[0].date()} but the models are estimated through "
            f"{origin.date()}; the holdout must postdate the estimation sample or the scores are in-sample."
        )
    _check_alignment(next(iter(fits.values())).data.index, holdout, origin)
    return origin


# --------------------------------------------------------------------------
# Mixture sampling
# --------------------------------------------------------------------------


def _mixture_draws(
    stacks: list[np.ndarray],
    weights: np.ndarray,
    n_draws: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a pooled sample by picking a model per draw, then a draw within it.

    Returns:
        Tuple of `(pooled draws of shape (N, H, n), membership of shape (N,))`.
    """
    sizes = np.array([stack.shape[0] for stack in stacks])
    total = int(sizes.min()) if n_draws is None else int(n_draws)
    membership = rng.choice(len(stacks), size=total, p=weights)
    position = np.floor(rng.random(total) * sizes[membership]).astype(int)
    pooled = np.empty((total, *stacks[0].shape[1:]))
    for i, stack in enumerate(stacks):
        picked = membership == i
        if picked.any():
            pooled[picked] = stack[position[picked]]
    return pooled, membership


def _pooled_forecast_result(
    pooled: np.ndarray,
    membership: np.ndarray,
    labels: list[str],
    var_names: list[str],
    time_index: pd.DatetimeIndex | None = None,
) -> ForecastResult:
    """Wrap pooled draws as a `(chain=1, draw=N, step, variable)` ForecastResult."""
    import xarray as xr

    steps = pooled.shape[1]
    coords: dict[str, object] = {
        "variable": var_names,
        "model": ("draw", np.asarray(labels, dtype=object)[membership]),
    }
    if time_index is not None:
        coords["time"] = ("step", time_index.to_numpy())
    forecast = xr.DataArray(
        pooled[np.newaxis],
        dims=["chain", "draw", "step", "variable"],
        coords=coords,
        name="forecast",
    )
    idata = az.InferenceData(posterior_predictive=xr.Dataset({"forecast": forecast}))
    return ForecastResult(idata=idata, steps=steps, var_names=list(var_names), mode="density")


def _flatten(forecast: ForecastResult) -> np.ndarray:
    """Collapse a ForecastResult's `(chain, draw, step, variable)` draws to `(S, H, n)`."""
    values = forecast.idata.posterior_predictive["forecast"].values
    return values.reshape(-1, *values.shape[2:])


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


class PredictivePool(ImpulsoBaseModel):
    """A set of fitted models with one weight each, defining a pooled predictive.

    Produced by `pool_forecasts`, which scores every model's density forecast
    on a held-out window. The weights are *static* — one per model, fixed
    across horizons and across time — and they are frozen once estimated:
    `combine` applies them to new forecasts without rescoring.

    Attributes:
        weights: Weight per model, indexed by label, summing to 1.
        log_scores: Log predictive density per held-out date (index) and
            model (columns) — the contract between scoring and weighting.
        method: Which weight rule produced `weights`.
        density: How forecast draws were turned into a predictive density.
        var_names: Endogenous variables, in the order every model shares.
        steps: Number of held-out periods scored, `H`.
        origin: Shared forecast origin — the last date of the estimation
            sample every model was fitted on.
        holdout_predictive: The pooled predictive over the *held-out window*,
            as a `ForecastResult` with a `(chain=1, draw=N, step, variable)`
            layout. For genuine forecasts, refit on the full sample and use
            `combine`.
        membership: Which model produced each pooled draw, as an index into
            `labels`. Read-only.
        converged: Whether the weight solver converged. Always True for
            `method="log_score"`, which is closed-form.
        optimiser_message: The solver's own status message.
    """

    weights: pd.Series
    log_scores: pd.DataFrame = Field(repr=False)
    method: WeightMethod
    density: DensityKind
    var_names: list[str]
    steps: int
    origin: pd.Timestamp
    holdout_predictive: ForecastResult = Field(repr=False)
    membership: np.ndarray = Field(repr=False)
    converged: bool = True
    optimiser_message: str = ""

    @model_validator(mode="after")
    def _validate(self) -> PredictivePool:
        weights = self.weights.rename_axis("model")
        log_scores = self.log_scores.rename_axis(columns="model")
        if list(weights.index) != list(log_scores.columns):
            raise ValueError(
                f"Weight labels {list(weights.index)} do not match the log-score columns {list(log_scores.columns)}."
            )
        values = weights.to_numpy(dtype=float)
        if bool((values < -1e-12).any()):
            raise ValueError(f"Pool weights must be non-negative, got {values.tolist()}.")
        if not np.isclose(values.sum(), 1.0, atol=1e-8):
            raise ValueError(f"Pool weights must sum to 1, got {values.sum()!r}.")

        membership = np.asarray(self.membership)
        if membership.ndim != 1:
            raise ValueError(f"membership must be 1-D, got {membership.ndim}-D.")
        if membership.size and (membership.min() < 0 or membership.max() >= values.size):
            raise ValueError(
                f"membership indexes {values.size} models but ranges over [{membership.min()}, {membership.max()}]."
            )
        membership = membership.astype(int, copy=True)
        membership.flags.writeable = False

        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "log_scores", log_scores)
        object.__setattr__(self, "membership", membership)
        return self

    @property
    def labels(self) -> list[str]:
        """Model labels, in the order the pool was built."""
        return list(self.weights.index)

    def pooled_log_score(self) -> float:
        """Total held-out log score of the pooled predictive."""
        return float(_pooled_row_scores(self.log_scores.to_numpy(), self.weights.to_numpy()).sum())

    def summary(self) -> pd.DataFrame:
        """Per-model weights and held-out scores, heaviest weight first.

        Returns:
            DataFrame indexed by label with `weight`, `log_score` (total over
            the holdout), `mean_log_score` (per held-out period), and `rank`.
        """
        totals = self.log_scores.sum(axis=0)
        frame = pd.DataFrame({
            "weight": self.weights,
            "log_score": totals,
            "mean_log_score": totals / self.steps,
        }).sort_values("weight", ascending=False)
        frame["rank"] = np.arange(1, len(frame) + 1)
        return frame

    def to_dataframe(self) -> pd.DataFrame:
        """Per-date log scores for every model plus the pooled predictive."""
        frame = self.log_scores.copy()
        frame["pooled"] = _pooled_row_scores(self.log_scores.to_numpy(), self.weights.to_numpy())
        return frame

    def realised_weights(self) -> pd.Series:
        """Empirical share of each model among the pooled draws.

        The Monte Carlo counterpart of `weights`: it converges to them as the
        pooled sample grows, and the gap is sampling noise, not a second
        estimate.
        """
        counts = np.bincount(self.membership, minlength=len(self.labels))
        return pd.Series(counts / counts.sum(), index=self.weights.index, name="realised_weight")

    def combine(
        self,
        forecasts: Mapping[str, ForecastResult],
        n_draws: int | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> ForecastResult:
        """Apply the frozen weights to a new set of forecasts.

        The usual workflow: estimate weights on a held-out window, refit every
        model on the full sample, forecast past the end of the data, and
        combine. The new forecasts need not have the pool's horizon — only the
        same models, the same variables, and the same horizon as each other.

        Args:
            forecasts: One density-mode `ForecastResult` per pooled model,
                keyed by the pool's labels.
            n_draws: Size of the pooled sample. Defaults to the smallest
                member's draw count.
            seed: RNG seed (int) or Generator for the mixture draws.

        Returns:
            ForecastResult holding the pooled draws, with a `model` coordinate
            recording which member produced each draw.

        Raises:
            ValueError: If the labels, variables, horizons, or forecast mode
                do not line up with the pool.
        """
        labels = self.labels
        if set(forecasts) != set(labels):
            raise ValueError(f"combine() needs one forecast per pooled model {labels}, got {sorted(forecasts)}.")
        steps = {forecasts[label].steps for label in labels}
        if len(steps) > 1:
            raise ValueError(
                f"All forecasts must run to the same number of steps, got {sorted(steps)}; the pooled "
                "draws are a mixture over a common horizon."
            )
        for label in labels:
            forecast = forecasts[label]
            if list(forecast.var_names) != self.var_names:
                raise ValueError(
                    f"Forecast {label!r} forecasts variables {list(forecast.var_names)} but the pool "
                    f"was estimated on {self.var_names}."
                )
            if forecast.mode != "density":
                raise ValueError(
                    f"Forecast {label!r} is a mean forecast; pooling combines predictive densities, so "
                    "call forecast(include_shock_uncertainty=True) for every member."
                )
        if n_draws is not None and n_draws < 1:
            raise ValueError(f"n_draws must be at least 1, got {n_draws}.")

        rng = _resolve_rng(seed)
        stacks = [_flatten(forecasts[label]) for label in labels]
        pooled, membership = _mixture_draws(stacks, self.weights.to_numpy(dtype=float), n_draws, rng)
        return _pooled_forecast_result(pooled, membership, labels, self.var_names)

    def plot(self) -> Figure:
        """Plot the pool weights as a ranked bar chart."""
        from impulso.plotting import plot_pool_weights

        return plot_pool_weights(self)


def pool_forecasts(
    fits: Mapping[str, FittedVAR],
    holdout: VARData,
    *,
    method: WeightMethod = "stacking",
    density: DensityKind = "gaussian",
    n_draws: int | None = None,
    seed: int | np.random.Generator | None = None,
) -> PredictivePool:
    """Weight several fitted models by their held-out predictive performance.

    Every model is forecast `H = len(holdout.index)` steps from the shared
    forecast origin (the last date of the estimation sample, which all models
    must share), each forecast's predictive density is scored at the held-out
    realisations, and the resulting `(H, M)` log-score matrix is turned into
    weights. The pool forecasts the models itself rather than accepting
    `ForecastResult`s, because only a `FittedVAR` carries the estimation
    metadata that makes "same origin, genuinely held out" checkable.

    Args:
        fits: Fitted models keyed by label. At least two, all estimated on
            the same variables in the same order and to the same sample end.
        holdout: The held-out window — the `H` periods immediately after the
            forecast origin. Must carry future exogenous values if any model
            was fitted with exogenous regressors.
        method: `"stacking"` (default) maximises the log score of the *pooled*
            predictive and keeps complementary models; `"log_score"` takes a
            softmax of each model's total score and collapses onto the single
            best model as `H` grows.
        density: `"gaussian"` (default) matches a joint Gaussian to each
            horizon's forecast draws; `"diagonal"` scores each variable on its
            own marginal, which is the escape hatch when the joint covariance
            is near-singular or the draw count is small.
        n_draws: Size of the pooled predictive sample. Defaults to the
            smallest member's draw count.
        seed: RNG seed (int) or Generator. Child generators are spawned in
            `fits` insertion order, so results are reproducible but depend on
            that order.

    Returns:
        PredictivePool holding the weights, the score matrix, and a pooled
        predictive sample over the held-out window.

    Raises:
        ValueError: On fewer than two models, mismatched variables, differing
            estimation ends, a holdout that does not immediately follow the
            origin, missing exogenous values, a degenerate predictive density,
            or an unknown `method`/`density`.

    Note:
        The score is an approximation, and comparable across models rather
        than exact. Each horizon's predictive density is a Gaussian matched to
        the forecast draws, while the true posterior predictive is a
        heavier-tailed mixture over draws; the gap widens with few draws,
        stochastic volatility, and fat tails. Scores are summed over horizons
        `1..H` from one fixed origin — not a joint-path density, and not a
        rolling-origin evaluation. Weights are static, so a short holdout
        makes them noisy, and a model that only wins late in the window
        cannot be given a horizon-specific weight. Finally, `holdout` is
        checked to postdate the estimation sample, but nothing can check that
        it was not peeked at while the candidates were being chosen.
    """
    if method not in ("stacking", "log_score"):
        raise ValueError(f"method must be 'stacking' or 'log_score', got {method!r}.")
    if density not in ("gaussian", "diagonal"):
        raise ValueError(f"density must be 'gaussian' or 'diagonal', got {density!r}.")
    if n_draws is not None and n_draws < 1:
        raise ValueError(f"n_draws must be at least 1, got {n_draws}.")

    origin = _validate_pool_inputs(fits, holdout)
    labels = list(fits)
    steps = len(holdout.index)
    realised = np.asarray(holdout.endog, dtype=float)

    rng = _resolve_rng(seed)
    children = _spawn(rng, len(labels) + 1)

    stacks: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    for child, label in zip(children[:-1], labels, strict=True):
        fit = fits[label]
        exog_future = np.asarray(holdout.exog, dtype=float) if fit.has_exog else None
        forecast = fit.forecast(
            steps=steps,
            include_shock_uncertainty=True,
            seed=child,
            exog_future=exog_future,
        )
        draws = _flatten(forecast)
        stacks.append(draws)
        columns.append(_gaussian_log_scores(draws, realised, density=density, label=label))

    log_scores = pd.DataFrame(
        np.column_stack(columns),
        index=pd.DatetimeIndex(holdout.index, name="time"),
        columns=pd.Index(labels, name="model"),
    )
    matrix = log_scores.to_numpy()
    if method == "stacking":
        weight_values, converged, message = _stacking_weights(matrix, index=log_scores.index)
    else:
        weight_values, converged, message = _log_score_weights(matrix, index=log_scores.index), True, _NO_OPTIMISER

    pooled, membership = _mixture_draws(stacks, weight_values, n_draws, children[-1])
    return PredictivePool(
        weights=pd.Series(weight_values, index=pd.Index(labels, name="model"), name="weight"),
        log_scores=log_scores,
        method=method,
        density=density,
        var_names=list(holdout.endog_names),
        steps=steps,
        origin=pd.Timestamp(origin),
        holdout_predictive=_pooled_forecast_result(
            pooled, membership, labels, list(holdout.endog_names), pd.DatetimeIndex(holdout.index)
        ),
        membership=membership,
        converged=converged,
        optimiser_message=message,
    )
