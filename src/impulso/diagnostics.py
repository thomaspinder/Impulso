"""VAR-aware convergence and stability diagnostics.

`convergence_report` answers the question a generic `arviz.summary` cannot:
*is this particular VAR posterior usable?* It differs from a plain summary
table in three ways.

**Parameter blocks.** A VAR posterior mixes quantities with very different
sampling behaviour — lag coefficients, intercepts, the covariance
parameterisation, volatility latents. A single worst-R-hat over all of them
hides which part of the model is failing, so every metric is reported per
block (see `assign_blocks`) with the offending coordinate named.

**Dynamic stability.** Convergence is necessary but not sufficient: a
perfectly-mixed posterior can put mass on explosive parameter draws, whose
impulse responses diverge with the horizon and whose forecast fans are
unbounded. The report computes the companion-matrix spectral radius of every
draw and reports the explosive fraction alongside the sampling metrics.

**The VAR failure mode.** Elevated R-hat with *zero* divergences is common in
VARs and unusual elsewhere: near-collinear lag regressors give an
ill-conditioned posterior that a diagonal mass matrix explores badly, and
NUTS never has to reject a trajectory to mix poorly. The report says so
explicitly, with remedies, rather than leaving the user to conclude that no
divergences means no problem.

The report never calls `warnings.warn`. The returned object *is* the
channel: `status`, `messages`, and the per-block table carry everything, so
callers decide whether to print, raise, or ignore.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Literal, Self

import arviz as az
import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from impulso._base import ImpulsoBaseModel, ImpulsoModel
from impulso._stability import spectral_radius

if TYPE_CHECKING:
    import xarray as xr

    from impulso.protocols import VolatilityProcess

# --------------------------------------------------------------------------
# Block taxonomy
# --------------------------------------------------------------------------

# Canonical report order. Blocks with no variables are omitted entirely.
_BLOCK_ORDER: tuple[str, ...] = (
    "coefficient",
    "intercept",
    "exog",
    "covariance",
    "volatility",
    "identification",
    "other",
)

# Tier 1 of block resolution: posterior variables Impulso itself registers.
_BLOCK_MAP: dict[str, str] = {
    "B": "coefficient",
    "intercept": "intercept",
    "B_exog": "exog",
    "sigma_sd": "covariance",
    "tril_offdiag": "covariance",
    "L": "covariance",
    "Sigma": "covariance",
    "h": "volatility",
    "R_chol": "volatility",
    "R_chol_offdiag": "volatility",
    "structural_shock_matrix": "identification",
    "P": "identification",
}

# Tier 2: per-variable stochastic-volatility latents are registered with a
# `v{i}_` prefix (see `impulso.sv.spec`), so the whole family maps by pattern.
_SV_PREFIX = re.compile(r"^v\d+_")

# The only sampler statistic PyMC and nutpie agree on the name of. Every other
# stat (tree depth, acceptance rate, energy) is spelled differently by the two
# backends, so the report reads this one and nothing else.
_DIVERGING_KEY = "diverging"


def assign_blocks(
    posterior: xr.Dataset,
    volatility: VolatilityProcess | None = None,
) -> dict[str, list[str]]:
    """Group posterior variables into diagnostic blocks.

    Resolution is three-tiered, first match wins:

    1. A static map of the variable names Impulso's own estimators register.
    2. The `v{i}_` prefix carried by every per-variable stochastic-volatility
       latent, present and future.
    3. The optional `posterior_var_names()` capability on the volatility
       process, letting a custom adapter claim the names it registered.
       Claimed names join `volatility` if the adapter is time-varying and
       `covariance` otherwise.

    Anything left over lands in `other`. Unknown variables are never an
    error — a hand-built or third-party posterior still gets a report, and
    the block's variable list makes clear what was not recognised.

    Args:
        posterior: The posterior Dataset (`idata.posterior`).
        volatility: Volatility process used at fit time, consulted for the
            optional `posterior_var_names()` capability. Optional.

    Returns:
        Mapping from block name to its sorted variable names, in canonical
        block order. Blocks with no variables are absent.
    """
    claimed: dict[str, str] = {}
    hook = getattr(volatility, "posterior_var_names", None)
    if hook is not None:
        target = "volatility" if getattr(volatility, "is_time_varying", False) else "covariance"
        claimed = dict.fromkeys(hook(), target)

    grouped: dict[str, list[str]] = {}
    for raw in posterior.data_vars:
        name = str(raw)
        block = _BLOCK_MAP.get(name)
        if block is None and _SV_PREFIX.match(name):
            block = "volatility"
        if block is None:
            block = claimed.get(name, "other")
        grouped.setdefault(block, []).append(name)
    return {block: sorted(grouped[block]) for block in _BLOCK_ORDER if block in grouped}


# --------------------------------------------------------------------------
# Result objects
# --------------------------------------------------------------------------


class ConvergenceThresholds(ImpulsoModel):
    """Cut-offs separating a passing report from warnings and failures.

    Comparisons are strict, so a metric sitting exactly on a threshold
    passes: `max_rhat == 1.01` does not warn.

    Attributes:
        rhat_warn: R-hat above this warns. Default 1.01, the rank-normalised
            split-R-hat cut-off of Vehtari et al. (2021).
        rhat_fail: R-hat above this fails. Default 1.05, the classic
            Gelman-Rubin rule of thumb.
        ess_warn: Effective sample size below this warns. Default 400 —
            100 per chain at the default four chains.
        ess_fail: Effective sample size below this fails. Default 100.
        divergence_fail_rate: Divergence rate at or above which the report
            fails. Default 0.01; any divergence at all warns.
        explosive_warn: Fraction of explosive draws at or above which the
            explosive-draw message is raised from informational to a
            warning. Default 0.05. Explosive draws never fail a report.
    """

    rhat_warn: float = 1.01
    rhat_fail: float = 1.05
    ess_warn: float = 400.0
    ess_fail: float = 100.0
    divergence_fail_rate: float = 0.01
    explosive_warn: float = 0.05


class DiagnosticMessage(ImpulsoModel):
    """A single machine-readable finding.

    Attributes:
        code: Stable identifier. Prose may be reworded between releases;
            the code is the contract programmatic callers match on.
        severity: `"info"`, `"warning"`, or `"failure"`. The report's
            `status` is the worst severity present.
        message: Human-readable explanation, including remedies where
            remedies exist.
        block: Parameter block the finding concerns, or None if global.
    """

    code: str
    severity: Literal["info", "warning", "failure"]
    message: str
    block: str | None = None


class BlockDiagnostics(ImpulsoModel):
    """Convergence metrics for one parameter block.

    Every metric is the worst value over the block's coordinates, paired
    with a label naming where it occurred (`"B[y2, L1.y1]"`). Metrics are
    None when undefined for every coordinate — R-hat from a single chain,
    for instance, or a deterministic that is constant across draws.

    Attributes:
        block: Block name (see `assign_blocks`).
        var_names: Posterior variables in this block.
        n_variables: Number of posterior variables in this block.
        n_coordinates: Number of scalar coordinates across those variables.
        max_rhat: Worst rank-normalised split R-hat, or None.
        max_rhat_coord: Coordinate label attaining `max_rhat`, or None.
        min_ess_bulk: Smallest bulk effective sample size, or None.
        min_ess_bulk_coord: Coordinate label attaining `min_ess_bulk`.
        min_ess_tail: Smallest tail effective sample size, or None.
        min_ess_tail_coord: Coordinate label attaining `min_ess_tail`.
    """

    block: str
    var_names: list[str]
    n_variables: int
    n_coordinates: int
    max_rhat: float | None = None
    max_rhat_coord: str | None = None
    min_ess_bulk: float | None = None
    min_ess_bulk_coord: str | None = None
    min_ess_tail: float | None = None
    min_ess_tail_coord: str | None = None


class StabilitySummary(ImpulsoBaseModel):
    """Posterior distribution of the companion-matrix spectral radius.

    A draw is explosive when its spectral radius reaches 1: the implied
    system has no stationary solution, its impulse responses grow without
    bound in the horizon, and its forecast fan widens indefinitely. Some
    explosive mass is normal on level data under a random-walk prior mean,
    which is why it never fails a report on its own.

    Attributes:
        radius: Read-only spectral radii with shape `(chain, draw)` — after
            thinning, if `stability_draws` was used.
        p_explosive: Fraction of draws with radius >= 1.
        max_radius: Largest radius over all draws.
        n_vars: Number of endogenous variables.
        n_lags: Lag order.
        hdi_prob: Default probability mass for `hdi`, also used by
            `to_dataframe`.
        thinned_from: Original number of draws per chain when the radii were
            computed on a thinned subset, else None.
    """

    radius: np.ndarray = Field(repr=False)
    p_explosive: float
    max_radius: float
    n_vars: int
    n_lags: int
    hdi_prob: float = 0.89
    thinned_from: int | None = None

    @model_validator(mode="after")
    def _make_readonly(self) -> Self:
        radius = np.asarray(self.radius).copy()
        radius.flags.writeable = False
        object.__setattr__(self, "radius", radius)
        return self

    def median(self) -> float:
        """Posterior median spectral radius."""
        return float(np.median(self.radius))

    def hdi(self, prob: float | None = None) -> tuple[float, float]:
        """Highest-density interval of the spectral radius.

        Args:
            prob: Probability mass. Defaults to `hdi_prob` (0.89).

        Returns:
            `(lower, upper)` bounds.
        """
        # Pooled over chains: the interval is a statement about the posterior,
        # not about any one chain, and a flat array sidesteps ArviZ's pending
        # reinterpretation of 2-D input as (chain, draw).
        pooled = np.asarray(self.radius).reshape(-1)
        bounds = az.hdi(pooled, hdi_prob=self.hdi_prob if prob is None else prob)
        return float(bounds[0]), float(bounds[1])

    def to_dataframe(self) -> pd.DataFrame:
        """Single-row frame of the stability summary."""
        lower, upper = self.hdi()
        return pd.DataFrame(
            [
                {
                    "median_radius": self.median(),
                    "hdi_lower": lower,
                    "hdi_upper": upper,
                    "max_radius": self.max_radius,
                    "p_explosive": self.p_explosive,
                    "n_vars": self.n_vars,
                    "n_lags": self.n_lags,
                }
            ],
            index=pd.Index(["stability"], name="quantity"),
        )


class ConvergenceReport(ImpulsoBaseModel):
    """VAR-aware convergence and stability diagnostics for one posterior.

    Produced by `convergence_report`, or by the delegating
    `FittedVAR.convergence_report` / `IdentifiedVAR.convergence_report`.

    Attributes:
        blocks: Per-block metrics in canonical order.
        stability: Spectral-radius summary over the posterior draws.
        divergences: Number of divergent transitions, or None when the
            sampler recorded no statistics.
        n_transitions: Total post-warmup transitions, or None.
        divergence_rate: `divergences / n_transitions`, or None.
        sampler_stats_available: Whether divergence statistics were found.
        n_chains: Number of chains in the posterior.
        n_draws: Number of post-warmup draws per chain.
        thresholds: Thresholds used to derive `status`.
        messages: Findings, each with a stable `code`.
        status: `"passed"`, `"warnings"`, or `"failed"` — the worst message
            severity. `"failed"` is reserved for sampler pathology.
    """

    blocks: list[BlockDiagnostics]
    stability: StabilitySummary
    divergences: int | None
    n_transitions: int | None
    divergence_rate: float | None
    sampler_stats_available: bool
    n_chains: int
    n_draws: int
    thresholds: ConvergenceThresholds
    messages: list[DiagnosticMessage]
    status: Literal["passed", "warnings", "failed"]

    @property
    def max_rhat(self) -> float | None:
        """Worst R-hat across all blocks, or None if undefined everywhere."""
        return _extreme([block.max_rhat for block in self.blocks], "max")

    @property
    def min_ess_bulk(self) -> float | None:
        """Smallest bulk effective sample size across all blocks."""
        return _extreme([block.min_ess_bulk for block in self.blocks], "min")

    @property
    def min_ess_tail(self) -> float | None:
        """Smallest tail effective sample size across all blocks."""
        return _extreme([block.min_ess_tail for block in self.blocks], "min")

    def to_dataframe(self) -> pd.DataFrame:
        """Per-block metric table, indexed by block in canonical order."""
        rows = [
            {
                "n_variables": block.n_variables,
                "n_coordinates": block.n_coordinates,
                "max_rhat": block.max_rhat,
                "max_rhat_coord": block.max_rhat_coord,
                "min_ess_bulk": block.min_ess_bulk,
                "min_ess_bulk_coord": block.min_ess_bulk_coord,
                "min_ess_tail": block.min_ess_tail,
                "min_ess_tail_coord": block.min_ess_tail_coord,
            }
            for block in self.blocks
        ]
        return pd.DataFrame(rows, index=pd.Index([block.block for block in self.blocks], name="block"))

    def summary(self) -> str:
        """Multi-line human-readable rendering of the whole report."""
        divergences = "unavailable" if self.divergences is None else str(self.divergences)
        header = [
            f"Convergence report: {self.status.upper()}",
            f"  {self.n_chains} chains x {self.n_draws} draws | divergences: {divergences}",
        ]
        table = self.to_dataframe()[["max_rhat", "min_ess_bulk", "min_ess_tail", "max_rhat_coord"]]
        lower, upper = self.stability.hdi()
        stability = (
            f"Stability: median spectral radius {self.stability.median():.3f} "
            f"[{lower:.3f}, {upper:.3f}] | max {self.stability.max_radius:.3f} | "
            f"explosive draws {self.stability.p_explosive:.1%}"
        )
        lines = [*header, "", table.to_string(), "", stability]
        if self.messages:
            lines += ["", "Messages:"]
            lines += [f"  [{msg.severity}] {msg.code}: {msg.message}" for msg in self.messages]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """One-line status plus headline numbers."""
        rhat = "n/a" if self.max_rhat is None else f"{self.max_rhat:.3f}"
        ess = "n/a" if self.min_ess_bulk is None else f"{self.min_ess_bulk:.0f}"
        divergences = "n/a" if self.divergences is None else str(self.divergences)
        return (
            f"ConvergenceReport(status={self.status!r}, max_rhat={rhat}, "
            f"min_ess_bulk={ess}, divergences={divergences}, "
            f"p_explosive={self.stability.p_explosive:.3f})"
        )


# --------------------------------------------------------------------------
# Metric computation
# --------------------------------------------------------------------------


def _extreme(values: list[float | None], mode: Literal["max", "min"]) -> float | None:
    """Worst of the non-None entries, or None when every entry is None."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present) if mode == "max" else min(present)


def _fallback_coords(var_names: list[str] | None, n_lags: int) -> dict[str, list[str]]:
    """Labels for dimensions a hand-built or conjugate posterior leaves bare.

    `VAR.fit` stamps coords on the posterior, so its labels are read
    directly. `ConjugateVAR` builds its Dataset from dims alone, and this
    supplies the same names so both estimators produce identical coordinate
    labels for the same model.
    """
    if not var_names:
        return {}
    names = list(var_names)
    coeff = [f"L{lag}.{name}" for lag in range(1, n_lags + 1) for name in names]
    return {"var": names, "var1": names, "var2": names, "variable": names, "response": names, "coeff": coeff}


def _worst_in_variable(
    name: str,
    da: xr.DataArray,
    mode: Literal["max", "min"],
    fallback: dict[str, list[str]],
) -> tuple[float, str] | None:
    """Extreme value and its coordinate label within one variable's metrics.

    Returns None when the metric is NaN at every coordinate — a single-chain
    R-hat, or a deterministic that never varies across draws.
    """
    values = np.asarray(da.values, dtype=float)
    if values.ndim == 0:
        scalar = float(values)
        return None if np.isnan(scalar) else (scalar, name)
    flat = values.reshape(-1)
    if bool(np.all(np.isnan(flat))):
        return None
    index = int(np.nanargmax(flat)) if mode == "max" else int(np.nanargmin(flat))
    position = np.unravel_index(index, values.shape)
    labels = []
    for axis, (dim, i) in enumerate(zip(da.dims, position, strict=True)):
        if dim in da.coords:
            labels.append(str(da.coords[dim].values[i]))
        elif dim in fallback and len(fallback[dim]) == values.shape[axis]:
            labels.append(fallback[dim][i])
        else:
            labels.append(str(i))
    return float(flat[index]), f"{name}[{', '.join(labels)}]"


def _worst_across(
    metrics: xr.Dataset,
    names: list[str],
    mode: Literal["max", "min"],
    fallback: dict[str, list[str]],
) -> tuple[float | None, str | None]:
    """Extreme metric value and label across every variable in a block."""
    best: tuple[float, str] | None = None
    for name in names:
        found = _worst_in_variable(name, metrics[name], mode, fallback)
        if found is None:
            continue
        if best is None or (found[0] > best[0] if mode == "max" else found[0] < best[0]):
            best = found
    return best if best is not None else (None, None)


def _block_metrics(
    posterior: xr.Dataset,
    block: str,
    names: list[str],
    fallback: dict[str, list[str]],
) -> BlockDiagnostics:
    """Compute R-hat and both effective sample sizes for one block."""
    subset = posterior[names]
    # A deterministic that is constant across draws divides by a zero
    # variance inside ArviZ and yields NaN, which is the honest answer and is
    # handled downstream. Silence the numpy notice so a report never emits a
    # warning of its own.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rhat = az.rhat(subset)
        ess_bulk = az.ess(subset, method="bulk")
        ess_tail = az.ess(subset, method="tail")

    max_rhat, max_rhat_coord = _worst_across(rhat, names, "max", fallback)
    min_bulk, min_bulk_coord = _worst_across(ess_bulk, names, "min", fallback)
    min_tail, min_tail_coord = _worst_across(ess_tail, names, "min", fallback)
    n_coordinates = sum(int(np.prod(rhat[name].shape, dtype=int)) for name in names)
    return BlockDiagnostics(
        block=block,
        var_names=list(names),
        n_variables=len(names),
        n_coordinates=n_coordinates,
        max_rhat=max_rhat,
        max_rhat_coord=max_rhat_coord,
        min_ess_bulk=min_bulk,
        min_ess_bulk_coord=min_bulk_coord,
        min_ess_tail=min_tail,
        min_ess_tail_coord=min_tail_coord,
    )


def _divergences(idata: az.InferenceData) -> tuple[int | None, int | None, float | None, bool]:
    """Global divergence counts read from `sample_stats` only.

    Divergences are a property of a *trajectory*, not of any one parameter,
    so they are never attributed to a block. Only the post-warmup group is
    read: nutpie also emits `warmup_sample_stats`, whose divergences belong
    to adaptation and say nothing about the retained draws.
    """
    if "sample_stats" not in idata.groups() or _DIVERGING_KEY not in idata.sample_stats:
        return None, None, None, False
    diverging = np.asarray(idata.sample_stats[_DIVERGING_KEY].values)
    count = int(diverging.sum())
    total = int(diverging.size)
    return count, total, (count / total if total else 0.0), True


# --------------------------------------------------------------------------
# Messages and status
# --------------------------------------------------------------------------

_NUTPIE_REMEDY = 'NUTSSampler(nuts_sampler="nutpie", nuts_sampler_kwargs={"low_rank_modified_mass_matrix": True})'


def _rhat_messages(
    report_max_rhat: float | None,
    coord: str | None,
    divergences: int | None,
    stats_available: bool,
    thresholds: ConvergenceThresholds,
) -> list[DiagnosticMessage]:
    """R-hat findings, including the VAR-specific zero-divergence case."""
    if report_max_rhat is None or report_max_rhat <= thresholds.rhat_warn:
        return []
    where = f" (worst: {coord})" if coord else ""
    if report_max_rhat > thresholds.rhat_fail:
        messages = [
            DiagnosticMessage(
                code="rhat_high",
                severity="failure",
                message=(
                    f"R-hat reaches {report_max_rhat:.3f}{where}, above the failure "
                    f"threshold of {thresholds.rhat_fail}. The chains have not mixed; "
                    "the posterior draws do not describe a single distribution and no "
                    "downstream quantity should be reported."
                ),
            )
        ]
    else:
        messages = [
            DiagnosticMessage(
                code="rhat_elevated",
                severity="warning",
                message=(
                    f"R-hat reaches {report_max_rhat:.3f}{where}, above the warning "
                    f"threshold of {thresholds.rhat_warn}. Mixing is imperfect; run "
                    "longer chains before trusting tail quantities."
                ),
            )
        ]
    if stats_available and divergences == 0:
        messages.append(
            DiagnosticMessage(
                code="rhat_without_divergences",
                severity="warning",
                message=(
                    f"R-hat up to {report_max_rhat:.3f} with zero divergences is common in "
                    "VARs: near-collinear lag regressors make the posterior ill-conditioned "
                    "and diagonal mass-matrix adaptation mixes poorly across it, without the "
                    "sampler ever having to reject a trajectory. Divergences are not a "
                    "reliable alarm here. Remedies, in order: switch to nutpie's low-rank "
                    f"mass matrix — {_NUTPIE_REMEDY} — then lengthen `tune`, then tighten the "
                    "Minnesota prior (smaller `tightness`), then reduce the number of "
                    "variables or lags."
                ),
            )
        )
    return messages


def _ess_messages(
    metric: float | None,
    coord: str | None,
    kind: str,
    thresholds: ConvergenceThresholds,
) -> list[DiagnosticMessage]:
    """Effective-sample-size findings for one flavour of ESS."""
    if metric is None or metric >= thresholds.ess_warn:
        return []
    where = f" (worst: {coord})" if coord else ""
    if metric < thresholds.ess_fail:
        return [
            DiagnosticMessage(
                code=f"ess_{kind}_low",
                severity="failure",
                message=(
                    f"{kind.capitalize()} effective sample size falls to {metric:.0f}{where}, "
                    f"below the failure threshold of {thresholds.ess_fail:.0f}. Posterior "
                    "summaries carry more Monte Carlo error than signal."
                ),
            )
        ]
    return [
        DiagnosticMessage(
            code=f"ess_{kind}_marginal",
            severity="warning",
            message=(
                f"{kind.capitalize()} effective sample size falls to {metric:.0f}{where}, "
                f"below the warning threshold of {thresholds.ess_warn:.0f}. Draw more "
                "samples before reporting intervals from this block."
            ),
        )
    ]


def _divergence_messages(
    divergences: int | None,
    n_transitions: int | None,
    rate: float | None,
    stats_available: bool,
    thresholds: ConvergenceThresholds,
) -> list[DiagnosticMessage]:
    """Divergence findings, or the informational note when none were recorded."""
    if not stats_available:
        return [
            DiagnosticMessage(
                code="sampler_stats_missing",
                severity="info",
                message=(
                    "No sampler statistics were found, so divergences could not be counted. "
                    "This is expected for `ConjugateVAR`, which draws coefficients in closed "
                    "form and has no trajectories to diverge, and for hand-built posteriors. "
                    "It does not by itself indicate a problem."
                ),
            )
        ]
    if not divergences:
        return []
    # `stats_available` guarantees both counts are present; default defensively
    # so the message code path never depends on a nullable arithmetic operand.
    observed_rate = rate if rate is not None else 0.0
    total = n_transitions if n_transitions is not None else 0
    severity = "failure" if observed_rate >= thresholds.divergence_fail_rate else "warning"
    return [
        DiagnosticMessage(
            code="divergences_present",
            severity=severity,
            message=(
                f"{divergences} of {total} transitions diverged ({observed_rate:.2%}). "
                "Divergent trajectories mean the sampler could not follow the posterior's "
                "geometry, so the draws are biased toward the regions it could reach. "
                "Raise `target_accept` toward 0.95, lengthen `tune`, or tighten the prior."
            ),
        )
    ]


def _chain_messages(n_chains: int) -> list[DiagnosticMessage]:
    """The single-chain note: R-hat is a between-chain statistic."""
    if n_chains >= 2:
        return []
    return [
        DiagnosticMessage(
            code="single_chain",
            severity="warning",
            message=(
                "Only one chain is present, so R-hat is undefined and is reported as None "
                "for every block; effective sample size is still computed. This is the "
                "normal shape of a `ConjugateVAR` posterior, where the coefficient and "
                "Cholesky draws are exact conditional draws (their effective sample size "
                "is nominal by construction) and only the hyperparameters — sampled by "
                "random-walk Metropolis — carry meaningful autocorrelation. For a NUTS "
                "fit, sample at least two chains before trusting any convergence claim."
            ),
        )
    ]


def _stability_messages(
    stability: StabilitySummary,
    thresholds: ConvergenceThresholds,
) -> list[DiagnosticMessage]:
    """The explosive-draw finding. Never a failure — see ADR-0008."""
    if stability.p_explosive <= 0:
        return []
    severity = "warning" if stability.p_explosive >= thresholds.explosive_warn else "info"
    return [
        DiagnosticMessage(
            code="explosive_draws",
            severity=severity,
            block="coefficient",
            message=(
                f"{stability.p_explosive:.1%} of draws are explosive (companion-matrix "
                f"spectral radius >= 1; largest {stability.max_radius:.3f}). Their impulse "
                "responses diverge with the horizon, their forecast fans are unbounded, "
                "their long-horizon FEVD shares are uninterpretable, and their historical "
                "decomposition baselines drift. This is a property of the model, not of the "
                "sampler, and near-unit-root mass is legitimate on level data under a "
                "random-walk prior mean. If it is not intended: difference the data (or "
                "test for cointegration), tighten the shrinkage, or restrict reported "
                "horizons to where the responses are still meaningful."
            ),
        )
    ]


def _derive_status(messages: list[DiagnosticMessage]) -> Literal["passed", "warnings", "failed"]:
    """Worst severity present. Only sampler pathology reaches `"failed"`."""
    severities = {message.severity for message in messages}
    if "failure" in severities:
        return "failed"
    if "warning" in severities:
        return "warnings"
    return "passed"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _stability_summary(
    posterior: xr.Dataset,
    n_lags: int,
    hdi_prob: float,
    stability_draws: int | None,
) -> StabilitySummary:
    """Spectral-radius summary over the posterior's lag coefficients."""
    B_da = posterior["B"]
    if set(B_da.dims) == {"chain", "draw", "var", "coeff"}:
        B_da = B_da.transpose("chain", "draw", "var", "coeff")
    B = np.asarray(B_da.values, dtype=float)

    thinned_from: int | None = None
    if stability_draws is not None:
        if stability_draws < 1:
            raise ValueError(f"stability_draws must be positive, got {stability_draws}")
        n_draws = B.shape[1]
        if stability_draws < n_draws:
            # Deterministic stride, never an RNG: two calls on one posterior
            # must return identical numbers.
            stride = -(-n_draws // stability_draws)
            B = B[:, ::stride]
            thinned_from = n_draws

    radius = spectral_radius(B, n_lags)
    return StabilitySummary(
        radius=radius,
        p_explosive=float(np.mean(radius >= 1.0)),
        max_radius=float(np.max(radius)),
        n_vars=B.shape[-2],
        n_lags=n_lags,
        hdi_prob=hdi_prob,
        thinned_from=thinned_from,
    )


def convergence_report(
    idata: az.InferenceData,
    *,
    n_lags: int,
    var_names: list[str] | None = None,
    volatility: VolatilityProcess | None = None,
    thresholds: ConvergenceThresholds | None = None,
    hdi_prob: float = 0.89,
    stability_draws: int | None = None,
) -> ConvergenceReport:
    """Build a VAR-aware convergence and stability report for a posterior.

    Reports R-hat and both effective sample sizes per parameter block, each
    with the coordinate that attains the worst value; the global divergence
    count; and the posterior distribution of the companion-matrix spectral
    radius. See the module docstring for why a VAR needs its own report.

    `"failed"` is reserved for sampler pathology — R-hat above
    `rhat_fail`, effective sample size below `ess_fail`, or a divergence
    rate at or above `divergence_fail_rate`. Explosive draws warn but never
    fail: mass near a unit root is a legitimate posterior statement about
    level data, not evidence that the sampler misbehaved.

    Cost is dominated by the eigendecomposition of one `(n * p, n * p)`
    companion matrix per draw. Pass `stability_draws` to compute the radii
    on a deterministically strided subset when the posterior is large.

    Args:
        idata: Posterior to diagnose. Must carry a `posterior` group with
            reduced-form lag coefficients `B`; `sample_stats` is read when
            present and its absence is reported, not an error.
        n_lags: Lag order, needed to build the companion matrix.
        var_names: Endogenous variable names, used to label coordinates on
            posteriors that carry no coords of their own (`ConjugateVAR`
            builds one such). Optional.
        volatility: Volatility process used at fit time. Consulted for the
            optional `posterior_var_names()` capability when assigning
            blocks. Optional.
        thresholds: Cut-offs driving `status`. Defaults to
            `ConvergenceThresholds()`.
        hdi_prob: Default probability mass for the spectral radius interval.
        stability_draws: Approximate number of draws per chain to retain for
            the spectral-radius computation. Thinning is a deterministic
            stride, so repeated calls agree exactly.

    Returns:
        A `ConvergenceReport`. Nothing is warned or raised on a bad
        posterior — the report is the channel.

    Raises:
        ValueError: If `idata` has no `posterior` group, if the posterior
            carries no `B` (a univariate `FittedSV` posterior has no lag
            coefficients and is not supported), or if `stability_draws` is
            not positive.
    """
    if "posterior" not in idata.groups():
        raise ValueError("convergence_report requires an InferenceData with a `posterior` group.")
    posterior = idata.posterior
    if "B" not in posterior:
        raise ValueError(
            "convergence_report requires reduced-form VAR lag coefficients `B` in the "
            "posterior, and this one has none. A univariate stochastic-volatility fit "
            "(`FittedSV`) has no lag coefficients and therefore no companion matrix, so "
            "it is not supported; diagnose it with ArviZ directly."
        )

    thresholds = thresholds or ConvergenceThresholds()
    fallback = _fallback_coords(var_names, n_lags)
    blocks = [
        _block_metrics(posterior, block, names, fallback)
        for block, names in assign_blocks(posterior, volatility).items()
    ]
    stability = _stability_summary(posterior, n_lags, hdi_prob, stability_draws)
    divergences, n_transitions, rate, stats_available = _divergences(idata)

    max_rhat = _extreme([block.max_rhat for block in blocks], "max")
    rhat_coord = next((block.max_rhat_coord for block in blocks if block.max_rhat == max_rhat), None)
    min_bulk = _extreme([block.min_ess_bulk for block in blocks], "min")
    bulk_coord = next((block.min_ess_bulk_coord for block in blocks if block.min_ess_bulk == min_bulk), None)
    min_tail = _extreme([block.min_ess_tail for block in blocks], "min")
    tail_coord = next((block.min_ess_tail_coord for block in blocks if block.min_ess_tail == min_tail), None)

    n_chains = int(posterior.sizes["chain"])
    messages = [
        *_rhat_messages(max_rhat, rhat_coord, divergences, stats_available, thresholds),
        *_ess_messages(min_bulk, bulk_coord, "bulk", thresholds),
        *_ess_messages(min_tail, tail_coord, "tail", thresholds),
        *_divergence_messages(divergences, n_transitions, rate, stats_available, thresholds),
        *_chain_messages(n_chains),
        *_stability_messages(stability, thresholds),
    ]

    return ConvergenceReport(
        blocks=blocks,
        stability=stability,
        divergences=divergences,
        n_transitions=n_transitions,
        divergence_rate=rate,
        sampler_stats_available=stats_available,
        n_chains=n_chains,
        n_draws=int(posterior.sizes["draw"]),
        thresholds=thresholds,
        messages=messages,
        status=_derive_status(messages),
    )
