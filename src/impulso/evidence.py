"""Model evidence for the conjugate VAR, and Bayes-factor comparison across fits.

The conjugate (Normal-Inverse-Wishart) VAR has a closed-form marginal likelihood, so
`ConjugateVAR.fit` can report the log evidence of the model it just estimated at no extra
cost. `ModelEvidence` is the value plus the metadata needed to decide whether two such
values are comparable; `compare_evidence` does the comparing and returns an
`EvidenceComparison` carrying Bayes factors and posterior model probabilities.

All arithmetic happens in log space. Bayes factors are exponentiated only at the last
step, with an explicit clamp, so an overwhelming log difference reports `inf` or `0.0`
rather than raising or emitting an overflow warning. The log and log10 columns are always
published alongside the ratio, and are the numbers to quote (Kass and Raftery 1995).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from impulso._base import ImpulsoBaseModel

if TYPE_CHECKING:
    from impulso.fitted import FittedVAR

# math.exp overflows above log(finfo(float).max) ~= 709.78; clamp just below.
_MAX_LOG_BAYES_FACTOR: float = 709.0


def _response_digest(endog: np.ndarray, var_names: Sequence[str], n_lags: int) -> str:
    """Byte-exact digest of the response block a VAR(`n_lags`) actually fits.

    The first `n_lags` rows are the presample the model conditions on, so they are
    excluded; columns are reordered by sorted variable name so that two fits which order
    the same variables differently still hash alike.

    Args:
        endog: Endogenous data of shape `(T_full, n_vars)`, before lag trimming.
        var_names: Column names of `endog`, in column order.
        n_lags: Lag order `p`; the first `p` rows are dropped.

    Returns:
        Hex SHA-256 digest of the trimmed, column-sorted response block.
    """
    order = sorted(range(len(var_names)), key=lambda i: var_names[i])
    block = np.ascontiguousarray(np.asarray(endog, dtype=np.float64)[n_lags:, order])
    return hashlib.sha256(block.tobytes()).hexdigest()


class ModelEvidence(ImpulsoBaseModel):
    """Closed-form log marginal likelihood of a conjugate VAR, with comparability metadata.

    The value is the log density of the observed response block given the presample and
    the hyperparameters,

    `log p(y_{p+1:T} | y_{1:p}, hyperparameters, model)`,

    with three properties worth stating explicitly:

    * It is a density over the **observed** data. When a deterministic volatility break
      rescales the sample, the change-of-variables Jacobian is included, so a break model
      and a homoscedastic model fitted to the same observations are directly comparable.
    * It is **conditional on the hyperparameters** in `hyperparameters`. With
      `NIWPrior(select=True)` those were chosen by maximising evidence times hyperprior,
      so a ratio of two such values is an empirical-Bayes (conditional) Bayes factor, not
      a fully marginal one. When `hyperparameters` is empty — a fixed prior with no
      volatility break — nothing was selected and the value is the full marginal
      likelihood of the model.
    * It **conditions on the presample**, so models with different lag orders condition
      on different initial rows. Fixing the response window and conditioning on initial
      conditions is the standard practice (Giannone, Lenza and Primiceri 2015), but the
      resulting Bayes factor is conditional on those initial conditions.

    Attributes:
        log_marginal_likelihood: The log evidence. Must be finite.
        n_obs: Response rows after lag trimming, `T_full - n_lags`.
        n_vars: Number of endogenous variables.
        var_names: Endogenous variable names, in column order.
        n_lags: Lag order `p`.
        volatility: Name of the deterministic volatility break, or None if homoscedastic.
        hyperparameters: Hyperparameters the evidence was evaluated at (empty when the
            prior is fixed and there is no volatility break).
        sample_start: Timestamp of the first response row.
        sample_end: Timestamp of the last response row.
        sample_digest: Digest of the response block (see `_response_digest`).
    """

    log_marginal_likelihood: float
    n_obs: int = Field(ge=1)
    n_vars: int = Field(ge=1)
    var_names: list[str]
    n_lags: int = Field(ge=1)
    volatility: str | None = None
    hyperparameters: dict[str, float] = Field(default_factory=dict)
    sample_start: pd.Timestamp = Field(repr=False)
    sample_end: pd.Timestamp = Field(repr=False)
    sample_digest: str = Field(repr=False)

    @field_validator("log_marginal_likelihood")
    @classmethod
    def _require_finite(cls, value: float) -> float:
        """Reject a non-finite evidence loudly at fit time rather than at comparison time."""
        if not math.isfinite(value):
            raise ValueError(
                f"the conjugate marginal likelihood evaluated to {value}; the hyperparameter "
                "mode fell outside the model's support (a degenerate design or an unstable "
                "volatility path)"
            )
        return value


def _safe_exp(delta: float) -> float:
    """Exponentiate a log difference, saturating at `inf` / `0.0` instead of overflowing."""
    if delta > _MAX_LOG_BAYES_FACTOR:
        return math.inf
    if delta < -_MAX_LOG_BAYES_FACTOR:
        return 0.0
    return math.exp(delta)


class EvidenceComparison(ImpulsoBaseModel):
    """Bayes factors and posterior model probabilities across named model evidences.

    Built by `compare_evidence`. The `reference` model is the denominator of every Bayes
    factor unless a different one is named per call.

    Attributes:
        evidences: Named `ModelEvidence` objects, in the order they were compared.
        reference: Key of the default denominator model.
    """

    evidences: dict[str, ModelEvidence]
    reference: str

    @model_validator(mode="after")
    def _validate_reference(self) -> EvidenceComparison:
        """Require at least two models and a reference that names one of them."""
        if len(self.evidences) < 2:
            raise ValueError(f"an evidence comparison needs at least two models, got {len(self.evidences)}")
        if self.reference not in self.evidences:
            raise ValueError(
                f"reference {self.reference!r} is not one of the compared models ({', '.join(sorted(self.evidences))})"
            )
        return self

    @property
    def best(self) -> str:
        """Name of the model with the highest log marginal likelihood."""
        return max(self.evidences, key=lambda name: self.evidences[name].log_marginal_likelihood)

    def _lookup(self, name: str) -> ModelEvidence:
        """Fetch one evidence by name, naming the alternatives when it is missing."""
        if name not in self.evidences:
            raise KeyError(f"{name!r} is not one of the compared models ({', '.join(sorted(self.evidences))})")
        return self.evidences[name]

    def log_bayes_factor(self, model: str, against: str | None = None) -> float:
        """Log Bayes factor of `model` against another model.

        Args:
            model: Name of the numerator model.
            against: Name of the denominator model. Defaults to `reference`.

        Returns:
            `log ML(model) - log ML(against)`. Positive favours `model`.

        Raises:
            KeyError: If either name is not part of the comparison.
        """
        denominator = self.reference if against is None else against
        return self._lookup(model).log_marginal_likelihood - self._lookup(denominator).log_marginal_likelihood

    def log10_bayes_factor(self, model: str, against: str | None = None) -> float:
        """Base-10 log Bayes factor — the unit Kass and Raftery (1995) tabulate.

        Args:
            model: Name of the numerator model.
            against: Name of the denominator model. Defaults to `reference`.

        Returns:
            `log10 BF(model, against)`.
        """
        return self.log_bayes_factor(model, against) / math.log(10.0)

    def bayes_factor(self, model: str, against: str | None = None) -> float:
        """Bayes factor of `model` against another model.

        Saturates rather than overflowing: a log difference beyond about 709 returns
        `inf` (or `0.0` below `-709`). Quote `log_bayes_factor` or
        `log10_bayes_factor` when the ratio is that large.

        Args:
            model: Name of the numerator model.
            against: Name of the denominator model. Defaults to `reference`.

        Returns:
            `BF(model, against)`.
        """
        return _safe_exp(self.log_bayes_factor(model, against))

    def posterior_probabilities(self) -> dict[str, float]:
        """Posterior model probabilities under equal prior model weights.

        Computed by a max-shifted softmax over the log marginal likelihoods, so the
        result is stable however far apart the models are.

        Returns:
            Model name to posterior probability; the values sum to one.
        """
        names = list(self.evidences)
        lml = np.array([self.evidences[name].log_marginal_likelihood for name in names], dtype=float)
        weights = np.exp(lml - lml.max())
        probs = weights / weights.sum()
        return dict(zip(names, (float(p) for p in probs), strict=True))

    def to_dataframe(self, reference: str | None = None) -> pd.DataFrame:
        """Tabulate the comparison, one row per model in comparison order.

        Args:
            reference: Denominator for the Bayes-factor columns. Defaults to `reference`.

        Returns:
            DataFrame indexed by model name (`index.name` is `"model"`) with columns
            `log_marginal_likelihood`, `log_bayes_factor`, `log10_bayes_factor`,
            `bayes_factor`, `posterior_probability`, `n_obs`, `n_vars`, `n_lags` and
            `volatility`. The reference row has a log Bayes factor of exactly 0.0.
            `volatility` is object dtype so a homoscedastic model reads as `None`
            rather than as a missing value.
        """
        denominator = self.reference if reference is None else reference
        self._lookup(denominator)
        probs = self.posterior_probabilities()
        rows = {
            name: {
                "log_marginal_likelihood": ev.log_marginal_likelihood,
                "log_bayes_factor": self.log_bayes_factor(name, denominator),
                "log10_bayes_factor": self.log10_bayes_factor(name, denominator),
                "bayes_factor": self.bayes_factor(name, denominator),
                "posterior_probability": probs[name],
                "n_obs": ev.n_obs,
                "n_vars": ev.n_vars,
                "n_lags": ev.n_lags,
            }
            for name, ev in self.evidences.items()
        }
        frame = pd.DataFrame.from_dict(rows, orient="index")
        # Object dtype, not the inferred string dtype: pandas would otherwise coerce the
        # None of a homoscedastic model into a missing value.
        frame["volatility"] = pd.Series({name: ev.volatility for name, ev in self.evidences.items()}, dtype=object)
        frame.index.name = "model"
        return frame


def _as_evidence(name: str, obj: Any) -> ModelEvidence:
    """Coerce a fitted VAR or a bare `ModelEvidence` into a `ModelEvidence`."""
    if isinstance(obj, ModelEvidence):
        return obj
    evidence = getattr(obj, "evidence", None)
    if isinstance(evidence, ModelEvidence):
        return evidence
    if evidence is None and hasattr(obj, "idata"):
        raise ValueError(
            f"{name!r} carries no model evidence. Only ConjugateVAR fits have a closed-form "
            "marginal likelihood; the PyMC/NUTS estimator (impulso.VAR) does not. Refit with "
            "impulso.ConjugateVAR to compare evidence."
        )
    raise TypeError(f"{name!r} is a {type(obj).__name__}; compare_evidence takes fitted VARs or ModelEvidence objects.")


def _check_compatible(name: str, evidence: ModelEvidence, ref_name: str, reference: ModelEvidence) -> None:
    """Reject an evidence that cannot be placed in a Bayes factor against `reference`."""
    if sorted(evidence.var_names) != sorted(reference.var_names):
        raise ValueError(
            f"{name!r} was fitted on variables {sorted(evidence.var_names)} but {ref_name!r} on "
            f"{sorted(reference.var_names)}. Model evidence is only comparable across the same "
            "variables — the marginal likelihood's dimension constants depend on the number of "
            "variables. (Variable order does not matter; the variable set does.)"
        )
    if evidence.n_obs != reference.n_obs:
        raise ValueError(
            f"{name!r} was fitted on {evidence.n_obs} observations but {ref_name!r} on "
            f"{reference.n_obs}. A marginal likelihood is a density over the observations it "
            "conditions on, so a Bayes factor across different effective samples is undefined. "
            "A VAR(p) trims p initial rows, so lag orders must be aligned before comparison — "
            "fit the lower-order model on data already trimmed by the largest lag order."
        )
    if evidence.sample_start != reference.sample_start or evidence.sample_end != reference.sample_end:
        raise ValueError(
            f"{name!r} and {ref_name!r} both use {evidence.n_obs} observations but over different "
            f"windows ({evidence.sample_start.date()}..{evidence.sample_end.date()} vs "
            f"{reference.sample_start.date()}..{reference.sample_end.date()})."
        )
    if evidence.sample_digest != reference.sample_digest:
        raise ValueError(
            f"{name!r} and {ref_name!r} cover the same window and variables but not the same "
            "observations (sample digest mismatch). Evidence is not comparable across "
            "transformations of the data — levels vs logs vs differences, rescaling, or a data "
            "revision. Refit both models on identical data."
        )


def compare_evidence(**fits: FittedVAR | ModelEvidence) -> EvidenceComparison:
    """Compare conjugate-VAR fits by marginal likelihood.

    Each keyword names a model, so the labels flow through to the Bayes-factor table:

    ```python
    comparison = impulso.compare_evidence(baseline=fit_p1, with_break=fit_p2)
    comparison.to_dataframe()
    ```

    The first fit passed becomes the reference (the denominator of every Bayes factor);
    override it per call with `bayes_factor(name, against=...)` or
    `to_dataframe(reference=...)`.

    Every fit must be comparable to the first: the same variable set (order is
    irrelevant), the same number of response observations, the same sample window and
    byte-identical response data. What is *allowed* to differ is exactly what is being
    compared — lag order, volatility break, prior settings and selected hyperparameters.
    Note that a VAR(p) drops p initial rows, so comparing lag orders means aligning the
    response windows by hand (fit the shorter-lag model on data already trimmed by the
    largest lag order).

    Three caveats carry over from `ModelEvidence`: the evidence conditions on the
    presample, it conditions on the selected hyperparameters (making the ratio an
    empirical-Bayes Bayes factor when `NIWPrior(select=True)` was used), and it includes
    the volatility-rescaling Jacobian so break and no-break models are comparable.

    Args:
        **fits: Two or more `FittedVAR` objects from `ConjugateVAR.fit` (or bare
            `ModelEvidence` objects), keyed by the label to report them under.

    Returns:
        An `EvidenceComparison` over the named evidences.

    Raises:
        ValueError: If fewer than two fits are given, if a fit carries no evidence
            (the PyMC/NUTS path), or if two fits are not comparable.
        TypeError: If an argument is neither a fitted VAR nor a `ModelEvidence`.
    """
    if len(fits) < 2:
        raise ValueError(
            f"compare_evidence needs at least two named fits, got {len(fits)}. Call it with "
            "keyword arguments, e.g. compare_evidence(baseline=fit_a, with_break=fit_b)."
        )
    evidences = {name: _as_evidence(name, obj) for name, obj in fits.items()}
    ref_name = next(iter(evidences))
    reference = evidences[ref_name]
    for name, evidence in evidences.items():
        if name != ref_name:
            _check_compatible(name, evidence, ref_name, reference)
    return EvidenceComparison(evidences=evidences, reference=ref_name)
