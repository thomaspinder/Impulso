"""Granger-causality strength read off a fitted reduced-form posterior.

`granger_causality(fitted, ...)` backs `FittedVAR.granger_causality`,
reading the tested lag coefficients straight out of an already-fitted
posterior. It works under any volatility process, because the coefficient
matrix `B` is time-invariant under all of them.

It reports no probability of *no* causality — see the
`GrangerCausalityResult` docstring for why that quantity does not exist
under continuous coefficient priors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from impulso.fitted import FittedVAR
    from impulso.results import GrangerCausalityResult, IntegrationOrderResult


def _coefficient_indices(n_vars: int, cause_index: int, n_lags_tested: int) -> list[int]:
    """Columns of `B` holding one cause's coefficients, lag 1 first.

    `VAR.fit` and `ConjugateVAR.fit` both stack the regressors lag-major —
    `X = [y_{t-1}, ..., y_{t-p}]`, each block holding all `n` variables —
    and the posterior's `coeff` coordinate labels that layout `L1.v1,
    L1.v2, ..., L2.v1, ...`. So column `(k - 1) * n + j` of `B` multiplies
    `y_{j, t-k}`, and the tested lags of variable `j` are strided `n` apart
    from column `j`.

    Args:
        n_vars: Number of endogenous variables `n`.
        cause_index: Position `j` of the cause in `var_names`.
        n_lags_tested: Number of lags to test, counting from lag 1.

    Returns:
        Column indices into the trailing axis of `B`, lag 1 first.
    """
    return [lag * n_vars + cause_index for lag in range(n_lags_tested)]


def _posterior_coefficients(fitted: FittedVAR) -> np.ndarray:
    """Read `B` as `(chain, draw, var, coeff)`.

    Hand-built posteriors may order their dimensions arbitrarily; realign
    by name when the canonical labels are present, otherwise trust the
    positional convention (the same contract as `dynamic_multiplier`).
    """
    B_da = fitted.idata.posterior["B"]
    if set(B_da.dims) == {"chain", "draw", "var", "coeff"}:
        B_da = B_da.transpose("chain", "draw", "var", "coeff")
    return np.asarray(B_da.values, dtype=float)


def _validate_pair(cause: str, effect: str, var_names: list[str]) -> tuple[int, int]:
    """Resolve the cause/effect names to positions, or explain why not.

    Raises:
        ValueError: If either name is unknown, or if they are the same
            variable.
    """
    unknown = [name for name in (cause, effect) if name not in var_names]
    if unknown:
        raise ValueError(f"unknown variable(s) {unknown}; this model's variables are {var_names}")
    if cause == effect:
        raise ValueError(
            f"cause and effect must be different variables, got {cause!r} for both; "
            "Granger causality compares one variable's past against another's own past."
        )
    return var_names.index(cause), var_names.index(effect)


def _build_result(
    fitted: FittedVAR,
    cause: str,
    effect: str,
    *,
    test_lags: int | None,
    rope: float | None,
    standardize: bool,
    augmentation_source: Literal["none", "integration_order", "user"] | None = None,
    integration_order_result: IntegrationOrderResult | None = None,
) -> GrangerCausalityResult:
    """Extract the tested coefficients and package them with their metadata.

    Sole construction site for `GrangerCausalityResult`, so both entry
    points agree on what `n_lags_tested`, `augmentation`, and `scale` mean.

    Args:
        fitted: The fitted reduced-form posterior to read `B` from.
        cause: Variable whose lags are tested.
        effect: Variable whose equation they are tested in.
        test_lags: Lags to test, counting from lag 1; `None` tests all
            fitted lags.
        rope: Region of practical equivalence, or `None`.
        standardize: Rescale the draws by `sd(cause) / sd(effect)`.
        augmentation_source: Provenance of the untested lags. `None` lets
            it be inferred: `"none"` when nothing was held back, `"user"`
            when the caller shortened `test_lags` by hand.
        integration_order_result: Diagnostics to attach, when consulted.

    Returns:
        GrangerCausalityResult in the requested reporting units.

    Raises:
        ValueError: On unknown or identical variable names, a `test_lags`
            outside `[1, n_lags]`, or a non-positive `rope`.
    """
    from impulso.results import GrangerCausalityResult

    var_names = list(fitted.var_names)
    cause_index, effect_index = _validate_pair(cause, effect, var_names)

    n_lags_fitted = fitted.n_lags
    n_lags_tested = n_lags_fitted if test_lags is None else int(test_lags)
    if not 1 <= n_lags_tested <= n_lags_fitted:
        raise ValueError(f"test_lags must lie in [1, {n_lags_fitted}] (the fitted lag order), got {test_lags}")
    if rope is not None and rope <= 0:
        raise ValueError(
            f"rope must be positive, got {rope}; it is a magnitude in the reporting units of the coefficients."
        )

    B = _posterior_coefficients(fitted)
    columns = _coefficient_indices(len(var_names), cause_index, n_lags_tested)
    coef_draws = B[..., effect_index, :][..., columns]  # (chain, draw, n_lags_tested)

    scale = 1.0
    if standardize:
        sd = np.asarray(fitted.data.endog, dtype=float).std(axis=0, ddof=1)
        scale = float(sd[cause_index] / sd[effect_index])

    augmentation = n_lags_fitted - n_lags_tested
    if augmentation_source is None:
        augmentation_source = "none" if augmentation == 0 else "user"

    return GrangerCausalityResult(
        cause=cause,
        effect=effect,
        n_lags_tested=n_lags_tested,
        n_lags_fitted=n_lags_fitted,
        augmentation=augmentation,
        augmentation_source=augmentation_source,
        standardize=standardize,
        scale=scale,
        rope=rope,
        coef_draws=coef_draws * scale,
        integration_order_result=integration_order_result,
    )


def granger_causality(
    fitted: FittedVAR,
    cause: str,
    effect: str,
    *,
    rope: float | None = None,
    standardize: bool = True,
    test_lags: int | None = None,
) -> GrangerCausalityResult:
    """Engine behind `FittedVAR.granger_causality`.

    Args:
        fitted: Fitted reduced-form posterior.
        cause: Variable whose lags are tested.
        effect: Variable whose equation they are tested in.
        rope: Region of practical equivalence for `p_rope`.
        standardize: Report in `sd(effect)` per `sd(cause)` units.
        test_lags: Lags to test; `None` tests every fitted lag.

    Returns:
        GrangerCausalityResult for the ordered pair.
    """
    return _build_result(
        fitted,
        cause,
        effect,
        test_lags=test_lags,
        rope=rope,
        standardize=standardize,
    )
