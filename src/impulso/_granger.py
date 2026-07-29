"""Granger-causality strength and the Toda-Yamamoto lag-augmented mode.

Two entry points share one result builder, so the metadata a
`GrangerCausalityResult` carries cannot drift between them:

* `granger_causality(fitted, ...)` — backs `FittedVAR.granger_causality`,
  reading the tested lag coefficients straight out of an already-fitted
  posterior. Works under any volatility process, because the coefficient
  matrix `B` is time-invariant under all of them.
* `toda_yamamoto(data, ...)` — the lag-augmented procedure of Toda and
  Yamamoto (1995) for possibly-integrated systems: fit `p + d` lags, test
  only the first `p`. It resolves `d` from the integration-order
  diagnostics unless the caller pins it, and fits with the closed-form
  conjugate estimator so the extra lags cost seconds rather than minutes.

Neither reports a probability of *no* causality — see the
`GrangerCausalityResult` docstring for why that quantity does not exist
under continuous coefficient priors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from impulso.data import VARData

if TYPE_CHECKING:
    from impulso.fitted import FittedVAR
    from impulso.priors import NIWPrior
    from impulso.results import GrangerCausalityResult, IntegrationOrderResult

_CRITERIA = ("aic", "bic", "hq")


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


def _resolve_lag_order(data: VARData, lags: int | str, max_lags: int) -> int:
    """Resolve `lags` to a positive integer `p`, selecting if asked."""
    if isinstance(lags, str):
        if lags not in _CRITERIA:
            raise ValueError(f"lags must be an int or one of {_CRITERIA}, got {lags!r}")
        from impulso._lag_selection import select_lag_order

        return int(getattr(select_lag_order(data, max_lags=max_lags), lags))
    p = int(lags)
    if p < 1:
        raise ValueError(f"lags must be >= 1, got {p}")
    return p


def _resolve_augmentation(
    data: VARData,
    d: int | None,
    integration_order_result: IntegrationOrderResult | None,
    *,
    max_order: int,
    alpha: float,
    regression: Literal["c", "ct"],
) -> tuple[int, Literal["integration_order", "user"], IntegrationOrderResult | None]:
    """Fix the augmentation `d`, either from the caller or from diagnostics.

    An explicit `d` skips the diagnostics entirely — deliberately, so the
    procedure runs without `statsmodels` installed and so a decision the
    analyst has already made is not silently re-litigated.

    Raises:
        ValueError: If `d` is negative, or if the diagnostics left any
            variable in `inconclusive`, where `d_max` is a floor rather
            than a finding and would under-augment the test.
    """
    if d is not None:
        if d < 0:
            raise ValueError(f"d must be non-negative, got {d}")
        return int(d), "user", None

    consulted = integration_order_result
    if consulted is None:
        from impulso._stationarity import integration_order

        consulted = integration_order(data, max_order=max_order, alpha=alpha, regression=regression)
    if consulted.inconclusive:
        raise ValueError(
            f"the integration order of {consulted.inconclusive} is unsettled: each of these is either "
            f"still non-stationary at max_order={consulted.max_order} (so its recorded order is a floor, "
            "not a finding) or had the two unit-root pretests disagree where the search stopped. d_max "
            "would then under-augment, and under-augmented Toda-Yamamoto inference is invalid. Inspect the full "
            "table with integration_order(...).summary(), then pass the augmentation explicitly as "
            "d=<int> once you have decided."
        )
    return consulted.d_max, "integration_order", consulted


def toda_yamamoto(
    data: VARData,
    cause: str,
    effect: str,
    *,
    lags: int | Literal["aic", "bic", "hq"] = "aic",
    max_lags: int = 12,
    d: int | None = None,
    integration_order_result: IntegrationOrderResult | None = None,
    max_order: int = 2,
    alpha: float = 0.05,
    regression: Literal["c", "ct"] = "c",
    rope: float | None = None,
    standardize: bool = True,
    prior: NIWPrior | None = None,
    draws: int = 1000,
    seed: int | None = None,
) -> GrangerCausalityResult:
    """Granger causality with Toda-Yamamoto lag augmentation.

    Toda and Yamamoto (1995) make Granger-causality inference valid without
    first deciding the integration and cointegration structure: fit the VAR
    in levels with `p + d` lags, where `p` is the lag order you would have
    chosen and `d` the highest integration order in the system, then test
    only the first `p` lags. The extra `d` lags are never tested — they
    exist to restore the standard asymptotics — and this function never
    silently changes the reported test lag order to match the fitted one:
    the result carries `n_lags_tested` and `n_lags_fitted` separately.

    `d` comes from `integration_order` unless it is passed explicitly. When
    the diagnostics leave any variable in `inconclusive`, `d_max` is a
    floor rather than a finding, so this function refuses to run rather
    than under-augment; read the full table and pass `d=` yourself.

    The fit uses the closed-form conjugate estimator (`ConjugateVAR` with
    an `NIWPrior`) because augmentation inflates the lag order and the
    conjugate path draws in closed form. For the NUTS estimator, a
    stochastic-volatility process, or exogenous regressors, run the
    procedure by hand — it is three calls:

    ```python
    d = integration_order(data).d_max
    fitted = VAR(lags=p + d).fit(data)
    fitted.granger_causality(cause, effect, test_lags=p)
    ```

    Args:
        data: Endogenous data, in levels. Exogenous regressors are not
            supported here (the conjugate estimator does not consume them).
        cause: Variable whose lags are tested.
        effect: Variable whose equation they are tested in.
        lags: Test lag order `p`, or an information criterion to select it
            with (`"aic"`, `"bic"`, `"hq"`).
        max_lags: Upper bound when `lags` is a criterion.
        d: Augmentation to use. Passing it skips the diagnostics entirely,
            and records `augmentation_source="user"`.
        integration_order_result: Diagnostics to reuse instead of running
            `integration_order` again. Ignored when `d` is given.
        max_order: `max_order` for `integration_order`, when it is run.
        alpha: Significance level for `integration_order`, when it is run.
        regression: Deterministic terms for `integration_order`'s level
            test, when it is run.
        rope: Region of practical equivalence for `p_rope`.
        standardize: Report in `sd(effect)` per `sd(cause)` units. Note
            that the standard deviations of integrated series carry their
            trends, so standardised magnitudes compare best within one fit.
        prior: Conjugate prior for the fit. Defaults to `NIWPrior()`.
        draws: Posterior draws to retain.
        seed: Seed for the conjugate sampler.

    Returns:
        GrangerCausalityResult with `n_lags_tested = p`,
        `augmentation = d`, and the consulted diagnostics attached when
        they were run.

    Raises:
        ValueError: If `data` carries exogenous regressors, if the names
            are unknown or identical, if `lags` or `d` is invalid, or if
            the integration-order diagnostics are inconclusive.
    """
    from impulso.conjugate import ConjugateVAR
    from impulso.priors import NIWPrior

    if data.exog is not None:
        raise ValueError(
            "toda_yamamoto fits with the conjugate estimator, which estimates endogenous dynamics "
            f"only, and this VARData carries exogenous regressors {list(data.exog_names or [])}. Run "
            "the procedure manually instead: d = integration_order(data).d_max, then "
            "fitted = VAR(lags=p + d).fit(data), then "
            "fitted.granger_causality(cause, effect, test_lags=p)."
        )
    # Validate the pair before any fitting or diagnostics, so a typo costs
    # nothing.
    _validate_pair(cause, effect, list(data.endog_names))

    p = _resolve_lag_order(data, lags, max_lags)
    augmentation, source, consulted = _resolve_augmentation(
        data,
        d,
        integration_order_result,
        max_order=max_order,
        alpha=alpha,
        regression=regression,
    )

    fitted = ConjugateVAR(
        lags=p + augmentation,
        prior=prior if prior is not None else NIWPrior(),
        draws=draws,
        seed=seed,
    ).fit(data)

    return _build_result(
        fitted,
        cause,
        effect,
        test_lags=p,
        rope=rope,
        standardize=standardize,
        augmentation_source=source,
        integration_order_result=consulted,
    )
