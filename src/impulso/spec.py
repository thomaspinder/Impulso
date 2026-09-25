"""VAR model specification."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from impulso._arviz_compat import InferenceDataLike
from impulso._base import ImpulsoBaseModel
from impulso._design import build_lag_design_matrix
from impulso._posterior import COEFFICIENTS, EXOG_COEFFICIENTS, INTERCEPT
from impulso.data import VARData, _format_names
from impulso.observation import Gaussian, StudentT
from impulso.priors import MinnesotaPrior
from impulso.protocols import ErrorDistribution, Prior, PyMCVolatilityProcess, Sampler
from impulso.sv.spec import StochasticVolatility
from impulso.volatility import Constant

if TYPE_CHECKING:
    import pytensor.tensor as pt

    from impulso.fitted import FittedVAR

_PRIOR_REGISTRY: dict[str, type] = {
    "minnesota": MinnesotaPrior,
}

_VOLATILITY_REGISTRY: dict[str, type] = {
    "constant": Constant,
    "sv": StochasticVolatility,
}

_ERROR_DIST_REGISTRY: dict[str, type] = {
    "gaussian": Gaussian,
    "student_t": StudentT,
}

# A column whose spread is below this fraction of its own largest absolute value is
# numerically constant even though it passed VARData's exactly-constant check. Using
# its raw standard deviation would inflate the prior towards infinity, so the floor
# substitutes a scale derived from the column's level instead.
_EXOG_SD_FLOOR_FRACTION: float = 1e-3


def _validate_sigma_is_usable(
    sigma: np.ndarray,
    endog_names: Sequence[str],
    *,
    source: Literal["ar1_residual_sd", "endog_scales"] = "ar1_residual_sd",
) -> None:
    """Reject a per-variable scale that would break every prior dividing by it (issue 07b).

    `sigma` (`ar1_residual_sd(endog)`, or the caller's `endog_scales`) is resolved
    once in `VAR.build_in_model` and shared by `Prior.build_priors` — whose
    Minnesota cross-lag entries scale by `sigma[i] / sigma[j]` (docs/adr/0015) —
    and by `_exog_prior_sigma` below. A zero entry collapses that column's own row
    of the coefficient prior toward zero, sends every other row's coefficient on
    its lag to `inf`, and turns the own-lag entry into `0.0 / 0.0 = nan`.

    `VARData` already rejects endogenous columns that are exactly constant over the
    whole sample, which is the common way a column ends up here with `sigma == 0`.
    This check exists for the rarer column that varies (so `VARData` accepts it) but
    is nonetheless perfectly predictable from its own first lag — a short, noiseless
    sample or an exact linear relationship — so `ar1_residual_sd` still returns
    exactly `0.0` for it. It does *not* fire on a column that is merely
    near-constant: dividing by a tiny but nonzero `sigma` gives a large, finite
    prior standard deviation (which can reach `1e12` or more — see
    `MinnesotaPrior.build_priors`), not `inf` or `nan`, and that column is a real,
    varying measurement whose coefficient is still identified.

    `source` only changes the wording of the error: `sigma` reaches this function
    either computed from the data (`ar1_residual_sd`) or supplied directly by the
    caller (`endog_scales`), and a message blaming `ar1_residual_sd` for a bad
    `endog_scales` is simply wrong — that function never ran (issue 08c).

    Args:
        sigma: Per-variable scale, shape `(n_vars,)` — `ar1_residual_sd(endog)`,
            or the caller's `endog_scales`.
        endog_names: Names for each endogenous variable, used only to name the
            offending columns in the error message.
        source: Which one produced `sigma`. `"ar1_residual_sd"` (the default)
            blames the data-derived residual scale; `"endog_scales"` blames the
            caller-supplied array instead.

    Raises:
        ValueError: If any entry of `sigma` is zero, negative, or non-finite.
    """
    bad = np.flatnonzero(~np.isfinite(sigma) | (sigma <= 0.0))
    if not bad.size:
        return
    labels = [endog_names[i] for i in bad]
    if source == "endog_scales":
        raise ValueError(
            f"endog_scales has a zero, negative, or non-finite entry for these columns: "
            f"{_format_names(labels)}. This scale is shared by the Minnesota cross-lag prior "
            "(docs/adr/0015) and the exogenous-coefficient prior, both of which divide by it. Supply a "
            "strictly positive, finite scale for every variable, or omit `endog_scales` so it is "
            "derived from the data instead."
        )
    raise ValueError(
        f"endog columns have a zero, negative, or non-finite scale: {_format_names(labels)}. "
        "`ar1_residual_sd` came out non-positive (or non-finite) for these columns, and this scale "
        "is shared by the Minnesota cross-lag prior (docs/adr/0015) and the exogenous-coefficient "
        "prior, both of which divide by it. VARData already rejects columns that are constant over "
        "the whole sample; this column varies but is nonetheless perfectly predictable from its own "
        "first lag (e.g. a very short, noiseless sample), so its residual scale is exactly zero. Add "
        "noise, drop the column, or otherwise make its scale identified."
    )


def _resolve_sigma(
    endog: np.ndarray,
    endog_scales: np.ndarray | Sequence[float] | None,
    endog_names: Sequence[str],
    n_vars: int,
) -> np.ndarray:
    """Resolve and validate `VAR.build_in_model`'s shared `sigma` (issue 08c).

    `endog_scales=None` computes `sigma` from `endog` via `ar1_residual_sd`.
    Otherwise the caller's array is coerced with `np.asarray(..., dtype=float)`
    — a plain list is accepted, not only an ndarray — and its shape checked
    against `n_vars` before validation, so a wrong-length `endog_scales` raises
    a clear `ValueError` here instead of an opaque `TypeError`/`IndexError`
    from numpy code further down. Either way, `_validate_sigma_is_usable` gets
    told which path produced `sigma`, so its error names the actual source.

    Args:
        endog: Endogenous data, shape `(T, n_vars)`.
        endog_scales: Caller-supplied scale, or `None` to derive it from `endog`.
        endog_names: Names for each endogenous column, used to name offending
            columns in a validation error.
        n_vars: Expected length of `sigma` — `endog.shape[1]`.

    Returns:
        `sigma`, shape `(n_vars,)`.

    Raises:
        ValueError: If `endog_scales` does not have shape `(n_vars,)`.
        ValueError: If any entry of the resolved `sigma` is zero, negative or
            non-finite (issue 07b).
    """
    # Lazy: `_conjugate` imports scipy at module level, and `spec` is on the
    # package import path.
    from impulso._conjugate import ar1_residual_sd

    if endog_scales is None:
        sigma = ar1_residual_sd(endog)
        _validate_sigma_is_usable(sigma, endog_names, source="ar1_residual_sd")
        return sigma

    sigma = np.asarray(endog_scales, dtype=float)
    if sigma.shape != (n_vars,):
        raise ValueError(f"endog_scales must have shape ({n_vars},) to match n_vars={n_vars}, got shape {sigma.shape}")
    _validate_sigma_is_usable(sigma, endog_names, source="endog_scales")
    return sigma


def _symbolic_endog_n_vars(
    endog: "pt.TensorVariable",
    endog_scales: np.ndarray | Sequence[float] | None,
    endog_names: Sequence[str],
) -> int:
    """Validate a symbolic `endog` for `VAR.build_in_model` and return `n_vars` (issue 09a).

    A symbolic `endog` (e.g. a `pm.Data` container the caller owns) cannot go
    through the numpy-only steps: `ar1_residual_sd`, the OLS pre-fit
    residuals, or an observed RV. Its column count may not be static, so
    `n_vars` comes from `endog_names`, checked against the static count when
    there is one.

    Raises:
        ValueError: If `endog` is not 2-D, if `endog_scales` is `None`, or if
            the static column count differs from `len(endog_names)`.
    """
    if endog.ndim != 2:
        raise ValueError(f"endog must be 2-D (T, n_vars), got a {endog.ndim}-D tensor")
    if endog_scales is None:
        raise ValueError(
            "endog_scales is required when endog is symbolic: the default scale comes from "
            "`ar1_residual_sd`, which needs concrete data. Pass the per-variable scales the "
            "Minnesota cross-lag prior and the exogenous prior should use, e.g. "
            "`impulso.ar1_residual_sd(values)` on the data behind the tensor."
        )
    n_vars = len(endog_names)
    static_n_vars = endog.type.shape[1]
    if static_n_vars is not None and static_n_vars != n_vars:
        raise ValueError(f"endog has {static_n_vars} columns but endog_names has {n_vars} names")
    return n_vars


def _ols_residuals(Y: np.ndarray, X_lag: np.ndarray, X_exog: np.ndarray | None) -> np.ndarray:
    """OLS residuals of `Y` on an intercept, `X_lag` and (optional) `X_exog`.

    Seeds the volatility process's per-variable priors in
    `VAR.build_in_model`. Numpy-only: it needs concrete data.
    """
    if X_exog is not None:
        X_full = np.hstack([np.ones((Y.shape[0], 1)), X_lag, X_exog])
    else:
        X_full = np.hstack([np.ones((Y.shape[0], 1)), X_lag])
    B_ols, *_ = np.linalg.lstsq(X_full, Y, rcond=None)
    return Y - X_full @ B_ols


def _time_coord(model: Any, n_rows: int | None) -> dict[str, object]:
    """The `"time"` coord `VAR.build_in_model` must add, if any.

    Returns `{}` when there is nothing to add, `{"time": range}` otherwise.

    Args:
        model: The active `pymc.Model`.
        n_rows: Number of likelihood rows, `T - n_lags`, or `None` for a
            symbolic `endog` whose static shape does not know `T`.

    Raises:
        ValueError: If `model` already has a `"time"` coord whose length is
            not `n_rows`.
    """
    if n_rows is None:
        # Symbolic `endog` of unknown length: its likelihood is a
        # `pm.Potential`, which carries no dims, so nothing needs a
        # "time" coord and there is no length to check one against.
        return {}
    if "time" in model.coords:
        # A previous call (this VAR's own wrapper, or a second VAR
        # embedded in the same model — coords are not prefixed by a
        # nested `pm.Model(name=...)`, see `VAR.build_in_model`) already
        # registered "time". `add_coords` only rejects a duplicate coord
        # whose *values* differ, and a plain length mismatch has equal
        # odds of matching by chance as differing, so silently reusing
        # it would either pass by luck or hand the likelihood a "time"
        # dim of the wrong length — a shape error that would only
        # surface much later, e.g. inside `sample_prior_predictive`.
        # Reject it here instead, at the point that actually knows both
        # lengths.
        existing_length = int(model.dim_lengths["time"].eval())
        if existing_length != n_rows:
            raise ValueError(
                f"the active model already has a 'time' coordinate of length "
                f"{existing_length}, but this call's likelihood has {n_rows} rows "
                "(T - n_lags). Coordinates are not prefixed by a nested "
                "pm.Model(name=...), so two VARs embedded in the same model share a "
                "single 'time' coordinate and must agree on its length. Give both VARs "
                "the same number of likelihood rows, or build them in separate "
                "pm.Model() instances."
            )
        return {}
    # PyMC requires a named dim used on an *observed* multivariate RV
    # to already exist (unlike a free RV's `dims`, which it will
    # auto-register). `_build_pymc_model` pre-registers "time" from
    # `data.index` before calling `build_in_model`; a standalone call
    # with no pre-registered "time" coord falls back to a plain
    # positional index.
    return {"time": list(range(n_rows))}


def _intercept_mask(endog_names: Sequence[str], intercept_equations: Sequence[str] | None) -> np.ndarray:
    """Boolean mask over `endog_names`: which equations get an intercept.

    `None` means every equation. Otherwise every name must appear in
    `endog_names`, at most once; the mask follows `endog_names`' order
    regardless of the order `intercept_equations` lists them in.
    """
    if intercept_equations is None:
        return np.ones(len(endog_names), dtype=bool)
    unknown = [name for name in intercept_equations if name not in endog_names]
    if unknown:
        raise ValueError(
            f"intercept_equations names {_format_names(unknown)}, which are not endogenous variables; "
            f"expected a subset of endog_names ({_format_names(endog_names)})."
        )
    duplicates = sorted({name for name in intercept_equations if list(intercept_equations).count(name) > 1})
    if duplicates:
        raise ValueError(f"intercept_equations lists {_format_names(duplicates)} more than once.")
    return np.array([name in intercept_equations for name in endog_names], dtype=bool)


def _exog_prior_sigma(
    sigma: np.ndarray,
    x_exog: np.ndarray,
    scale: float,
    exog_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Prior standard deviations for the exogenous coefficients `B_exog`.

    The coefficient on an exogenous regressor is not a unit-free quantity: it
    converts the regressor's units into the dependent variable's. A prior fixed
    in coefficient space therefore encodes a different belief for every dataset
    — crushing coefficients on small-scale regressors and leaving coefficients
    on large-scale ones effectively unrestricted. This scales the prior so the
    belief lives in *contribution* space instead:

        sd[i, j] = scale * sigma_i / s_j

    where `sigma_i` is the AR(1) residual standard deviation of endogenous
    variable `i` (the same scale `MinnesotaPrior.build_priors` uses for its lag
    coefficients — see docs/adr/0015) and `s_j` is the sample standard
    deviation of exogenous column `j`. One prior standard deviation of
    `B_exog[i, j]` then moves variable `i` by `scale` of its own residual
    standard deviation when regressor `j` moves by one of its own. The default
    `scale` is deliberately loose (see `VAR.exog_prior_scale`).

    Args:
        sigma: Per-endogenous-variable AR(1) residual standard deviation,
            shape `(n_vars,)` — `ar1_residual_sd(endog)`, or the caller's
            `endog_scales`. `VAR.build_in_model` resolves this once and passes
            the same array here and to `Prior.build_priors`, so the
            coefficient and exogenous priors are expressed in the same units.
        x_exog: Exogenous regressor block of shape `(T_eff, n_exog)`, already
            trimmed to the rows the likelihood sees.
        scale: Multiplier in units of "residual standard deviations of the
            dependent variable per standard deviation of the regressor".
        exog_names: Optional column names, used only to make the
            constant-column error message readable.

    Returns:
        Array of shape `(n_vars, n_exog)` of prior standard deviations.

    Raises:
        ValueError: If a column of `x_exog` is exactly constant. `VARData`
            rejects columns that are constant over the whole sample, but
            trimming the first `n_lags` rows can flatten a column that did
            vary — a dummy that only switches inside the initial conditions,
            say. What the likelihood then sees is collinear with the
            intercept, so the coefficient is not identified; the floor below
            would happily hand it a wide prior and hide that.
    """
    s = x_exog.std(axis=0, ddof=1)
    # Checked before the floor is applied: the floor exists to tame columns with
    # tiny-but-real variation, not to manufacture a scale for columns with none.
    degenerate = np.flatnonzero(s <= 0.0)
    if degenerate.size:
        labels = [exog_names[j] if exog_names is not None else f"column {j}" for j in degenerate]
        raise ValueError(
            f"exog columns are constant over the estimation sample: {_format_names(labels)}. "
            "The first n_lags rows are consumed as initial conditions, and what remains of these columns "
            "does not vary, so their coefficients are collinear with the intercept and not identified. "
            "Drop the columns, or reduce `lags` so the rows that do vary enter the estimation sample."
        )
    peak = np.abs(x_exog).max(axis=0)
    s_eff = np.maximum(s, _EXOG_SD_FLOOR_FRACTION * peak)
    return scale * np.outer(sigma, 1.0 / s_eff)


@dataclass(frozen=True)
class VARModelHandles:
    """PyMC variables `VAR.build_in_model` registers into the active model.

    Handed back so a caller embedding a VAR inside a larger PyMC model (or
    inspecting the graph `fit`/`prior_predictive` build) can reach the
    pieces directly, without re-deriving PyMC's own name-mangling inside a
    nested `pm.Model(name=...)`.

    Attributes:
        intercept: Per-equation intercept. `dims=("var",)` when every
            equation has one (the default); `dims=("var_intercept",)`,
            covering only the included equations, when `build_in_model` was
            given a strict subset via `intercept_equations`; `None` when
            `intercept_equations` excluded every equation.
        B: VAR lag coefficients, `dims=("var", "coeff")`.
        B_exog: Exogenous coefficients, `dims=("var", "exog")`, or `None`
            when no exogenous block was registered.
        L: Lower-triangular Cholesky factor of the structural-shock scale
            matrix — `(n_vars, n_vars)` for constant volatility, `(T,
            n_vars, n_vars)` for stochastic volatility.
        obs: The registered observation likelihood. For a numpy `endog`,
            `error_dist.build_likelihood`'s return value, an observed RV.
            For a symbolic `endog`, the `pm.Potential` wrapping
            `error_dist.logp`, which has no dims and does not appear in
            `sample_prior_predictive`. Named `"obs"` either way.
    """

    intercept: "pt.TensorVariable | None"
    B: "pt.TensorVariable"
    B_exog: "pt.TensorVariable | None"
    L: "pt.TensorVariable"
    obs: "pt.TensorVariable"


class VAR(ImpulsoBaseModel):
    """Immutable VAR model specification.

    `VAR` specifies the *reduced-form* model — lag order, coefficient prior,
    volatility process, and observation error distribution. Nothing here says
    which shock is which: structural meaning is layered on afterwards, by
    applying an identification scheme to the `FittedVAR` that `fit` returns.

    Attributes:
        lags: Fixed lag order (int >= 1) or selection criterion string.
        max_lags: Upper bound for automatic selection. Only valid with string lags.
        prior: Prior shorthand string or Prior protocol instance.
        volatility: Volatility shorthand string or PyMCVolatilityProcess protocol instance.
        exog_prior_scale: Tightness of the prior on the exogenous coefficients
            `B_exog`, read in contribution space: one prior standard deviation
            moves an endogenous variable by this many of its own AR(1) residual
            standard deviations when the regressor moves by one of its own. The
            default of 100 is deliberately loose — deterministic and exogenous
            terms are conventionally left near-uninformative (the conjugate
            engine uses `Vc = 10e6` on the intercept), and the prior's job here
            is to stop the scale of the regressor from silently setting the
            answer, not to shrink. Lower it to shrink `B_exog` towards zero.
            Applies only to `VAR.fit`; `prior` governs the lag coefficients.
        error_dist: Observation error distribution — shorthand string
            (`"gaussian"`, the default, or `"student_t"`) or an
            `ErrorDistribution` protocol instance. The string form takes the
            adapter's defaults, so `error_dist="student_t"` *infers* the
            degrees of freedom; pass `StudentT(nu=5.0)` to fix them. Heavy-
            tailed errors are rejected in combination with time-varying
            volatility.
            Governs the exogenous block only; `prior` governs the lag
            coefficients. Both `VAR.fit` and `VAR.prior_predictive` build the
            same graph, so it applies to either.
    """

    lags: int | Literal["aic", "bic", "hq"] = Field(...)
    max_lags: int | None = None
    prior: Literal["minnesota"] | Prior = "minnesota"
    volatility: Literal["constant", "sv"] | PyMCVolatilityProcess = "constant"
    exog_prior_scale: float = Field(100.0, gt=0)
    error_dist: Literal["gaussian", "student_t"] | ErrorDistribution = "gaussian"

    @model_validator(mode="after")
    def _validate_spec(self) -> Self:
        if self.max_lags is not None and isinstance(self.lags, int):
            raise ValueError("max_lags is only valid when lags is a selection criterion ('aic', 'bic', 'hq')")
        if isinstance(self.lags, int) and self.lags < 1:
            raise ValueError(f"lags must be >= 1, got {self.lags}")
        if self.resolved_error_dist.is_heavy_tailed and self.resolved_volatility.is_time_varying:
            raise ValueError(
                "Heavy-tailed observation errors are not yet supported with "
                "time-varying volatility: the degrees of freedom and the "
                "log-volatility innovation variance both absorb outliers, so "
                "the two are only weakly identified jointly and NUTS mixes "
                "poorly. Use volatility='constant' with error_dist='student_t', "
                "or stochastic volatility with Gaussian errors."
            )
        return self

    @property
    def resolved_prior(self) -> Prior:
        """Resolve string prior shorthand to a Prior instance."""
        if isinstance(self.prior, str):
            return _PRIOR_REGISTRY[self.prior]()
        return self.prior

    @property
    def resolved_volatility(self) -> PyMCVolatilityProcess:
        """Resolve string volatility shorthand to a PyMCVolatilityProcess instance."""
        if isinstance(self.volatility, str):
            return _VOLATILITY_REGISTRY[self.volatility]()
        return self.volatility

    @property
    def resolved_error_dist(self) -> ErrorDistribution:
        """Resolve string error-distribution shorthand to an ErrorDistribution instance."""
        if isinstance(self.error_dist, str):
            return _ERROR_DIST_REGISTRY[self.error_dist]()
        return self.error_dist

    @staticmethod
    def _default_sampler() -> Sampler:
        """Default sampler for VAR: cores=1 (macOS PyMC segfault), target_accept=0.8."""
        from impulso.samplers import NUTSSampler

        return NUTSSampler(cores=1, chains=4)

    def fit(
        self,
        data: VARData,
        sampler: Sampler | None = None,
    ) -> "FittedVAR":
        """Estimate the Bayesian VAR model.

        Args:
            data: VARData instance.
            sampler: Sampler protocol instance. Defaults to `_default_sampler()`
                (`cores=1`, `chains=4`, `target_accept=0.8`). Pass an explicit
                `NUTSSampler(cores=n)` to opt into parallel chains.

        Returns:
            FittedVAR with posterior draws.
        """
        from impulso.fitted import FittedVAR

        if sampler is None:
            sampler = self._default_sampler()

        model, n_lags = self._build_pymc_model(data)

        # Sample
        idata = sampler.sample(model)

        return FittedVAR.from_posterior(
            idata,
            data,
            n_lags,
            volatility=self.resolved_volatility,
            error_dist=self.resolved_error_dist,
            pymc_model=model,
        )

    def prior_predictive(
        self,
        data: VARData,
        *,
        draws: int = 500,
        random_seed: int | np.random.Generator | None = None,
    ) -> InferenceDataLike:
        """Simulate data from the prior, before seeing the likelihood.

        Builds the same PyMC graph `fit` builds and calls
        `pymc.sample_prior_predictive` on it, so the prior that gets
        simulated is exactly the prior that gets sampled — no hand-rolled
        second implementation to drift out of sync.

        The simulated `obs` paths are **one-step-ahead given the observed
        lags**: for each prior draw, `y_t = c + B x_t^obs (+ B_exog z_t) +
        L_t eps_t` where `x_t^obs` stacks the *observed* lags of `data`.
        The design matrices are baked into the graph, so this is the prior
        predictive of the estimation-sample conditional means, not a
        simulated path iterated from initial conditions. That is what
        `arviz.plot_ppc(..., group="prior")` expects and what makes the
        prior comparable to the data on the same time axis.

        Note:
            Under `volatility="sv"` the per-variable log-volatility priors
            are seeded from the OLS residuals of `data` (see
            `StochasticVolatility.build_pymc_latent`), so the "prior" is
            mildly data-informed in its scale. The constant-volatility
            default is not.

        Note:
            PyMC returns a single chain, so the `obs` variable has shape
            `(1, draws, T - n_lags, n_vars)`.

        Args:
            data: VARData instance. Anchors the prior simulation on the real
                lags (and, if present, the real exogenous regressors), and
                fixes the lag order when `lags` is a selection criterion.
            draws: Number of prior draws.
            random_seed: Seed or Generator passed straight through to
                `pymc.sample_prior_predictive`.

        Returns:
            InferenceData-schema container with `prior` (every latent), `prior_predictive`
            (the simulated `obs`, dims `(chain, draw, time, var)`) and
            `observed_data` (the realised `obs`) groups.
        """
        import pymc as pm

        model, _ = self._build_pymc_model(data)
        with model:
            return pm.sample_prior_predictive(draws=draws, random_seed=random_seed)

    def build_in_model(
        self,
        endog: "np.ndarray | pt.TensorVariable",
        exog: np.ndarray | None,
        n_lags: int,
        endog_names: Sequence[str],
        exog_names: Sequence[str] | None = None,
        endog_scales: np.ndarray | Sequence[float] | None = None,
        intercept_equations: Sequence[str] | None = None,
    ) -> VARModelHandles:
        """Register this VAR specification into the active PyMC model.

        The public counterpart of `_build_pymc_model`: where that wrapper
        opens a fresh model and converts a `VARData` into arrays, this
        method takes the arrays directly and registers the intercept, lag
        coefficients, (optional) exogenous coefficients, volatility latents
        and observation likelihood into whichever `pymc.Model` is active on
        entry (`pymc.modelcontext(None)`). `_build_pymc_model` routes
        through this method too, so `fit` and `prior_predictive` share the
        same code path with a caller embedding a VAR inside a larger PyMC
        model — a marketing-mix model with a VAR-shaped baseline, say.

        Symbolic `endog`: `endog` may be a PyTensor variable instead of a
        numpy array, e.g. a `pm.Data` container the caller registered. The
        observed block's likelihood is then `error_dist.logp` wrapped in a
        `pm.Potential` named `"obs"` (a symbolic value cannot be an RV's
        `observed`), with the same density the numpy path's observed RV
        contributes. The numpy-only steps are skipped: `endog_scales` is
        required, since `ar1_residual_sd` needs concrete data, and the
        volatility process gets `data=None` instead of OLS pre-fit
        residuals, which only `Constant` volatility accepts, as it ignores
        them. Callers resolve string lag-selection criteria (e.g.
        via `select_lag_order`) before calling this method — it always
        takes a concrete integer `n_lags`.

        Nesting: open a `pm.Model(name=prefix)` before calling this method
        and every free random variable, `Deterministic` and the likelihood
        it registers come out named `prefix::...` — ordinary PyMC nested-
        model behaviour (see the "Nested `pm.Model(name=prefix)`" section
        of `prototype/REPORT.md`). Coordinates are the one exception: PyMC
        does not prefix coords, so `add_coords` below always lands on the
        *root* model, shared by every nested submodel. Embed at most one
        VAR's variable labelling per model — two VARs with different
        `endog_names`/`exog_names` embedded in the same model will collide
        on `var`/`coeff`/`exog` (identical labels are shared silently;
        different labels raise `ValueError`). `"time"` is a coordinate too,
        so it is subject to the same sharing: two VARs embedded in the same
        model must agree on its *length* (see "Time coordinate" below) —
        checked explicitly, because `add_coords` alone only rejects a
        duplicate coordinate whose *values* differ, not one whose length
        happens to differ while its (unlabelled) content still matches.

        Time coordinate: the likelihood is registered with `dims=("time",
        "var")`, and PyMC requires an *observed* multivariate RV's named
        dims to already be coordinates on the model — unlike a free RV, it
        will not silently auto-register them. If the active model does not
        already carry a `"time"` coordinate, this method adds a plain
        positional one (`range(T_eff)`). `_build_pymc_model` pre-registers
        `"time"` from `data.index` before calling this method, so `fit` and
        `prior_predictive` keep real dates; a caller invoking this method
        directly gets the positional fallback unless it registers `"time"`
        itself first. If the active model *already* carries a `"time"`
        coordinate — this VAR's own wrapper, or a second VAR embedded in
        the same model — and its length does not match this call's number
        of likelihood rows (`T - n_lags`), this method raises `ValueError`
        rather than silently reusing the wrong length; equal length is
        fine regardless of the actual values. A symbolic `endog` is handled
        the same way when its static shape knows `T`. When it does not, as
        for `pm.Data` or `pytensor.shared`, whose value can be swapped for
        another length, no `"time"` coordinate is registered or checked:
        the `pm.Potential` likelihood carries no dims.

        Args:
            endog: Endogenous data, shape `(T, n_vars)`: a numpy array, or a
                2-D PyTensor variable (see "Symbolic `endog`" above).
            exog: Optional exogenous regressors, shape `(T, n_exog)`. `None`
                if the model has no exogenous block.
            n_lags: Lag order. Always a concrete integer — resolving a
                string selection criterion is the caller's job.
            endog_names: Names for each endogenous column, length
                `n_vars`. Labels the `var`/`var1`/`var2`/`coeff` coordinates.
            exog_names: Names for each exogenous column, length `n_exog`.
                Required when `exog` is given; labels the `exog` coordinate.
            endog_scales: Per-variable scale `sigma`, shape `(n_vars,)` — an
                array or any array-like (e.g. a plain list) accepted by
                `np.asarray(..., dtype=float)`. `None` (the default) computes
                it from `endog` with `ar1_residual_sd`; required when `endog`
                is symbolic. The same array feeds
                both the prior's Minnesota cross-lag scaling `sigma_i /
                sigma_j` (`Prior.build_priors`, docs/adr/0015) and the
                exogenous prior (`_exog_prior_sigma`, docs/adr/0012), so it
                matters even when `exog` is `None`.
            intercept_equations: Names of the endogenous equations that get
                an intercept, a subset of `endog_names` in any order. `None`
                (the default) means every equation — the same graph as
                before this argument existed. An excluded equation has no
                intercept term at all (its `mu` adds zero), so its series is
                modelled as a zero-mean deviation around a level owned
                elsewhere in the model. Naming every equation, in any order,
                is the same as `None`: the intercept keeps `dims="var"`. A
                strict subset gets its own `"var_intercept"` coordinate,
                ordered like `endog_names` (not like this argument), and
                like every Impulso coordinate it is not prefixed by a nested
                model. An empty sequence registers no intercept variable,
                and the returned handles' `intercept` is `None`.

        Returns:
            `VARModelHandles` wrapping the intercept, coefficient,
            volatility and likelihood variables this call registered.

        Raises:
            ValueError: If the active model already carries a `"time"`
                coordinate whose length does not match this call's number
                of likelihood rows (`T - n_lags`) — see "Time coordinate"
                above.
            ValueError: If any entry of the scale — computed or supplied via
                `endog_scales` — is zero, negative or non-finite (issue 07b).
            ValueError: If `endog_scales` does not have shape `(n_vars,)`
                (issue 08c).
            ValueError: If `endog` is symbolic and `endog_scales` is `None`,
                if it is not 2-D, or if its static column count differs
                from `len(endog_names)` (issue 09a).
            ValueError: If `intercept_equations` names an equation not in
                `endog_names`, or names one more than once.
        """
        import pymc as pm
        import pytensor.tensor as pt
        from pytensor.graph.basic import Variable

        model = pm.modelcontext(None)

        # A symbolic `endog` (issue 09a) — e.g. a `pm.Data` container the
        # caller owns — cannot go through the numpy-only steps below:
        # `ar1_residual_sd`, the OLS pre-fit residuals, or an observed RV.
        symbolic = isinstance(endog, Variable)
        n_vars = _symbolic_endog_n_vars(endog, endog_scales, endog_names) if symbolic else endog.shape[1]

        # `sigma` is the per-variable scale — computed once here (or taken
        # from the caller) and reused for both the Minnesota lag-coefficient
        # prior (cross-lag sigma_i/sigma_j scaling, docs/adr/0015) and the
        # exogenous-coefficient prior below (#192). `_resolve_sigma` coerces
        # and shape-checks a caller-supplied `endog_scales` and validates
        # either path immediately: a zero/non-finite entry would blow up
        # both (issue 07b), and it tailors the error to the actual source
        # (issue 08c).
        sigma = _resolve_sigma(endog, endog_scales, endog_names, n_vars)
        intercept_mask = _intercept_mask(endog_names, intercept_equations)
        prior_params = self.resolved_prior.build_priors(n_vars=n_vars, n_lags=n_lags, sigma=sigma)

        Y, X_lag, X_exog = build_lag_design_matrix(endog, n_lags, exog)

        # Number of likelihood rows, `T - n_lags`. A symbolic `endog` only
        # knows it when its static shape does: `pm.Data` and
        # `pytensor.shared` leave it `None`, since their value can be
        # swapped for one of a different length. Then it stays `None`.
        if symbolic:
            static_T = endog.type.shape[0]
            n_rows = None if static_T is None else static_T - n_lags
        else:
            n_rows = Y.shape[0]

        # OLS pre-fit residuals seed the volatility process's per-variable
        # priors. Numpy-only: they need concrete data, so a symbolic `endog`
        # passes `None`. Constant-volatility adapters ignore this; only
        # stochastic ones use it.
        resid = None if symbolic else _ols_residuals(Y, X_lag, X_exog)

        # Coordinates make the posterior self-describing: `B` comes back labelled
        # by variable and by "L<lag>.<variable>" coefficient instead of positional
        # `B_dim_0` / `B_dim_1`. Variable names come from `impulso._posterior` —
        # the schema ConjugateVAR constructs against too, so both estimators
        # agree. `coeff` is lag-major to mirror the X_lag hstack above. Not
        # prefixed by a nested `pm.Model(name=...)` — see the docstring above.
        coords: dict[str, object] = {
            "var": list(endog_names),
            "var1": list(endog_names),
            "var2": list(endog_names),
            "coeff": [f"L{lag}.{name}" for lag in range(1, n_lags + 1) for name in endog_names],
        }
        if exog_names is not None:
            coords["exog"] = list(exog_names)
        intercepted = [name for name, keep in zip(endog_names, intercept_mask, strict=True) if keep]
        if 0 < len(intercepted) < n_vars:
            coords["var_intercept"] = intercepted
        coords.update(_time_coord(model, n_rows))
        model.add_coords(coords)

        # Intercept. Every equation gets one by default (`dims="var"`, the
        # graph from before `intercept_equations` existed). A strict subset
        # gets a shorter free variable on its own coord, scattered into a
        # length-`n_vars` vector with literal zeros for the excluded
        # equations; excluding every equation drops the term entirely.
        if len(intercepted) == n_vars:
            intercept = pm.Normal(INTERCEPT, mu=0, sigma=1, dims="var")
            intercept_term = intercept
        elif intercepted:
            intercept = pm.Normal(INTERCEPT, mu=0, sigma=1, dims="var_intercept")
            intercept_term = pt.zeros(n_vars)[np.flatnonzero(intercept_mask)].set(intercept)
        else:
            intercept = None
            intercept_term = pt.zeros(n_vars)

        # VAR coefficients with Minnesota prior
        B = pm.Normal(
            COEFFICIENTS,
            mu=prior_params["B_mu"],
            sigma=prior_params["B_sigma"],
            dims=("var", "coeff"),
        )

        # Exogenous coefficients. The prior scales with the data so that it
        # encodes the same belief regardless of the units the regressors
        # happen to be measured in (#192).
        if X_exog is not None:
            B_exog = pm.Normal(
                EXOG_COEFFICIENTS,
                mu=0,
                sigma=_exog_prior_sigma(sigma, X_exog, self.exog_prior_scale, exog_names),
                dims=("var", "exog"),
            )
            mu = intercept_term + pm.math.dot(X_lag, B.T) + pm.math.dot(X_exog, B_exog.T)
        else:
            B_exog = None
            mu = intercept_term + pm.math.dot(X_lag, B.T)

        # Volatility process: registers latent vars, returns L (Cholesky factor of Σ_t).
        # For constant volatility, L is (n_vars, n_vars) and time-invariant.
        # For stochastic volatility, L is (T, n_vars, n_vars) — per-t.
        volatility = self.resolved_volatility
        # `T` is ignored by `Constant`, the one adapter the symbolic path
        # supports; an unknown static length falls back to the symbolic one.
        L = volatility.build_pymc_latent(n_vars=n_vars, T=Y.shape[0] if n_rows is None else n_rows, data=resid)
        # Sigma deterministic is only registered for time-invariant L —
        # for SV, materialising (T, n, n) per draw is wasteful; users can
        # reconstruct per-t Σ via `volatility.cholesky_at(posterior, t)`.
        if L.ndim == 2:
            pm.Deterministic("Sigma", pm.math.dot(L, L.T), dims=("var1", "var2"))

        # Likelihood. The error-distribution seam owns which law is
        # registered; PyMC handles batched chol natively either way (for
        # 2D L every observation uses the same chol; for 3D L (T, n, n)
        # observation t uses chol[t]). Under Student-t errors, L L' is the
        # *scale* matrix rather than the covariance — see ADR-0007.
        #
        # A symbolic `endog` cannot be an RV's `observed` value, so its
        # likelihood is the same density as a `pm.Potential` under the same
        # name. A Potential is invisible to `sample_prior_predictive`.
        error_dist = self.resolved_error_dist
        if symbolic:
            obs = pm.Potential("obs", error_dist.logp(mu=mu, chol=L, value=Y))
        else:
            obs = error_dist.build_likelihood("obs", mu=mu, chol=L, observed=Y, dims=("time", "var"))

        return VARModelHandles(intercept=intercept, B=B, B_exog=B_exog, L=L, obs=obs)

    def _build_pymc_model(self, data: VARData) -> tuple[Any, int]:
        """Build the PyMC model graph for this specification.

        Resolves the lag order (running `select_lag_order` when `lags` is a
        criterion string), opens a fresh `pymc.Model`, and delegates to
        `build_in_model` to register the intercept, coefficient, exogenous,
        volatility and likelihood nodes. The design matrices are baked into
        the graph as constants, so the returned model is tied to `data`.

        Shared by `fit` (which samples the graph) and `prior_predictive`
        (which draws from it without conditioning on the observations).

        Args:
            data: VARData instance.

        Returns:
            Tuple of the built `pymc.Model` and the resolved lag order. The
            model is typed `Any` so that importing `impulso.spec` does not
            pull in PyMC — the same reason `FittedVAR.pymc_model` is.
        """
        import pymc as pm

        from impulso._lag_selection import select_lag_order

        # Resolve lags
        if isinstance(self.lags, str):
            max_lags = self.max_lags or 12
            ic = select_lag_order(data, max_lags=max_lags)
            n_lags = getattr(ic, self.lags)
        else:
            n_lags = self.lags

        # "time" is registered here, from `data.index`, rather than inside
        # `build_in_model` — that method has no date index to draw one
        # from, only arrays. Registering it before `build_in_model` runs
        # means the likelihood's `dims=("time", "var")` binds to real dates.
        with pm.Model(coords={"time": data.index[n_lags:]}) as model:
            self.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=n_lags,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
            )

        return model, n_lags
