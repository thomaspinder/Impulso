"""VAR model specification."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from impulso._base import ImpulsoBaseModel
from impulso.data import VARData, _format_names
from impulso.priors import MinnesotaPrior
from impulso.protocols import Prior, PyMCVolatilityProcess, Sampler
from impulso.sv.spec import StochasticVolatility
from impulso.volatility import Constant

if TYPE_CHECKING:
    from impulso.fitted import FittedVAR

_PRIOR_REGISTRY: dict[str, type] = {
    "minnesota": MinnesotaPrior,
}

_VOLATILITY_REGISTRY: dict[str, type] = {
    "constant": Constant,
    "sv": StochasticVolatility,
}

# A column whose spread is below this fraction of its own largest absolute value is
# numerically constant even though it passed VARData's exactly-constant check. Using
# its raw standard deviation would inflate the prior towards infinity, so the floor
# substitutes a scale derived from the column's level instead.
_EXOG_SD_FLOOR_FRACTION: float = 1e-3


def _exog_prior_sigma(
    endog: np.ndarray,
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
    variable `i` (the same scale the Minnesota prior uses) and `s_j` is the
    sample standard deviation of exogenous column `j`. One prior standard
    deviation of `B_exog[i, j]` then moves variable `i` by `scale` of its own
    residual standard deviation when regressor `j` moves by one of its own.
    The default `scale` is deliberately loose (see `VAR.exog_prior_scale`).

    Args:
        endog: Endogenous data of shape `(T, n_vars)`. Passed whole — `sigma_i`
            is a property of the series, not of the estimation sample.
        x_exog: Exogenous regressor block of shape `(T_eff, n_exog)`, already
            trimmed to the rows the likelihood sees.
        scale: Multiplier in units of "residual standard deviations of the
            dependent variable per standard deviation of the regressor".
        exog_names: Optional column names, used only to make the all-zero
            error message readable.

    Returns:
        Array of shape `(n_vars, n_exog)` of prior standard deviations.

    Raises:
        ValueError: If a column of `x_exog` is identically zero. `VARData`
            rejects columns that are constant over the whole sample, but
            trimming the first `n_lags` rows can still leave nothing behind
            (a pulse dummy at the very start of the sample, say). Such a
            regressor never enters the likelihood, so there is no scale to
            key the prior off and the posterior would be the prior.
    """
    # Lazy: `_conjugate` imports scipy at module level, and `spec` is on the
    # package import path.
    from impulso._conjugate import ar1_residual_sd

    sigma = ar1_residual_sd(endog)
    s = x_exog.std(axis=0, ddof=1)
    peak = np.abs(x_exog).max(axis=0)
    s_eff = np.maximum(s, _EXOG_SD_FLOOR_FRACTION * peak)
    degenerate = np.flatnonzero(s_eff <= 0.0)
    if degenerate.size:
        labels = [exog_names[j] if exog_names is not None else f"column {j}" for j in degenerate]
        raise ValueError(
            f"exog columns are identically zero over the estimation sample: {_format_names(labels)}. "
            "Only the first n_lags rows carry any signal, and those are consumed as initial conditions, "
            "so the coefficient never enters the likelihood. Drop the column or extend the sample."
        )
    return scale * np.outer(sigma, 1.0 / s_eff)


class VAR(ImpulsoBaseModel):
    """Immutable VAR model specification.

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
    """

    lags: int | Literal["aic", "bic", "hq"] = Field(...)
    max_lags: int | None = None
    prior: Literal["minnesota"] | Prior = "minnesota"
    volatility: Literal["constant", "sv"] | PyMCVolatilityProcess = "constant"
    exog_prior_scale: float = Field(100.0, gt=0)

    @model_validator(mode="after")
    def _validate_spec(self) -> Self:
        if self.max_lags is not None and isinstance(self.lags, int):
            raise ValueError("max_lags is only valid when lags is a selection criterion ('aic', 'bic', 'hq')")
        if isinstance(self.lags, int) and self.lags < 1:
            raise ValueError(f"lags must be >= 1, got {self.lags}")
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
        import pymc as pm

        from impulso._lag_selection import select_lag_order
        from impulso.fitted import FittedVAR

        if sampler is None:
            sampler = self._default_sampler()

        # Resolve lags
        if isinstance(self.lags, str):
            max_lags = self.max_lags or 12
            ic = select_lag_order(data, max_lags=max_lags)
            n_lags = getattr(ic, self.lags)
        else:
            n_lags = self.lags

        # Build prior arrays
        prior = self.resolved_prior
        n_vars = data.endog.shape[1]
        prior_params = prior.build_priors(n_vars=n_vars, n_lags=n_lags)

        # Build data matrices
        y = data.endog
        Y = y[n_lags:]
        X_parts = []
        for lag in range(1, n_lags + 1):
            X_parts.append(y[n_lags - lag : -lag])
        X_lag = np.hstack(X_parts)

        X_exog = data.exog[n_lags:] if data.exog is not None else None

        # OLS residuals seed per-variable SV priors. Constant-volatility adapters
        # ignore `data`; only stochastic adapters use it.
        if X_exog is not None:
            X_full = np.hstack([np.ones((Y.shape[0], 1)), X_lag, X_exog])
        else:
            X_full = np.hstack([np.ones((Y.shape[0], 1)), X_lag])
        B_ols, *_ = np.linalg.lstsq(X_full, Y, rcond=None)
        resid = Y - X_full @ B_ols

        # Coordinates make the posterior self-describing: `B` comes back labelled
        # by variable and by "L<lag>.<variable>" coefficient instead of positional
        # `B_dim_0` / `B_dim_1`. Names match ConjugateVAR's posterior so both
        # estimators agree. `coeff` is lag-major to mirror the X_lag hstack above.
        coords: dict[str, object] = {
            "var": data.endog_names,
            "var1": data.endog_names,
            "var2": data.endog_names,
            "coeff": [f"L{lag}.{name}" for lag in range(1, n_lags + 1) for name in data.endog_names],
            "time": data.index[n_lags:],
        }
        if data.exog_names is not None:
            coords["exog"] = data.exog_names

        # Build PyMC model
        with pm.Model(coords=coords) as model:
            # Intercept
            intercept = pm.Normal("intercept", mu=0, sigma=1, dims="var")

            # VAR coefficients with Minnesota prior
            B = pm.Normal(
                "B",
                mu=prior_params["B_mu"],
                sigma=prior_params["B_sigma"],
                dims=("var", "coeff"),
            )

            # Exogenous coefficients. The prior scales with the data so that it
            # encodes the same belief regardless of the units the regressors
            # happen to be measured in (#192).
            if X_exog is not None:
                B_exog = pm.Normal(
                    "B_exog",
                    mu=0,
                    sigma=_exog_prior_sigma(y, X_exog, self.exog_prior_scale, data.exog_names),
                    dims=("var", "exog"),
                )
                mu = intercept + pm.math.dot(X_lag, B.T) + pm.math.dot(X_exog, B_exog.T)
            else:
                mu = intercept + pm.math.dot(X_lag, B.T)

            # Volatility process: registers latent vars, returns L (Cholesky factor of Σ_t).
            # For constant volatility, L is (n_vars, n_vars) and time-invariant.
            # For stochastic volatility, L is (T, n_vars, n_vars) — per-t.
            volatility = self.resolved_volatility
            L = volatility.build_pymc_latent(n_vars=n_vars, T=Y.shape[0], data=resid)
            # Sigma deterministic is only registered for time-invariant L —
            # for SV, materialising (T, n, n) per draw is wasteful; users can
            # reconstruct per-t Σ via `volatility.cholesky_at(posterior, t)`.
            if L.ndim == 2:
                pm.Deterministic("Sigma", pm.math.dot(L, L.T), dims=("var1", "var2"))

            # Likelihood. PyMC handles batched chol natively: for 2D L, every
            # observation uses the same chol; for 3D L (T, n, n), each
            # observation t uses chol[t].
            pm.MvNormal("obs", mu=mu, chol=L, observed=Y, dims=("time", "var"))

        # Sample
        idata = sampler.sample(model)

        return FittedVAR.model_construct(
            idata=idata,
            n_lags=n_lags,
            data=data,
            var_names=data.endog_names,
            volatility=self.resolved_volatility,
            pymc_model=model,
        )
