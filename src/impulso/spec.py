"""VAR model specification."""

from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from impulso._base import ImpulsoBaseModel
from impulso.data import VARData
from impulso.priors import MinnesotaPrior
from impulso.protocols import Prior, Sampler, VolatilityProcess
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


class VAR(ImpulsoBaseModel):
    """Immutable VAR model specification.

    Attributes:
        lags: Fixed lag order (int >= 1) or selection criterion string.
        max_lags: Upper bound for automatic selection. Only valid with string lags.
        prior: Prior shorthand string or Prior protocol instance.
        volatility: Volatility shorthand string or VolatilityProcess protocol instance.
    """

    lags: int | Literal["aic", "bic", "hq"] = Field(...)
    max_lags: int | None = None
    prior: Literal["minnesota"] | Prior = "minnesota"
    volatility: Literal["constant", "sv"] | VolatilityProcess = "constant"

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
    def resolved_volatility(self) -> VolatilityProcess:
        """Resolve string volatility shorthand to a VolatilityProcess instance."""
        if isinstance(self.volatility, str):
            return _VOLATILITY_REGISTRY[self.volatility]()
        return self.volatility

    def fit(
        self,
        data: VARData,
        sampler: Sampler | None = None,
    ) -> "FittedVAR":
        """Estimate the Bayesian VAR model.

        Args:
            data: VARData instance.
            sampler: Sampler protocol instance. Defaults to NUTSSampler().

        Returns:
            FittedVAR with posterior draws.
        """
        import pymc as pm

        from impulso._lag_selection import select_lag_order
        from impulso.fitted import FittedVAR
        from impulso.samplers import NUTSSampler

        if sampler is None:
            sampler = NUTSSampler()

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

        # Build PyMC model
        with pm.Model() as model:
            # Intercept
            intercept = pm.Normal("intercept", mu=0, sigma=1, shape=n_vars)

            # VAR coefficients with Minnesota prior
            B = pm.Normal(
                "B",
                mu=prior_params["B_mu"],
                sigma=prior_params["B_sigma"],
                shape=(n_vars, n_vars * n_lags),
            )

            # Exogenous coefficients
            if X_exog is not None:
                n_exog = X_exog.shape[1]
                B_exog = pm.Normal("B_exog", mu=0, sigma=1, shape=(n_vars, n_exog))
                mu = intercept + pm.math.dot(X_lag, B.T) + pm.math.dot(X_exog, B_exog.T)
            else:
                mu = intercept + pm.math.dot(X_lag, B.T)

            # Volatility process: registers latent vars, returns L (Cholesky factor of Σ_t).
            # For constant volatility, L is (n_vars, n_vars) and time-invariant.
            # For stochastic volatility, L is (T, n_vars, n_vars) — per-t.
            volatility = self.resolved_volatility
            L = volatility.build_pymc_latent(n_vars=n_vars, T=Y.shape[0])
            # Sigma deterministic is only registered for time-invariant L —
            # for SV, materialising (T, n, n) per draw is wasteful; users can
            # reconstruct per-t Σ via `volatility.cholesky_at(posterior, t)`.
            if L.ndim == 2:
                pm.Deterministic("Sigma", pm.math.dot(L, L.T))

            # Likelihood. PyMC handles batched chol natively: for 2D L, every
            # observation uses the same chol; for 3D L (T, n, n), each
            # observation t uses chol[t].
            pm.MvNormal("obs", mu=mu, chol=L, observed=Y)

        # Sample
        idata = sampler.sample(model)

        return FittedVAR.model_construct(
            idata=idata,
            n_lags=n_lags,
            data=data,
            var_names=data.endog_names,
            volatility=self.resolved_volatility,
        )
