"""VAR model specification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

import numpy as np
from impulso.data import VARData
from impulso.priors import MinnesotaPrior
from impulso.protocols import Prior, Sampler
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from impulso.fitted import FittedVAR

_PRIOR_REGISTRY: dict[str, type] = {
    "minnesota": MinnesotaPrior,
}


class VAR(BaseModel):
    """Immutable VAR model specification.

    Attributes:
        lags: Fixed lag order (int >= 1) or selection criterion string.
        max_lags: Upper bound for automatic selection. Only valid with string lags.
        prior: Prior shorthand string or Prior protocol instance.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    lags: int | Literal["aic", "bic", "hq"] = Field(...)
    max_lags: int | None = None
    prior: Union[Literal["minnesota"], Prior] = "minnesota"  # noqa: UP007

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

    def fit(
        self,
        data: VARData,
        sampler: Sampler | None = None,
    ) -> FittedVAR:
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

            # Residual covariance (Cholesky parameterization)
            import pytensor.tensor as pt

            sd = pm.HalfCauchy("sigma_sd", beta=2.5, shape=n_vars)
            n_tril = n_vars * (n_vars - 1) // 2
            L = pt.zeros((n_vars, n_vars))
            L = pt.set_subtensor(L[np.diag_indices(n_vars)], sd)
            if n_tril > 0:
                tril_vals = pm.Normal("tril_offdiag", mu=0, sigma=0.5, shape=n_tril)
                idx = 0
                for i in range(1, n_vars):
                    for j in range(i):
                        L = pt.set_subtensor(L[i, j], tril_vals[idx] * sd[i])
                        idx += 1
            chol = L
            pm.Deterministic("Sigma", pm.math.dot(chol, chol.T))

            # Likelihood
            pm.MvNormal("obs", mu=mu, chol=chol, observed=Y)

        # Sample
        idata = sampler.sample(model)

        return FittedVAR(
            idata=idata,
            n_lags=n_lags,
            data=data,
            var_names=data.endog_names,
        )
