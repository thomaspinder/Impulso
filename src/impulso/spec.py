"""VAR model specification."""

from typing import TYPE_CHECKING, Any, Literal, Self

import arviz as az
import numpy as np
from pydantic import Field, model_validator

from impulso._base import ImpulsoBaseModel
from impulso.data import VARData
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


class VAR(ImpulsoBaseModel):
    """Immutable VAR model specification.

    Attributes:
        lags: Fixed lag order (int >= 1) or selection criterion string.
        max_lags: Upper bound for automatic selection. Only valid with string lags.
        prior: Prior shorthand string or Prior protocol instance.
        volatility: Volatility shorthand string or PyMCVolatilityProcess protocol instance.
    """

    lags: int | Literal["aic", "bic", "hq"] = Field(...)
    max_lags: int | None = None
    prior: Literal["minnesota"] | Prior = "minnesota"
    volatility: Literal["constant", "sv"] | PyMCVolatilityProcess = "constant"

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
        from impulso.fitted import FittedVAR

        if sampler is None:
            sampler = self._default_sampler()

        model, n_lags = self._build_pymc_model(data)

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

    def prior_predictive(
        self,
        data: VARData,
        *,
        draws: int = 500,
        random_seed: int | np.random.Generator | None = None,
    ) -> az.InferenceData:
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
            ArviZ InferenceData with `prior` (every latent), `prior_predictive`
            (the simulated `obs`, dims `(chain, draw, time, var)`) and
            `observed_data` (the realised `obs`) groups.
        """
        import pymc as pm

        model, _ = self._build_pymc_model(data)
        with model:
            return pm.sample_prior_predictive(draws=draws, random_seed=random_seed)

    def _build_pymc_model(self, data: VARData) -> tuple[Any, int]:
        """Build the PyMC model graph for this specification.

        Resolves the lag order (running `select_lag_order` when `lags` is a
        criterion string), assembles the design matrices, and registers the
        intercept, coefficient, volatility and likelihood nodes. The design
        matrices are baked into the graph as constants, so the returned model
        is tied to `data`.

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

            # Exogenous coefficients
            if X_exog is not None:
                B_exog = pm.Normal("B_exog", mu=0, sigma=1, dims=("var", "exog"))
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

        return model, n_lags
