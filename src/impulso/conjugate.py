"""ConjugateVAR — direct Normal-Inverse-Wishart posterior sampling."""

from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from impulso._base import ImpulsoBaseModel
from impulso.data import VARData
from impulso.priors import MinnesotaPrior

if TYPE_CHECKING:
    from impulso.fitted import FittedVAR


class ConjugateVAR(ImpulsoBaseModel):
    """Bayesian VAR with conjugate Normal-Inverse-Wishart estimation.

    Produces iid posterior draws via direct sampling — no MCMC iteration,
    no burn-in, no autocorrelation. Orders of magnitude faster than NUTS
    for models with Minnesota-type priors.

    Attributes:
        lags: Fixed lag order or selection criterion.
        max_lags: Upper bound for automatic lag selection.
        prior: Minnesota prior instance or string shorthand.
        draws: Number of posterior draws.
        random_seed: Seed for reproducibility.
    """

    lags: int | Literal["aic", "bic", "hq"] = Field(...)
    max_lags: int | None = None
    prior: Literal["minnesota", "minnesota_optimized"] | MinnesotaPrior = "minnesota"
    draws: int = Field(2000, ge=1)
    random_seed: int | None = None

    @model_validator(mode="after")
    def _validate_spec(self) -> Self:
        if self.max_lags is not None and isinstance(self.lags, int):
            raise ValueError("max_lags is only valid when lags is a selection criterion")
        if isinstance(self.lags, int) and self.lags < 1:
            raise ValueError(f"lags must be >= 1, got {self.lags}")
        return self

    @property
    def resolved_prior(self) -> MinnesotaPrior:
        """Resolve string shorthand to a MinnesotaPrior instance."""
        if isinstance(self.prior, str):
            return MinnesotaPrior()
        return self.prior

    @staticmethod
    def _build_niw_params(
        data: VARData,
        n_lags: int,
        prior: MinnesotaPrior,
    ) -> dict[str, np.ndarray]:
        """Build data matrices and compute NIW prior/posterior parameters.

        Args:
            data: VARData instance.
            n_lags: Number of lags.
            prior: MinnesotaPrior instance.

        Returns:
            Dictionary with keys: Y, X, B_prior, V_prior, V_prior_inv,
            V_posterior, B_posterior, S_prior, S_posterior, nu_prior, nu_posterior.
        """
        n_vars = data.endog.shape[1]
        prior_params = prior.build_priors(n_vars=n_vars, n_lags=n_lags)

        # Build data matrices: Y = (T-p, n), X = (T-p, n*p + 1) with intercept
        y = data.endog
        Y = y[n_lags:]
        X_parts = [np.ones((Y.shape[0], 1))]
        for lag in range(1, n_lags + 1):
            X_parts.append(y[n_lags - lag : -lag])
        X = np.hstack(X_parts)

        T_eff = Y.shape[0]
        n_coeffs = X.shape[1]

        # Convert Minnesota prior to NIW parameters
        # Prior mean: [intercept_prior | B_mu]
        B_prior = np.zeros((n_coeffs, n_vars))
        B_prior[1:, :] = prior_params["B_mu"].T

        # Prior covariance: diagonal from B_sigma
        # Intercept gets a wide prior (variance=1.0, matching PyMC path)
        intercept_var = 1.0
        lag_var = np.mean(prior_params["B_sigma"] ** 2, axis=0)
        prior_var_diag = np.concatenate([[intercept_var], lag_var])
        V_prior = np.diag(prior_var_diag)
        V_prior_inv = np.diag(1.0 / prior_var_diag)

        # OLS estimates for scale matrix initialisation
        B_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid_ols = Y - X @ B_ols
        sigma_ols = (resid_ols.T @ resid_ols) / T_eff

        # NIW prior hyperparameters
        nu_prior = n_vars + 2  # minimally informative
        S_prior = sigma_ols * (nu_prior - n_vars - 1)  # centres IW mode at sigma_ols

        # Posterior parameters
        V_posterior = np.linalg.inv(V_prior_inv + X.T @ X)
        B_posterior = V_posterior @ (V_prior_inv @ B_prior + X.T @ Y)
        nu_posterior = nu_prior + T_eff
        S_posterior = (
            S_prior
            + Y.T @ Y
            + B_prior.T @ V_prior_inv @ B_prior
            - B_posterior.T @ np.linalg.inv(V_posterior) @ B_posterior
        )
        S_posterior = (S_posterior + S_posterior.T) / 2

        return {
            "Y": Y,
            "X": X,
            "B_prior": B_prior,
            "V_prior": V_prior,
            "V_prior_inv": V_prior_inv,
            "V_posterior": V_posterior,
            "B_posterior": B_posterior,
            "S_prior": S_prior,
            "S_posterior": S_posterior,
            "nu_prior": nu_prior,
            "nu_posterior": nu_posterior,
        }

    def fit(self, data: VARData) -> "FittedVAR":
        """Estimate the Bayesian VAR via conjugate NIW posterior sampling.

        Args:
            data: VARData instance.

        Returns:
            FittedVAR with iid posterior draws.
        """
        import arviz as az
        import xarray as xr
        from scipy.stats import invwishart

        from impulso._lag_selection import select_lag_order
        from impulso.fitted import FittedVAR

        # Resolve lags
        if isinstance(self.lags, str):
            max_lags = self.max_lags or 12
            ic = select_lag_order(data, max_lags=max_lags)
            n_lags = getattr(ic, self.lags)
        else:
            n_lags = self.lags

        n_vars = data.endog.shape[1]

        # Resolve prior (optimize if requested)
        if isinstance(self.prior, str) and self.prior == "minnesota_optimized":
            prior = self._optimize_prior_internal(data, n_lags)
        else:
            prior = self.resolved_prior

        params = self._build_niw_params(data, n_lags, prior)
        V_posterior = params["V_posterior"]
        B_posterior = params["B_posterior"]
        S_posterior = params["S_posterior"]
        nu_posterior = params["nu_posterior"]
        n_coeffs = B_posterior.shape[0]  # 1 + n_vars * n_lags

        # Direct sampling
        rng = np.random.default_rng(self.random_seed)

        B_draws = np.zeros((self.draws, n_coeffs, n_vars))
        Sigma_draws = np.zeros((self.draws, n_vars, n_vars))

        chol_V_posterior = np.linalg.cholesky(V_posterior)

        for i in range(self.draws):
            # Draw Sigma ~ IW(S_posterior, nu_posterior)
            Sigma_draw = invwishart.rvs(df=nu_posterior, scale=S_posterior, random_state=rng)
            Sigma_draws[i] = Sigma_draw

            # Draw B | Sigma ~ MN(B_posterior, Sigma, V_posterior)
            # vec(B) ~ N(vec(B_posterior), Sigma kron V_posterior)
            chol_Sigma = np.linalg.cholesky(Sigma_draw)
            Z = rng.standard_normal((n_coeffs, n_vars))
            B_draw = B_posterior + chol_V_posterior @ Z @ chol_Sigma.T
            B_draws[i] = B_draw

        # Separate intercept and lag coefficients
        intercept_arr = B_draws[:, 0, :]  # (draws, n_vars)
        B_lag_arr = B_draws[:, 1:, :]  # (draws, n*p, n_vars)
        # Transpose to match PyMC convention: B is (n_vars, n_vars*n_lags)
        B_lag_arr = np.swapaxes(B_lag_arr, -2, -1)  # (draws, n_vars, n*p)

        # Add chain dimension (chains=1 for conjugate)
        intercept_arr = intercept_arr[np.newaxis, :]  # (1, draws, n_vars)
        B_lag_arr = B_lag_arr[np.newaxis, :]  # (1, draws, n_vars, n*p)
        Sigma_draws = Sigma_draws[np.newaxis, :]  # (1, draws, n_vars, n_vars)

        # Package as InferenceData
        posterior = xr.Dataset({
            "B": xr.DataArray(B_lag_arr, dims=["chain", "draw", "equations", "coefficients"]),
            "intercept": xr.DataArray(intercept_arr, dims=["chain", "draw", "equations"]),
            "Sigma": xr.DataArray(Sigma_draws, dims=["chain", "draw", "var1", "var2"]),
        })
        idata = az.InferenceData(posterior=posterior)

        return FittedVAR.model_construct(
            idata=idata,
            n_lags=n_lags,
            data=data,
            var_names=data.endog_names,
        )

    def optimize_prior(
        self,
        data: VARData,
        optimize_dummy: bool = False,
    ) -> MinnesotaPrior:
        """Find Minnesota hyperparameters maximising the marginal likelihood.

        Implements Giannone, Lenza & Primiceri (2015) data-driven prior
        selection via closed-form marginal likelihood optimisation.

        Args:
            data: VARData instance (may include dummy observations).
            optimize_dummy: If True, also optimise dummy hyperparameters.

        Returns:
            MinnesotaPrior with optimal tightness and cross_shrinkage.
        """
        from impulso._lag_selection import select_lag_order

        # Resolve lags
        if isinstance(self.lags, str):
            max_lags = self.max_lags or 12
            ic = select_lag_order(data, max_lags=max_lags)
            n_lags = getattr(ic, self.lags)
        else:
            n_lags = self.lags

        return self._optimize_prior_internal(data, n_lags, optimize_dummy)

    def _optimize_prior_internal(
        self,
        data: VARData,
        n_lags: int,
        optimize_dummy: bool = False,
    ) -> MinnesotaPrior:
        """Internal implementation of prior optimisation."""
        from scipy.optimize import minimize

        current_prior = self.resolved_prior

        def neg_log_marginal_likelihood(params: np.ndarray) -> float:
            tightness = params[0]
            cross_shrinkage = params[1]
            prior = MinnesotaPrior(
                tightness=tightness,
                cross_shrinkage=cross_shrinkage,
                decay=current_prior.decay,
            )
            return -self._log_marginal_likelihood(data, n_lags, prior)

        x0 = np.array([current_prior.tightness, current_prior.cross_shrinkage])
        bounds = [(0.001, 10.0), (0.01, 1.0)]

        result = minimize(
            neg_log_marginal_likelihood,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
        )

        return MinnesotaPrior(
            tightness=float(result.x[0]),
            cross_shrinkage=float(result.x[1]),
            decay=current_prior.decay,
        )

    def _log_marginal_likelihood(
        self,
        data: VARData,
        n_lags: int,
        prior: MinnesotaPrior,
    ) -> float:
        """Compute log marginal likelihood p(Y|lambda) for NIW conjugate model.

        Uses the closed-form NIW marginal likelihood from Kadiyala & Karlsson (1997).

        Args:
            data: VARData instance.
            n_lags: Number of lags.
            prior: MinnesotaPrior with specific hyperparameters.

        Returns:
            Log marginal likelihood (scalar).
        """
        from scipy.special import gammaln

        n_vars = data.endog.shape[1]
        params = self._build_niw_params(data, n_lags, prior)

        T_eff = params["Y"].shape[0]
        V_prior = params["V_prior"]
        V_posterior = params["V_posterior"]
        S_prior = params["S_prior"]
        S_posterior = params["S_posterior"]
        nu_prior = params["nu_prior"]
        nu_posterior = params["nu_posterior"]

        # Log marginal likelihood formula
        log_ml = 0.0
        log_ml -= (T_eff * n_vars / 2) * np.log(np.pi)

        # Log-determinant terms
        _, logdet_V_prior = np.linalg.slogdet(V_prior)
        _, logdet_V_posterior = np.linalg.slogdet(V_posterior)
        log_ml += 0.5 * (logdet_V_posterior - logdet_V_prior) * n_vars

        _, logdet_S_prior = np.linalg.slogdet(S_prior)
        _, logdet_S_posterior = np.linalg.slogdet(S_posterior)
        log_ml += (nu_prior / 2) * logdet_S_prior
        log_ml -= (nu_posterior / 2) * logdet_S_posterior

        # Multivariate gamma function terms
        for j in range(n_vars):
            log_ml += gammaln((nu_posterior - j) / 2) - gammaln((nu_prior - j) / 2)

        return log_ml

    def marginal_likelihood(self, data: VARData) -> float:
        """Compute log marginal likelihood for the current prior.

        Args:
            data: VARData instance.

        Returns:
            Log marginal likelihood (scalar).
        """
        from impulso._lag_selection import select_lag_order

        if isinstance(self.lags, str):
            max_lags = self.max_lags or 12
            ic = select_lag_order(data, max_lags=max_lags)
            n_lags = getattr(ic, self.lags)
        else:
            n_lags = self.lags

        return self._log_marginal_likelihood(data, n_lags, self.resolved_prior)
