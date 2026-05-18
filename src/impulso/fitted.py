"""FittedVAR — reduced-form posterior from Bayesian VAR estimation."""

from typing import TYPE_CHECKING

import arviz as az
import numpy as np
from pydantic import Field, computed_field

from impulso._base import ImpulsoBaseModel
from impulso.data import VARData
from impulso.protocols import IdentificationScheme, VolatilityProcess

if TYPE_CHECKING:
    from impulso.identified import IdentifiedVAR
    from impulso.results import ForecastResult


class FittedVAR(ImpulsoBaseModel):
    """Immutable container for a fitted (reduced-form) Bayesian VAR.

    Attributes:
        idata: ArviZ InferenceData with posterior draws.
        n_lags: Lag order used in estimation.
        data: Original VARData used for fitting.
        var_names: Names of endogenous variables.
        volatility: Volatility process used at fit time. Required;
            populated by VAR.fit from VAR.volatility (default at the
            spec level is "constant", which resolves to Constant()).
    """

    idata: az.InferenceData = Field(repr=False)
    n_lags: int
    data: VARData
    var_names: list[str]
    volatility: VolatilityProcess

    @computed_field
    @property
    def has_exog(self) -> bool:
        """Whether the model includes exogenous variables."""
        return self.data.exog is not None

    @property
    def coefficients(self) -> np.ndarray:
        """Posterior draws of B coefficient matrices."""
        return self.idata.posterior["B"].values

    @property
    def intercepts(self) -> np.ndarray:
        """Posterior draws of intercept vectors."""
        return self.idata.posterior["intercept"].values

    def sigma(self) -> np.ndarray:
        """Posterior draws of the structural-shock covariance Σ.

        Dispatches to the configured volatility adapter so the returned
        shape depends on whether Σ is time-invariant or time-varying:

        * Constant volatility — Σ is shared across time, so the result
          has shape ``(chains, draws, n_vars, n_vars)``.
        * Stochastic volatility — Σ_t evolves, so the result has shape
          ``(chains, draws, T, n_vars, n_vars)`` where ``T`` is the
          in-sample length after lag trimming. Callers needing a single
          slice should call ``volatility.cholesky_at(posterior, t)`` and
          square the factor themselves.

        Note:
            **Breaking change vs. v0.0.4 and earlier**: ``sigma`` is now
            a method, not a property. Call sites that used ``fitted.sigma``
            must be updated to ``fitted.sigma()``.

        Returns:
            Posterior draws of Σ (or Σ_t for SV) computed from the
            volatility adapter's Cholesky factor as ``L @ L.T``.
        """
        if self.volatility.is_time_varying:
            T = self.data.endog.shape[0] - self.n_lags
            L_path = self.volatility.cholesky_path(self.idata.posterior, T=T)
            # Σ_t = L_t @ L_t.T, broadcast over (chains, draws, T).
            return np.einsum("cdtij,cdtkj->cdtik", L_path, L_path)
        L = self.volatility.cholesky_at(self.idata.posterior, t=None)
        # Σ = L @ L.T, broadcast over (chains, draws).
        return np.einsum("cdij,cdkj->cdik", L, L)

    def forecast(
        self,
        steps: int,
        exog_future: np.ndarray | None = None,
    ) -> "ForecastResult":
        """Produce h-step-ahead forecasts from the reduced-form posterior.

        Args:
            steps: Number of forecast steps.
            exog_future: Future exogenous values, shape (steps, k). Required if model has exog.

        Returns:
            ForecastResult with posterior forecast draws.
        """
        import xarray as xr

        from impulso.results import ForecastResult

        if self.has_exog and exog_future is None:
            raise ValueError("exog_future is required when model includes exogenous variables")
        if not self.has_exog and exog_future is not None:
            raise ValueError("exog_future provided but model has no exogenous variables")

        B_draws = self.coefficients  # (C, D, n_vars, n_vars*n_lags)
        intercept_draws = self.intercepts  # (C, D, n_vars)
        n_chains, n_draws, n_vars, _ = B_draws.shape

        # Last n_lags observations — broadcast to (C, D, n_lags, n)
        y_hist = self.data.endog[-self.n_lags :]  # (p, n)
        y_buffer = np.broadcast_to(y_hist, (n_chains, n_draws, self.n_lags, n_vars)).copy()

        forecasts = np.zeros((n_chains, n_draws, steps, n_vars))

        for h in range(steps):
            # Build lag vector: concatenate y[t-1], y[t-2], ..., y[t-p]
            x_lag = np.concatenate(
                [y_buffer[:, :, -(lag + 1), :] for lag in range(self.n_lags)], axis=-1
            )  # (C, D, n*p)

            y_new = intercept_draws + np.einsum("cdij,cdj->cdi", B_draws, x_lag)

            if self.has_exog and exog_future is not None:
                B_exog = self.idata.posterior["B_exog"].values  # (C, D, n, k)
                y_new = y_new + np.einsum("cdij,j->cdi", B_exog, exog_future[h])

            forecasts[:, :, h, :] = y_new
            # Roll buffer forward
            y_buffer = np.concatenate([y_buffer[:, :, 1:, :], y_new[:, :, np.newaxis, :]], axis=2)

        # Package into InferenceData
        forecast_da = xr.DataArray(
            forecasts,
            dims=["chain", "draw", "step", "variable"],
            coords={"variable": self.var_names},
            name="forecast",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"forecast": forecast_da}))

        return ForecastResult(idata=idata, steps=steps, var_names=self.var_names)

    def set_identification_strategy(self, scheme: IdentificationScheme) -> "IdentifiedVAR":
        """Apply a structural identification scheme.

        Queries the fitted volatility process for the Cholesky factor of Σ
        and passes it to the scheme. For constant volatility, the factor is
        the same across all time points; for stochastic volatility (P3),
        ``cholesky_at(t=None)`` returns the most-recent slice and
        downstream IRF/FEVD/HD methods can re-query at other ``at`` values.

        Args:
            scheme: An IdentificationScheme protocol instance (e.g. Cholesky,
                SignRestriction).

        Returns:
            IdentifiedVAR with structural_shock_matrix in the posterior.
        """
        import xarray as xr

        from impulso.identified import IdentifiedVAR

        L = self.volatility.cholesky_at(self.idata.posterior, t=None)
        P = scheme.identify(L, self.var_names, posterior=self.idata.posterior)

        # Wrap into a DataArray with named coords.
        shock_coords = scheme.shock_coords(n_vars=len(self.var_names))
        P_da = xr.DataArray(
            P,
            dims=["chain", "draw", "response", "shock"],
            coords={
                "response": self.var_names,
                "shock": shock_coords,
            },
        )

        new_posterior = self.idata.posterior.assign(structural_shock_matrix=P_da)

        # Carry through SignRestriction's acceptance rate if present.
        acceptance_rate = getattr(scheme, "_last_acceptance_rate", None)
        if isinstance(acceptance_rate, float) and acceptance_rate < 1.0:
            new_posterior.attrs["sign_restriction_acceptance_rate"] = acceptance_rate

        identified_idata = az.InferenceData(posterior=new_posterior)
        return IdentifiedVAR.model_construct(
            idata=identified_idata,
            n_lags=self.n_lags,
            data=self.data,
            var_names=self.var_names,
            volatility=self.volatility,
            scheme=scheme,
        )
