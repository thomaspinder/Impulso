"""FittedVAR — reduced-form posterior from Bayesian VAR estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import arviz as az
import numpy as np
from pydantic import BaseModel, ConfigDict

from impulso.data import VARData
from impulso.protocols import IdentificationScheme

if TYPE_CHECKING:
    from impulso.identified import IdentifiedVAR
    from impulso.results import ForecastResult


class FittedVAR(BaseModel):
    """Immutable container for a fitted (reduced-form) Bayesian VAR.

    Attributes:
        idata: ArviZ InferenceData with posterior draws.
        n_lags: Lag order used in estimation.
        data: Original VARData used for fitting.
        var_names: Names of endogenous variables.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    idata: az.InferenceData
    n_lags: int
    data: VARData
    var_names: list[str]

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

    @property
    def sigma(self) -> np.ndarray:
        """Posterior draws of residual covariance matrix."""
        return self.idata.posterior["Sigma"].values

    def forecast(
        self,
        steps: int,
        exog_future: np.ndarray | None = None,
    ) -> ForecastResult:
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

        B_draws = self.coefficients  # (chains, draws, n_vars, n_vars*n_lags)
        intercept_draws = self.intercepts  # (chains, draws, n_vars)
        n_chains, n_draws, n_vars, _ = B_draws.shape

        # Last n_lags observations for initial conditions
        y_hist = self.data.endog[-self.n_lags :]  # (n_lags, n_vars)

        forecasts = np.zeros((n_chains, n_draws, steps, n_vars))

        for c in range(n_chains):
            for d in range(n_draws):
                B = B_draws[c, d]
                intercept = intercept_draws[c, d]
                y_buffer = list(y_hist)

                for h in range(steps):
                    x_lag = np.concatenate([y_buffer[-(lag)] for lag in range(1, self.n_lags + 1)])
                    y_new = intercept + B @ x_lag
                    if self.has_exog and exog_future is not None:
                        B_exog = self.idata.posterior["B_exog"].values[c, d]
                        y_new = y_new + B_exog @ exog_future[h]
                    forecasts[c, d, h] = y_new
                    y_buffer.append(y_new)

        # Package into InferenceData
        forecast_da = xr.DataArray(
            forecasts,
            dims=["chain", "draw", "step", "variable"],
            coords={"variable": self.var_names},
            name="forecast",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"forecast": forecast_da}))

        return ForecastResult(idata=idata, steps=steps, var_names=self.var_names)

    def set_identification_strategy(self, scheme: IdentificationScheme) -> IdentifiedVAR:
        """Apply a structural identification scheme.

        Args:
            scheme: An IdentificationScheme protocol instance (e.g. Cholesky).

        Returns:
            IdentifiedVAR with structural shock matrix in the posterior.
        """
        from impulso.identified import IdentifiedVAR

        identified_idata = scheme.identify(self.idata, self.var_names)
        return IdentifiedVAR(
            idata=identified_idata,
            n_lags=self.n_lags,
            data=self.data,
            var_names=self.var_names,
        )

    def __repr__(self) -> str:
        n_vars = len(self.var_names)
        posterior = self.idata.posterior
        n_chains = posterior.sizes["chain"]
        n_draws = posterior.sizes["draw"]
        return f"FittedVAR(n_lags={self.n_lags}, n_vars={n_vars}, n_draws={n_draws}, n_chains={n_chains})"
