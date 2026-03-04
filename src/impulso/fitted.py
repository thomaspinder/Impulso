"""FittedVAR — reduced-form posterior from Bayesian VAR estimation."""

from typing import TYPE_CHECKING

import arviz as az
import numpy as np
from pydantic import Field, computed_field

from impulso._base import ImpulsoBaseModel
from impulso.data import VARData
from impulso.protocols import IdentificationScheme

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
    """

    idata: az.InferenceData = Field(repr=False)
    n_lags: int
    data: VARData
    var_names: list[str]

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

    @property
    def sigma(self) -> np.ndarray:
        """Posterior draws of residual covariance matrix."""
        return self.idata.posterior["Sigma"].values

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

        Args:
            scheme: An IdentificationScheme protocol instance (e.g. Cholesky).

        Returns:
            IdentifiedVAR with structural shock matrix in the posterior.
        """
        from impulso.identified import IdentifiedVAR

        identified_idata = scheme.identify(self.idata, self.var_names)
        return IdentifiedVAR.model_construct(
            idata=identified_idata,
            n_lags=self.n_lags,
            data=self.data,
            var_names=self.var_names,
        )
