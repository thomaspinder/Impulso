"""FittedSV — posterior container for a fitted univariate SV model."""

from typing import TYPE_CHECKING, Literal

import arviz as az
import numpy as np
from pydantic import Field

from impulso._base import ImpulsoBaseModel
from impulso.sv.data import SVData

if TYPE_CHECKING:
    from impulso.results import SVForecastResult, VolatilityResult


class FittedSV(ImpulsoBaseModel):
    """Immutable container for a fitted univariate SV model.

    Attributes:
        idata: ArviZ InferenceData with posterior draws of h, mu, sigma_eta.
        data: Original SVData used for fitting.
        dynamics: Log-vol dynamics used ("random_walk" or "ar1").
    """

    idata: az.InferenceData = Field(repr=False)
    data: SVData
    dynamics: Literal["random_walk", "ar1"]

    @property
    def log_volatility(self) -> np.ndarray:
        """Posterior draws of log-volatility path. Shape (chains, draws, T)."""
        return self.idata.posterior["h"].values

    def volatility(self) -> "VolatilityResult":
        """Posterior of conditional SD sigma_t = exp(h_t / 2)."""
        from impulso.results import VolatilityResult

        return VolatilityResult(
            idata=self.idata,
            series_name=self.data.name,
            index=self.data.index,
        )

    def forecast(self, steps: int) -> "SVForecastResult":
        """Density forecast of y_{T+h} integrating over h_{T+h} uncertainty.

        For random-walk dynamics:
            h_{T+h} = h_T + sum_{s=1..h} sigma_eta * eta_s
        For AR(1) dynamics:
            h_{T+h} = alpha + phi^h * (h_T - alpha) + innovation sum

        y_{T+h} = mu + exp(h_{T+h}/2) * epsilon_{T+h}

        Args:
            steps: Number of forecast steps.

        Returns:
            SVForecastResult wrapping posterior predictive draws.
        """
        import xarray as xr

        from impulso.results import SVForecastResult

        rng = np.random.default_rng(0)
        posterior = self.idata.posterior

        h_draws = posterior["h"].values  # (C, D, T)
        mu_draws = posterior["mu"].values  # (C, D)
        sigma_eta_draws = posterior["sigma_eta"].values  # (C, D)
        n_chains, n_draws, _ = h_draws.shape
        h_last = h_draws[:, :, -1]  # (C, D)

        h_forecast = np.zeros((n_chains, n_draws, steps))
        if self.dynamics == "random_walk":
            # Cumulative-sum of innovations, scaled per-draw
            innov = rng.standard_normal((n_chains, n_draws, steps))
            h_increments = sigma_eta_draws[:, :, None] * innov
            h_forecast = h_last[:, :, None] + np.cumsum(h_increments, axis=-1)
        else:  # ar1
            phi_draws = posterior["phi"].values  # (C, D)
            alpha_draws = posterior["alpha"].values  # (C, D)
            h_prev = h_last.copy()
            for s in range(steps):
                innov = rng.standard_normal((n_chains, n_draws))
                h_cur = alpha_draws + phi_draws * (h_prev - alpha_draws) + sigma_eta_draws * innov
                h_forecast[:, :, s] = h_cur
                h_prev = h_cur

        eps = rng.standard_normal((n_chains, n_draws, steps))
        y_forecast = mu_draws[:, :, None] + np.exp(0.5 * h_forecast) * eps

        forecast_da = xr.DataArray(
            y_forecast,
            dims=["chain", "draw", "step"],
            name="forecast",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"forecast": forecast_da}))
        return SVForecastResult(
            idata=idata,
            series_name=self.data.name,
            steps=steps,
        )
