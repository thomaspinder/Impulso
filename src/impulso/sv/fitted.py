"""FittedSV — posterior container for a fitted univariate SV model."""

from typing import TYPE_CHECKING

import arviz as az
import numpy as np
from pydantic import Field

from impulso._base import ImpulsoBaseModel
from impulso.sv.data import SVData
from impulso.sv.dynamics import SVDynamics

if TYPE_CHECKING:
    from impulso.results import SVForecastResult, VolatilityResult


class FittedSV(ImpulsoBaseModel):
    """Immutable container for a fitted univariate SV model.

    Attributes:
        idata: ArviZ InferenceData with posterior draws of h, mu, sigma_eta.
        data: Original SVData used for fitting.
        dynamics: Resolved SVDynamics instance used for fitting.
    """

    idata: az.InferenceData = Field(repr=False)
    data: SVData
    dynamics: SVDynamics

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

    def forecast(
        self,
        steps: int,
        random_seed: int | np.random.Generator | None = None,
    ) -> "SVForecastResult":
        """Density forecast of y_{T+h} integrating over h_{T+h} uncertainty.

        The log-volatility path is extrapolated via the fitted dynamics
        (random walk or AR(1)), then combined with the posterior of mu
        and a fresh observation-equation innovation to produce draws of
        y_{T+h} = mu + exp(h_{T+h}/2) * epsilon_{T+h}.

        Args:
            steps: Number of forecast steps.
            random_seed: Controls the RNG used to draw forecast innovations.
                Accepts an int seed, a ``numpy.random.Generator``, or ``None``
                (default) for fresh non-deterministic draws. Pass an int or
                ``Generator`` for reproducible forecasts.

        Returns:
            SVForecastResult wrapping posterior predictive draws.
        """
        import xarray as xr

        from impulso.results import SVForecastResult

        rng = np.random.default_rng(random_seed)
        posterior = self.idata.posterior

        h_forecast = self.dynamics.forecast_log_vol(posterior, steps, rng)
        mu_draws = posterior["mu"].values
        n_chains, n_draws, _ = h_forecast.shape
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
