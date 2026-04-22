"""Log-volatility dynamics for univariate stochastic volatility models."""

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import xarray as xr

from impulso._base import ImpulsoModel

if TYPE_CHECKING:
    import pytensor.tensor as pt


@runtime_checkable
class SVDynamics(Protocol):
    """Contract for SV log-volatility dynamics.

    Implementations own both the PyMC construction of the latent log-vol
    path and the forward simulation used for density forecasts.
    """

    name: str

    def build_latent_path(
        self,
        prior_params: dict,
        T: int,
        sigma_eta: "pt.TensorVariable",
    ) -> "pt.TensorVariable":
        """Register and return the latent log-volatility path inside the active PyMC model."""
        ...

    def forecast_log_vol(
        self,
        posterior: xr.Dataset,
        steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw posterior-predictive log-volatility paths, shape (chains, draws, steps)."""
        ...


class RandomWalk(ImpulsoModel):
    """Random-walk log-volatility dynamics (Primiceri 2005)."""

    name: ClassVar[str] = "random_walk"

    def build_latent_path(self, prior_params: dict, T: int, sigma_eta: Any) -> Any:
        import pymc as pm
        import pytensor.tensor as pt

        h0 = pm.Normal("h0", mu=prior_params["h0_mu"], sigma=prior_params["h0_sigma"])
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=T - 1)
        h_path = pt.concatenate([pt.as_tensor_variable([h0]), h0 + sigma_eta * pt.cumsum(z)])
        return pm.Deterministic("h", h_path)

    def forecast_log_vol(
        self,
        posterior: xr.Dataset,
        steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        h_draws = posterior["h"].values
        sigma_eta_draws = posterior["sigma_eta"].values
        n_chains, n_draws, _ = h_draws.shape
        h_last = h_draws[:, :, -1]

        innov = rng.standard_normal((n_chains, n_draws, steps))
        h_increments = sigma_eta_draws[:, :, None] * innov
        return h_last[:, :, None] + np.cumsum(h_increments, axis=-1)


class AR1(ImpulsoModel):
    """AR(1) log-volatility dynamics (Kim-Shephard-Chib 1998)."""

    name: ClassVar[str] = "ar1"

    def build_latent_path(self, prior_params: dict, T: int, sigma_eta: Any) -> Any:
        import pymc as pm
        import pytensor.tensor as pt

        phi = pm.Beta("phi", alpha=prior_params["phi_a"], beta=prior_params["phi_b"])
        alpha = pm.Normal("alpha", mu=prior_params["alpha_mu"], sigma=prior_params["alpha_sigma"])
        # Floor guards against phi^2 = 1 rounding producing NaN log-prob under Beta(20, 1.5).
        stationary_var = 1.0 / pm.math.maximum(1.0 - pt.mul(phi, phi), 1e-6)
        g_init = pm.Normal.dist(mu=0.0, sigma=pm.math.sqrt(stationary_var))
        g = pm.AR(
            "g",
            rho=pt.stack([pt.as_tensor_variable(0.0), phi]),
            sigma=1.0,
            constant=True,
            init_dist=g_init,
            shape=T,
        )
        return pm.Deterministic("h", alpha + pt.mul(sigma_eta, g))

    def forecast_log_vol(
        self,
        posterior: xr.Dataset,
        steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        h_draws = posterior["h"].values
        sigma_eta_draws = posterior["sigma_eta"].values
        phi_draws = posterior["phi"].values
        alpha_draws = posterior["alpha"].values
        n_chains, n_draws, _ = h_draws.shape
        h_prev = h_draws[:, :, -1].copy()

        h_forecast = np.zeros((n_chains, n_draws, steps))
        for s in range(steps):
            innov = rng.standard_normal((n_chains, n_draws))
            h_prev = alpha_draws + phi_draws * (h_prev - alpha_draws) + sigma_eta_draws * innov
            h_forecast[:, :, s] = h_prev
        return h_forecast


SV_DYNAMICS_REGISTRY: dict[str, type[SVDynamics]] = {
    RandomWalk.name: RandomWalk,
    AR1.name: AR1,
}
