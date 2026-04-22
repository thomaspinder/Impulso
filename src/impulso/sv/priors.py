"""Prior specifications for univariate stochastic volatility models."""

from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import Field

from impulso._base import ImpulsoModel


@runtime_checkable
class SVPrior(Protocol):
    """Contract for univariate SV prior specifications.

    Implementations build a dict of prior hyperparameters consumed by
    the StochasticVolatility spec's PyMC model.
    """

    def build_priors(self, y: np.ndarray) -> dict[str, float]: ...


class SVDefaultPrior(ImpulsoModel):
    """Default weakly-informative prior for univariate SV.

    Matches Primiceri (2005) / Cogley-Sargent (2005) conventions for
    random-walk log-volatility, plus AR(1) variants (Kim-Shephard-Chib 1998).

    Attributes:
        mu_scale: Scale multiplier for the prior SD on the level mu.
            Prior SD = max(mu_scale * sample_std, 1e-6). The floor
            guarantees a valid (strictly positive, finite) PyMC prior
            even if `build_priors` is called directly with a near-constant
            series. `SVData` rejects exactly-constant series at construction.
        h0_scale: Prior SD on the initial log-volatility h_0.
        sigma_eta_scale: HalfNormal scale for sigma_eta.
        phi_a: Beta(a, b) alpha parameter for phi (AR(1) only).
        phi_b: Beta(a, b) beta parameter for phi (AR(1) only).
        alpha_scale: Prior SD on alpha (AR(1) level, AR(1) only).
    """

    mu_scale: float = Field(10.0, gt=0)
    h0_scale: float = Field(1.0, gt=0)
    sigma_eta_scale: float = Field(0.2, gt=0)
    phi_a: float = Field(20.0, gt=0)
    phi_b: float = Field(1.5, gt=0)
    alpha_scale: float = Field(10.0, gt=0)

    def build_priors(self, y: np.ndarray) -> dict[str, float]:
        """Build prior hyperparameters from the observed series.

        Args:
            y: 1-D array of observations.

        Returns:
            Dict of prior hyperparameters keyed by name.
        """
        y_mean = float(np.mean(y))
        y_std = float(np.std(y, ddof=1))
        y_var = float(np.var(y, ddof=1))
        # Sanitize ddof=1 NaN on singleton inputs; downstream `max(..., floor)` does
        # not help because max(NaN, x) == NaN in Python.
        if not np.isfinite(y_std):
            y_std = 0.0
        if not np.isfinite(y_var):
            y_var = 0.0
        # Guard against log(0) when all observations are equal
        log_var = float(np.log(max(y_var, 1e-8)))
        # Guard mu_sigma: avoid 0 (constant y) or NaN (singleton y) reaching PyMC.
        # SVData already rejects these, but build_priors is public API.
        mu_sigma = max(self.mu_scale * y_std, 1e-6)

        return {
            "mu_mu": y_mean,
            "mu_sigma": mu_sigma,
            "h0_mu": log_var,
            "h0_sigma": self.h0_scale,
            "sigma_eta_scale": self.sigma_eta_scale,
            "phi_a": self.phi_a,
            "phi_b": self.phi_b,
            "alpha_mu": log_var,
            "alpha_sigma": self.alpha_scale,
        }
