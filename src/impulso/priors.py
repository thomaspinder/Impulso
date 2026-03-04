"""Prior specifications for VAR models."""

from typing import Literal

import numpy as np
from pydantic import Field

from impulso._base import ImpulsoModel


class MinnesotaPrior(ImpulsoModel):
    """Minnesota prior for VAR coefficient shrinkage.

    Attributes:
        tightness: Overall shrinkage toward prior mean. Must be > 0.
        decay: How coefficients shrink on longer lags.
        cross_shrinkage: Shrinkage on other variables' lags vs own. 0 = only own lags, 1 = equal.
    """

    tightness: float = Field(0.1, gt=0)
    decay: Literal["harmonic", "geometric"] = "harmonic"
    cross_shrinkage: float = Field(0.5, ge=0, le=1)

    def build_priors(self, n_vars: int, n_lags: int) -> dict[str, np.ndarray]:
        """Build prior mean and standard deviation arrays for VAR coefficients.

        Args:
            n_vars: Number of endogenous variables.
            n_lags: Number of lags.

        Returns:
            Dictionary with keys 'B_mu' and 'B_sigma' as numpy arrays.
        """
        n_coeffs = n_vars * n_lags
        B_mu = np.zeros((n_vars, n_coeffs))
        B_sigma = np.ones((n_vars, n_coeffs))

        for eq in range(n_vars):
            for lag in range(1, n_lags + 1):
                lag_decay = 1.0 / lag if self.decay == "harmonic" else 1.0 / (lag**2)

                for var in range(n_vars):
                    col = (lag - 1) * n_vars + var
                    if var == eq:
                        # Own lag: prior mean = 1.0 for first lag, 0 otherwise
                        if lag == 1:
                            B_mu[eq, col] = 1.0
                        B_sigma[eq, col] = self.tightness * lag_decay
                    else:
                        # Cross lag
                        B_sigma[eq, col] = self.tightness * lag_decay * self.cross_shrinkage

        return {"B_mu": B_mu, "B_sigma": B_sigma}
