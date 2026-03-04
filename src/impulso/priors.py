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

        # B_mu: identity on the first lag block, zero elsewhere
        B_mu = np.zeros((n_vars, n_coeffs))
        B_mu[np.arange(n_vars), np.arange(n_vars)] = 1.0

        # Lag decay per column: each lag's decay repeated n_vars times
        lags = np.arange(1, n_lags + 1)
        lag_decay = 1.0 / lags if self.decay == "harmonic" else 1.0 / lags**2
        decay_per_col = np.repeat(lag_decay, n_vars)  # (n_coeffs,)

        # Own vs cross mask: 1.0 on own-variable columns, cross_shrinkage elsewhere
        col_var = np.arange(n_coeffs) % n_vars
        is_own = col_var[np.newaxis, :] == np.arange(n_vars)[:, np.newaxis]
        cross_mask = np.where(is_own, 1.0, self.cross_shrinkage)

        B_sigma = self.tightness * decay_per_col[np.newaxis, :] * cross_mask

        return {"B_mu": B_mu, "B_sigma": B_sigma}
