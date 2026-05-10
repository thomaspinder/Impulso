"""Volatility processes for the VAR pipeline.

Defines concrete adapters of the VolatilityProcess Protocol declared in
protocols.py. The constant adapter (Constant) holds today's homoscedastic
manual-Cholesky parameterisation; stochastic adapters live elsewhere
(StochasticVolatility in impulso.sv) and arrive in later phases.

See docs/adr/0001-volatility-process-seam-exposes-cholesky-factor.md.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field

from impulso._base import ImpulsoModel

if TYPE_CHECKING:
    import pytensor.tensor as pt
    import xarray as xr


class Constant(ImpulsoModel):
    """Homoscedastic volatility — single Σ shared across all time points.

    Lifts today's manual-Cholesky parameterisation from
    ``spec.py:_build_pymc_model`` into the volatility-process seam:
    HalfCauchy(beta=sigma_sd_beta) on the diagonal scales,
    Normal(mu=0, sigma=tril_offdiag_sigma) on the lower-triangular
    off-diagonals (scaled by the row's diagonal). For ``n_vars == 1``
    the off-diagonal block is empty.

    The PyMC variable names produced inside ``build_pymc_latent``
    (``sigma_sd``, ``tril_offdiag``, ``Sigma``) match today's posterior
    contents exactly so existing identification and downstream code
    keep working unchanged.

    Attributes:
        name: Discriminator key for the registry (always ``"constant"``).
        sigma_sd_beta: HalfCauchy scale on diagonal SDs.
        tril_offdiag_sigma: Normal SD on off-diagonal correlation factors.
    """

    name: Literal["constant"] = "constant"

    sigma_sd_beta: float = Field(2.5, gt=0)
    tril_offdiag_sigma: float = Field(0.5, gt=0)

    def build_pymc_latent(self, n_vars: int, T: int) -> "pt.TensorVariable":
        """Register the constant-volatility latent vars in the active PyMC model.

        Lifts the manual-Cholesky parameterisation from the previous
        location in ``spec.py:_build_pymc_model``. PyMC variable names
        (``sigma_sd``, ``tril_offdiag``) match the prior contents byte-for-byte
        so existing posterior-consuming code keeps working unchanged.

        Args:
            n_vars: Number of endogenous variables.
            T: Number of observations after lag trimming. Ignored for
                constant volatility — kept in the signature for parity
                with stochastic adapters.

        Returns:
            Lower-triangular Cholesky factor L of shape (n_vars, n_vars).
        """
        import pymc as pm
        import pytensor.tensor as pt

        sd = pm.HalfCauchy("sigma_sd", beta=self.sigma_sd_beta, shape=n_vars)
        n_tril = n_vars * (n_vars - 1) // 2
        L = pt.zeros((n_vars, n_vars))
        L = pt.set_subtensor(L[np.diag_indices(n_vars)], sd)
        if n_tril > 0:
            tril_vals = pm.Normal("tril_offdiag", mu=0, sigma=self.tril_offdiag_sigma, shape=n_tril)
            idx = 0
            for i in range(1, n_vars):
                for j in range(i):
                    L = pt.set_subtensor(L[i, j], tril_vals[idx] * sd[i])
                    idx += 1
        return L

    def cholesky_at(self, posterior: "xr.Dataset", t: int | None) -> np.ndarray:
        raise NotImplementedError  # Task 7

    def forecast_cholesky_path(
        self,
        posterior: "xr.Dataset",
        steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError  # Task 8
