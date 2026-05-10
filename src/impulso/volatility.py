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
        sigma_sd_beta: HalfCauchy scale on diagonal SDs.
        tril_offdiag_sigma: Normal SD on off-diagonal correlation factors.
    """

    name: Literal["constant"] = "constant"

    sigma_sd_beta: float = Field(2.5, gt=0)
    tril_offdiag_sigma: float = Field(0.5, gt=0)

    def build_pymc_latent(self, n_vars: int, T: int) -> "pt.TensorVariable":
        raise NotImplementedError  # Task 3

    def cholesky_at(self, posterior: "xr.Dataset", t: int | None) -> np.ndarray:
        raise NotImplementedError  # Task 7

    def forecast_cholesky_path(
        self,
        posterior: "xr.Dataset",
        steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError  # Task 8
