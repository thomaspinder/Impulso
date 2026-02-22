"""FittedVAR — reduced-form posterior from Bayesian VAR estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import arviz as az
import numpy as np
from pydantic import BaseModel, ConfigDict

from litterman.data import VARData
from litterman.protocols import IdentificationScheme

if TYPE_CHECKING:
    from litterman.identified import IdentifiedVAR


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

    def set_identification_strategy(self, scheme: IdentificationScheme) -> IdentifiedVAR:
        """Apply a structural identification scheme.

        Args:
            scheme: An IdentificationScheme protocol instance (e.g. Cholesky).

        Returns:
            IdentifiedVAR with structural shock matrix in the posterior.
        """
        from litterman.identified import IdentifiedVAR

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
