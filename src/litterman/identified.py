"""IdentifiedVAR — structurally identified VAR model."""

from __future__ import annotations

import arviz as az
from pydantic import BaseModel, ConfigDict

from litterman.data import VARData


class IdentifiedVAR(BaseModel):
    """Immutable structural VAR with identified shocks.

    Placeholder — full implementation in phase-4-identification.

    Attributes:
        idata: InferenceData with structural_shock_matrix in posterior.
        n_lags: Lag order.
        data: Original VARData.
        var_names: Endogenous variable names.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    idata: az.InferenceData
    n_lags: int
    data: VARData
    var_names: list[str]
