"""Protocol definitions for extensible components."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import arviz as az
    import pymc as pm


@runtime_checkable
class Prior(Protocol):
    """Contract for prior specifications."""

    def build_priors(self, n_vars: int, n_lags: int) -> dict[str, np.ndarray]: ...


@runtime_checkable
class Sampler(Protocol):
    """Contract for posterior sampling strategies."""

    def sample(self, model: "pm.Model") -> "az.InferenceData": ...


@runtime_checkable
class IdentificationScheme(Protocol):
    """Contract for structural identification schemes."""

    def identify(self, idata: "az.InferenceData", var_names: list[str]) -> "az.InferenceData": ...
