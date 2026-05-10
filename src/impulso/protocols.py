"""Protocol definitions for extensible components."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt
    import xarray as xr


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


@runtime_checkable
class VolatilityProcess(Protocol):
    """Contract for volatility processes.

    A VolatilityProcess owns the construction of the structural-shock
    covariance Σ_t — for constant adapters, Σ is shared across time;
    for stochastic adapters, Σ_t evolves. The seam's primary output is
    the lower-triangular Cholesky factor L_t such that Σ_t = L_t @ L_t.T.
    See docs/adr/0001-volatility-process-seam-exposes-cholesky-factor.md.

    Adapters own their downstream computation: time-`t` query and
    forward simulation for forecasts.
    """

    name: str

    def build_pymc_latent(self, n_vars: int, T: int) -> "pt.TensorVariable":
        """Register volatility latent variables in the active PyMC model.

        Returns the lower-triangular Cholesky factor as a PyTensor variable.
        For constant volatility, shape is (n_vars, n_vars). For stochastic
        volatility, shape is (T, n_vars, n_vars).
        """
        ...

    def cholesky_at(self, posterior: "xr.Dataset", t: int | None) -> np.ndarray:
        """Posterior draws of the Cholesky factor at time `t`.

        Returns shape (chains, draws, n_vars, n_vars). For constant
        volatility, `t` is ignored. For stochastic volatility, indexes
        into the time dimension; `t=None` defaults to the most recent
        period.
        """
        ...

    def forecast_cholesky_path(
        self,
        posterior: "xr.Dataset",
        steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Posterior-predictive Cholesky factors for steps ahead.

        Returns shape (chains, draws, steps, n_vars, n_vars). For constant
        volatility, broadcasts the constant L across `steps`. For stochastic
        volatility, simulates forward from posterior dynamics.
        """
        ...
