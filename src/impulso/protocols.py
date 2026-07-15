"""Protocol definitions for extensible components."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt
    import xarray as xr

    from impulso.data import VARData


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

    def identify(
        self,
        L: np.ndarray,
        var_names: list[str],
        posterior: "xr.Dataset | None" = None,
        data: "VARData | None" = None,
        n_lags: int | None = None,
    ) -> np.ndarray:
        """Identify the structural shock matrix from a Cholesky factor.

        Args:
            L: Lower-triangular Cholesky factor of the structural-shock
                covariance, shape (chains, draws, n_vars, n_vars). Produced
                by `volatility.cholesky_at(...)`.
            var_names: Endogenous variable names, in the order they appear
                in the underlying data.
            posterior: Full posterior xarray Dataset. Optional; provided by
                the pipeline so schemes that need additional draws (e.g.,
                SignRestriction with restriction_horizon > 0 needs B for the
                MA recursion) can reach for them. Schemes that only need L
                may ignore this argument. Schemes that need `posterior`
                for context but receive `None` should raise a clear
                `ValueError`.
            data: The VARData used at fit time. Optional; provided by the
                pipeline so schemes that need the observed sample — e.g.
                `ProxySVAR`, which reconstructs reduced-form residuals and
                aligns an instrument by date — can reach for it. Schemes
                that only need L may ignore this argument.
            n_lags: Lag order of the fitted VAR. Provided alongside `data`
                because residual reconstruction needs it.

        Returns:
            Structural shock matrix array of shape (chains, draws, n_vars, n_vars).
            Caller is responsible for wrapping into an xarray DataArray with
            named coords.
        """
        ...

    def shock_coords(self, n_vars: int) -> list[str]:
        """Return the labels for the `shock` coordinate of the structural matrix.

        The pipeline calls this after `identify` to label the columns of
        the structural shock matrix when wrapping into an xarray DataArray.

        Args:
            n_vars: Number of endogenous variables (i.e. width of the
                structural shock matrix).

        Returns:
            A list of length `n_vars` naming each shock column.
        """
        ...


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
    is_time_varying: bool
    """Whether Σ_t evolves over time. `False` for homoscedastic adapters
    (`Constant`); `True` for stochastic-volatility adapters. Drives
    branching in `FittedVAR.sigma` and `IdentifiedVAR.historical_decomposition`
    so neither has to string-sniff the adapter `name`."""

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

    def cholesky_path(self, posterior: "xr.Dataset", T: int) -> np.ndarray:
        """Posterior draws of the Cholesky factor path across all in-sample t.

        Returns shape (chains, draws, T, n_vars, n_vars). For constant
        volatility, broadcasts the time-invariant L across the requested
        `T`. For stochastic volatility, indexes into the latent log-vol
        posterior to construct L_t for each t.
        """
        ...


@runtime_checkable
class PyMCVolatilityProcess(VolatilityProcess, Protocol):
    """Volatility process that can build its latents inside a PyMC model.

    Extends the query surface with `build_pymc_latent`, required by the
    PyMC/NUTS estimation path (`VAR.fit`). Query-only adapters (e.g. the
    conjugate volatility break) implement `VolatilityProcess` alone and
    need not provide this method.
    """

    def build_pymc_latent(
        self,
        n_vars: int,
        T: int,
        data: np.ndarray | None = None,
    ) -> "pt.TensorVariable":
        """Register volatility latent variables in the active PyMC model.

        Returns the lower-triangular Cholesky factor as a PyTensor variable.
        For constant volatility, shape is (n_vars, n_vars). For stochastic
        volatility, shape is (T, n_vars, n_vars).

        Args:
            n_vars: Number of endogenous variables.
            T: Number of in-sample observations (after lag trimming).
            data: Optional per-variable series of shape `(T, n_vars)`,
                typically OLS residuals from the VAR pipeline. Stochastic
                adapters use this to seed per-variable priors; constant
                adapters ignore it.
        """
        ...
