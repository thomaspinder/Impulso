"""Protocol definitions for extensible components."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import arviz as az
    import pandas as pd
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
class DeterministicTerm(Protocol):
    """Contract for deterministic regressor terms.

    A term is a total function of a timestamp: given an index it returns
    real-valued columns with no missing entries, and it does so from the
    calendar alone — never from the endogenous data. `DeterministicDesign`
    composes terms into an exogenous design matrix.

    The `(origin, alias)` pair is resolved **once** from the estimation
    index and handed to every term, for both in-sample construction and
    out-of-sample extension. Terms that count elapsed time (trends,
    harmonics) must anchor on it rather than on the position of a row
    inside the index they are handed; that is what makes
    `design.extend(index, h)` reproduce the rows `design.build` would
    have written had the index been `h` periods longer.

    Concrete implementations: `Trend`, `Fourier`, `SeasonalDummies`,
    `BreakDummy` (all in `impulso.deterministic`).
    """

    @property
    def column_names(self) -> list[str]:
        """Names of the columns this term contributes, in build order.

        Must be knowable without an index — the design's column contract
        is static — and must have the same length as `build`'s second
        axis.
        """
        ...

    def build(self, index: "pd.DatetimeIndex", origin: "pd.Timestamp", alias: str) -> np.ndarray:
        """Evaluate the term on `index`.

        Args:
            index: Timestamps to evaluate at. Not necessarily the
                estimation index — `DeterministicDesign.extend` passes a
                future index while keeping `origin` and `alias` fixed.
            origin: First timestamp of the estimation index; the zero
                point for elapsed-time counts.
            alias: pandas period alias for the sampling frequency (e.g.
                `"M"`, `"Q-DEC"`, `"D"`, `"15D"`), used to convert
                timestamps to integer period ordinals. Note that a
                multiplied alias stores ordinals in its *base* unit, so
                elapsed time must be divided by the multiplier to be
                counted in sampling periods.

        Returns:
            Float array of shape `(len(index), len(self.column_names))`
            containing only finite values.
        """
        ...


@runtime_checkable
class IdentificationScheme(Protocol):
    """Contract for structural identification schemes.

    Optional capability flag: schemes that *sample* rotations inside
    `identify` (a fresh draw per call, as `SignRestriction` does) must set
    a truthy class attribute `_samples_rotations`. Forecast-side scenario
    machinery reads it via `getattr(scheme, "_samples_rotations", False)`
    and refuses time-varying volatility for such schemes — no single
    structural coordinate system would span the forecast steps otherwise.
    Deterministic schemes (`Cholesky`, `ProxySVAR`) omit the flag.
    """

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
class ErrorDistribution(Protocol):
    """Contract for observation error distributions.

    An ErrorDistribution owns *how the observation error enters the model*:
    which likelihood is registered inside PyMC, and how standardised
    innovations are drawn on the forecast side. It does not own the error's
    scale — that belongs to the `VolatilityProcess`, which supplies the
    Cholesky factor `L` of the scale matrix Ω = L Lᵀ. Concrete adapters:
    `Gaussian` (the default) and `StudentT`.

    Under heavy-tailed adapters Ω is the *scale* matrix, not the covariance;
    `variance_inflation` supplies the factor relating the two, which is what
    `FittedVAR.innovation_covariance` multiplies `sigma()` by. See
    docs/adr/0007-student-t-errors-use-the-scale-matrix-convention.md.

    RNG contract:
        `draw_standardised_innovations` must consume `rng.standard_normal`
        **first**, before any other generator call. The Gaussian adapter
        consumes nothing else, so its stream is a strict prefix of every
        other adapter's and seeded Gaussian forecasts stay bit-identical to
        releases predating this seam. `FittedVAR.forecast` and the scenario
        engine share that stream order, which is what makes a matched-seed
        `conditional_forecast` with no pins equal `forecast()` draw for draw.

    Note:
        This is a single Protocol covering both the PyMC-side
        (`build_likelihood`) and query-side (`draw_standardised_innovations`,
        `variance_inflation`) surfaces, unlike the volatility seam, which is
        split into `VolatilityProcess` and `PyMCVolatilityProcess`. Both
        adapters here implement all three methods, so the split would buy
        nothing today. Splitting is warranted the moment a query-only adapter
        arrives — an error distribution reconstructed from a stored posterior
        that can never rebuild its own likelihood.
    """

    name: str
    is_heavy_tailed: bool
    """Whether the error law has heavier-than-Gaussian tails. `False` for
    `Gaussian`; `True` for `StudentT`. Drives the cross-seam rejection in
    `VAR._validate_spec` and the Gaussian-only guards on
    `FittedVAR.conditional_forecast` / `IdentifiedVAR.structural_scenario`,
    so neither has to string-sniff the adapter `name`."""

    def build_likelihood(
        self,
        name: str,
        mu: "pt.TensorVariable",
        chol: "pt.TensorVariable",
        observed: np.ndarray,
        dims: tuple[str, ...] | None = None,
    ) -> "pt.TensorVariable":
        """Register the observation likelihood in the active PyMC model.

        Args:
            name: Name for the observed random variable (the pipeline uses
                `"obs"`).
            mu: Conditional mean tensor of shape `(T, n_vars)`.
            chol: Lower-triangular Cholesky factor of the scale matrix Ω,
                shape `(n_vars, n_vars)` or `(T, n_vars, n_vars)`.
            observed: Observed endogenous matrix of shape `(T, n_vars)`.
            dims: PyMC dims for the observed variable.

        Returns:
            The registered PyMC random variable.
        """
        ...

    def draw_standardised_innovations(
        self,
        shape: tuple[int, ...],
        rng: np.random.Generator,
        posterior: "xr.Dataset",
    ) -> np.ndarray:
        """Draw standardised innovations ξ to be scaled by `L`.

        The forecast recursion adds `L_h @ ξ` to the conditional mean, so ξ
        carries the shape of the error law with the scale divided out.

        Args:
            shape: Draw shape, `(chains, draws, n_vars)`.
            rng: Generator to consume — see the RNG contract on the class.
            posterior: Posterior Dataset, for adapters whose innovation law
                depends on estimated parameters (`StudentT` reads `nu`).

        Returns:
            Array of shape `shape`.
        """
        ...

    def variance_inflation(self, posterior: "xr.Dataset") -> "float | np.ndarray":
        """Factor converting the scale matrix Ω into the innovation covariance.

        `Cov[y_t] = variance_inflation * Ω`. `1.0` for Gaussian errors;
        `nu/(nu-2)` per draw for Student-t.

        Args:
            posterior: Posterior Dataset.

        Returns:
            A scalar, or an array of shape `(chains, draws)` broadcastable
            against the leading dims of `FittedVAR.sigma()`.
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
