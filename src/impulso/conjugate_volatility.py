"""Deterministic volatility break for the conjugate VAR (Lenza-Primiceri, 2020).

`ConjugateVolatility` is the query-surface adapter a conjugate fit attaches when
volatility is time-varying. It reports `L_t = s_t * L_base`, where
`L_base = posterior["L"]` is the base Cholesky factor drawn by the conjugate
engine and `s_t >= 0` is a deterministic volatility multiplier (so
`Sigma_t = s_t**2 * Sigma_base`). Alongside the read-only query surface it
exposes the two estimation-side hooks the marginal-likelihood sampler consumes:

- `hyperparameter_priors()` — the 1-D priors on the free volatility hyperparameters.
- `log_scales(theta, T)` — the in-sample log-scale path `log s_t` fed to
  `impulso._conjugate.log_marginal_likelihood(log_scales=...)`.

`PandemicBreak` implements the Lenza-Primiceri (2020) COVID-19 break: three free
outbreak scales at `t*`, `t*+1`, `t*+2` (March-May 2020) with Pareto(1, 1) priors
and a Beta-distributed geometric decay `rho` back toward 1 afterwards.

See docs/adr/0004-conjugate-var-is-a-sibling-estimator.md and the build contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
from pydantic import Field
from scipy import stats

from impulso._base import ImpulsoModel

if TYPE_CHECKING:
    import xarray as xr

# Hyperparameter names shared by `hyperparameter_priors`, `log_scales`, and the
# posterior variables the sampler packs. Kept in one place so the three stay in sync.
_SCALE_KEYS: tuple[str, str, str] = ("s_march", "s_april", "s_may")
_DECAY_KEY: str = "rho"

# Beta(a, b) decay prior on rho: mode 0.8, sd 0.2 (Lenza-Primiceri, 2020).
_RHO_BETA_A: float = 3.03568545
_RHO_BETA_B: float = 1.50892136


class _FrozenDist(Protocol):
    """Minimal structural type for a SciPy frozen 1-D distribution."""

    def logpdf(self, x: float) -> float: ...

    def support(self) -> tuple[float, float]: ...


@dataclass(frozen=True)
class Prior1D:
    """A one-dimensional hyperparameter prior: log-density plus bounded support.

    Thin wrapper over a SciPy frozen distribution so the marginal-likelihood
    sampler has a stable seam — `logpdf` for the objective, `support` for
    proposal bounds — that does not depend on SciPy's private frozen type.

    Attributes:
        dist: The underlying SciPy frozen distribution.
    """

    dist: _FrozenDist

    def logpdf(self, x: float) -> float:
        """Log prior density at `x` (`-inf` outside the support)."""
        return self.dist.logpdf(x)

    @property
    def support(self) -> tuple[float, float]:
        """`(lower, upper)` bounds of the prior support."""
        lower, upper = self.dist.support()
        return float(lower), float(upper)


class ConjugateVolatility(ImpulsoModel):
    """Query-surface adapter for a deterministically time-varying conjugate VAR.

    Reports `L_t = s_t * L_base`, where `L_base = posterior["L"]` is the base
    Cholesky factor drawn by the conjugate engine and `s_t` is a deterministic
    volatility multiplier whose schedule is defined by the subclass. The
    multiplier's free hyperparameters are estimated by the marginal-likelihood
    sampler through `hyperparameter_priors` and `log_scales`.

    Subclasses implement the schedule via `log_scales`, `hyperparameter_priors`,
    `_posterior_scales`, and `_forecast_indices`; the query surface itself
    (`cholesky_at` / `cholesky_path` / `forecast_cholesky_path`) is shared here.

    Attributes:
        is_time_varying: Always `True` — `Sigma_t` varies across `t`.
    """

    is_time_varying: bool = True

    # --- estimation-side surface (subclass-provided) ---
    def hyperparameter_priors(self) -> dict[str, Prior1D]:
        """1-D priors on the free volatility hyperparameters, keyed by name."""
        raise NotImplementedError

    def log_scales(self, theta: dict[str, float], T: int) -> np.ndarray:
        """In-sample log-scale path `log s_t` of shape `(T,)` for hyperparameters `theta`.

        Fed to `impulso._conjugate.log_marginal_likelihood(log_scales=...)`.
        """
        raise NotImplementedError

    # --- scale hooks (subclass-provided) ---
    def _posterior_scales(self, posterior: xr.Dataset, time_indices: np.ndarray) -> np.ndarray:
        """Posterior scale draws `s_t` at absolute in-sample `time_indices`.

        Returns shape `(chains, draws, len(time_indices))`.
        """
        raise NotImplementedError

    def _forecast_indices(self, steps: int) -> np.ndarray:
        """Absolute in-sample-equivalent time indices for forecast steps `0..steps-1`."""
        raise NotImplementedError

    # --- query surface (shared) ---
    def cholesky_at(self, posterior: xr.Dataset, t: int | None) -> np.ndarray:
        """Cholesky factor `L_t = s_t * L_base` at time `t` for every draw.

        Args:
            posterior: Dataset with the base factor `L` (chains, draws, n, n) and
                the subclass's scale-hyperparameter draws.
            t: In-sample time index. `None` returns the baseline factor
                (`s_t = 1`), i.e. the pandemic-free covariance — a deterministic
                break stores no explicit "most recent" time to resolve.

        Returns:
            `(chains, draws, n_vars, n_vars)`.
        """
        base = posterior["L"].values
        if t is None:
            return base.copy()
        scale = self._posterior_scales(posterior, np.asarray([t]))[:, :, 0]  # (C, D)
        return scale[:, :, None, None] * base

    def cholesky_path(self, posterior: xr.Dataset, T: int) -> np.ndarray:
        """Cholesky factor path `L_t` for `t` in `0..T-1`.

        Returns `(chains, draws, T, n_vars, n_vars)`.
        """
        base = posterior["L"].values
        scale = self._posterior_scales(posterior, np.arange(T))  # (C, D, T)
        return scale[:, :, :, None, None] * base[:, :, None, :, :]

    def forecast_cholesky_path(
        self,
        posterior: xr.Dataset,
        steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Cholesky factor path for `steps` ahead, continuing the deterministic decay.

        `rng` is accepted for parity with stochastic adapters and ignored — the
        post-sample scale path is deterministic given the posterior draws.

        Returns `(chains, draws, steps, n_vars, n_vars)`.
        """
        base = posterior["L"].values
        scale = self._posterior_scales(posterior, self._forecast_indices(steps))  # (C, D, steps)
        return scale[:, :, :, None, None] * base[:, :, None, :, :]


class PandemicBreak(ConjugateVolatility):
    """Lenza-Primiceri (2020) deterministic COVID-19 volatility break.

    Three free outbreak scales at `t*`, `t*+1`, `t*+2` (March-May 2020) inflate
    the residual covariance, after which volatility decays geometrically back
    toward its pre-pandemic level (`j = t - t*`)::

        s_t = 1                              for t < t*        (pre-pandemic)
        s_t = s_march, s_april, s_may        at t*, t*+1, t*+2 (outbreak)
        s_t = 1 + (s_may - 1) * rho**(j - 2) for t >= t*+3     (decay)

    The forecast path continues the decay: step `k` (June 2020 onward, `t*+3+k`)
    uses `1 + (s_may - 1) * rho**(k + 1)`. Each outbreak scale carries a
    Pareto(1, 1) prior (support `>= 1`); `rho` carries a Beta prior with mode 0.8
    and sd 0.2.

    Attributes:
        name: Discriminator key (always `"pandemic_break"`).
        start: Index of `t*` (March 2020) in the lag-trimmed in-sample data.
    """

    name: Literal["pandemic_break"] = "pandemic_break"
    start: int = Field(ge=0)

    def hyperparameter_priors(self) -> dict[str, Prior1D]:
        outbreak = Prior1D(stats.pareto(b=1.0, scale=1.0))
        priors: dict[str, Prior1D] = dict.fromkeys(_SCALE_KEYS, outbreak)
        priors[_DECAY_KEY] = Prior1D(stats.beta(a=_RHO_BETA_A, b=_RHO_BETA_B))
        return priors

    def log_scales(self, theta: dict[str, float], T: int) -> np.ndarray:
        s_march, s_april, s_may = (theta[key] for key in _SCALE_KEYS)
        rho = theta[_DECAY_KEY]
        return np.log(self._scale_series(s_march, s_april, s_may, rho, np.arange(T)))

    def _posterior_scales(self, posterior: xr.Dataset, time_indices: np.ndarray) -> np.ndarray:
        s_march, s_april, s_may = (posterior[key].values for key in _SCALE_KEYS)
        rho = posterior[_DECAY_KEY].values
        return self._scale_series(s_march, s_april, s_may, rho, time_indices)

    def _forecast_indices(self, steps: int) -> np.ndarray:
        # In-sample ends at t*+2 (May 2020); forecast step k continues the decay
        # at absolute index t*+3+k, so step 0 (June 2020) uses rho**1.
        return self.start + 3 + np.arange(steps)

    def _scale_series(
        self,
        s_march: float | np.ndarray,
        s_april: float | np.ndarray,
        s_may: float | np.ndarray,
        rho: float | np.ndarray,
        time_indices: np.ndarray,
    ) -> np.ndarray:
        """Volatility multiplier `s_t` at `time_indices`, shared by estimation and query paths.

        Accepts scalar hyperparameters (theta, giving shape `(M,)`) or
        `(chains, draws)` posterior-draw arrays (giving `(chains, draws, M)`),
        where `M = len(time_indices)`.
        """
        s_march, s_april, s_may, rho = (np.asarray(v, dtype=float) for v in (s_march, s_april, s_may, rho))
        batch = np.broadcast_shapes(s_march.shape, s_april.shape, s_may.shape, rho.shape)
        offset = np.asarray(time_indices) - self.start  # j = t - t*, shape (M,)

        def col(value: np.ndarray) -> np.ndarray:
            return np.broadcast_to(value, batch)[..., np.newaxis]  # (*batch, 1)

        s_may_c, rho_c = col(s_may), col(rho)
        scales = np.ones(batch + offset.shape)  # t < t*  ->  1
        scales = np.where(offset == 0, col(s_march), scales)
        scales = np.where(offset == 1, col(s_april), scales)
        scales = np.where(offset == 2, s_may_c, scales)
        decay = 1.0 + (s_may_c - 1.0) * rho_c ** (offset - 2)
        return np.where(offset >= 3, decay, scales)
