"""External-instrument (proxy) identification."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import PrivateAttr

from impulso._base import ImpulsoBaseModel
from impulso.identification._cache import _CACHE_MISS, _PosteriorCache
from impulso.identification._shared import pad_shock_coords

if TYPE_CHECKING:
    from impulso.data import VARData


class ProxySVAR(ImpulsoBaseModel):
    """External-instrument (proxy) identification for one structural shock.

    Identifies a single structural shock from an external instrument `z_t`
    that is correlated with the target shock (relevance) and uncorrelated
    with all others (exogeneity). Under those conditions the covariance
    between the instrument and the reduced-form residuals is proportional
    to the target shock's impact column:
    `E[z_t u_t] = phi * p_1`.

    Per posterior draw, the impact column is estimated as the sample
    covariance between the (date-aligned) instrument and that draw's
    reconstructed residuals, normalised on `policy_variable`. The
    remaining columns are completed orthogonally, consistent with the
    draw's shock covariance, so downstream code that needs a full
    invertible matrix (historical decomposition) keeps working — but
    those columns are rotation-arbitrary and are labelled
    `unidentified_1..` accordingly. Downstream guard rails respond to
    that labelling: `IdentifiedVAR.fevd` masks the unidentified columns'
    shares to NaN, and `IdentifiedVAR.historical_decomposition` collapses
    them into a single `unidentified_remainder` column (their sum is
    well-defined even though the split is not).

    Attributes:
        instrument: Instrument series with a DatetimeIndex. Aligned to the
            estimation sample by date at identify() time (inner join —
            months missing from the instrument are dropped, matching the
            reindex-and-drop convention in the proxy-SVAR literature).
            Periods where no event occurred should be zero, not NaN.
        policy_variable: Endogenous variable used to normalise the shock.
        shock_name: Label of the identified shock column.
        scale: If None (default), the identified column is a one-standard-
            deviation shock, consistent with the draw's shock covariance
            (`P @ P.T = Sigma` holds exactly). If a float, the column is
            rescaled per draw so the shock moves `policy_variable` by
            `scale` units on impact (unit-effect normalisation, e.g.
            `scale=10.0` for a +10% impact on a log*100 variable); the
            matrix then no longer reproduces Sigma, which is inherent to
            unit-effect normalisation.
    """

    instrument: pd.Series
    policy_variable: str
    shock_name: str = "instrumented"
    scale: float | None = None

    # Single-call scratchpad backing `last_diagnostics`: identify() writes
    # first-stage diagnostics; the pipeline reads them immediately
    # afterwards and attaches to the shock matrix attrs.
    _last_diagnostics: dict[str, float] = PrivateAttr(default_factory=dict)

    @property
    def last_diagnostics(self) -> dict[str, float]:
        """Diagnostics from the most recent `identify()` call.

        Scheme-prefixed scalars (see CONTEXT.md "Identification
        diagnostics"), overwritten per call and surfaced onto
        `IdentifiedVAR.shock_matrix().attrs` by the pipeline. Returns a copy.
        """
        return dict(self._last_diagnostics)

    # Memoised impact direction. The instrument-residual covariance (and
    # the first-stage diagnostics) depend only on (posterior, data, n_lags)
    # — not on L — so under time-varying volatility, where the pipeline
    # calls identify() once per period with the same posterior/data, the
    # expensive residual reconstruction runs once instead of T times.
    # Keyed by object identity, with weak references as the validity token:
    # valid while the caller holds the same posterior/data objects, which
    # is exactly the per-t loop's lifetime. See `_PosteriorCache`.
    _impact_cache: _PosteriorCache = PrivateAttr(default_factory=_PosteriorCache)

    def identify(
        self,
        L: np.ndarray,
        var_names: list[str],
        posterior: "xr.Dataset | None" = None,
        data: "VARData | None" = None,
        n_lags: int | None = None,
    ) -> np.ndarray:
        """Apply external-instrument identification.

        Args:
            L: Lower-triangular Cholesky factor, shape (chains, draws, n_vars, n_vars).
            var_names: Variable names in the data's natural order.
            posterior: Full posterior; required (residual reconstruction
                needs `B` and `intercept` draws).
            data: The VARData used at fit time; required for residual
                reconstruction and date alignment.
            n_lags: Lag order of the fitted VAR; required.

        Returns:
            Structural shock matrix, shape (chains, draws, n_vars, n_vars).
            Column 0 is the identified shock; columns 1.. are an arbitrary
            orthogonal completion.

        Raises:
            ValueError: If posterior/data/n_lags are missing, the policy
                variable is unknown, or the instrument does not overlap
                the estimation sample.
        """
        if posterior is None or data is None or n_lags is None:
            raise ValueError(
                "ProxySVAR.identify requires posterior, data, and n_lags — "
                "they are supplied automatically by "
                "FittedVAR.set_identification_strategy(...); pass them "
                "explicitly if calling identify() directly."
            )
        if self.policy_variable not in var_names:
            raise ValueError(f"policy_variable {self.policy_variable!r} not in var_names {var_names}")
        policy_idx = var_names.index(self.policy_variable)

        d = self._impact_cache.get((posterior, data), (n_lags,))
        if d is _CACHE_MISS:
            z, u = self._aligned_residuals(posterior, data, n_lags)

            # Impact direction: per-draw covariance between instrument and
            # residuals (both demeaned), normalised on the policy variable.
            z_c = z - z.mean()
            u_c = u - u.mean(axis=2, keepdims=True)
            s = np.einsum("t,cdti->cdi", z_c, u_c) / len(z)  # (C, D, n)
            d = s / s[:, :, policy_idx][:, :, np.newaxis]  # d[policy] = 1

            # First-stage strength: F of u_policy ~ const + z, per draw.
            f_draws = self._first_stage_f(z_c, u_c[:, :, :, policy_idx])
            f_median = float(np.median(f_draws))
            self._last_diagnostics = {
                "proxy_first_stage_f_median": f_median,
                "proxy_first_stage_f_q05": float(np.quantile(f_draws, 0.05)),
                "proxy_first_stage_f_q95": float(np.quantile(f_draws, 0.95)),
                "proxy_impact_cache_hit": 0.0,
            }
            self._impact_cache.set((posterior, data), (n_lags,), d)
            if f_median < 10.0:
                import warnings

                warnings.warn(
                    f"Weak instrument: posterior-median first-stage F = {f_median:.2f} < 10. "
                    "The identified impact column is unreliable.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            self._last_diagnostics = {**self._last_diagnostics, "proxy_impact_cache_hit": 1.0}

        # Complete the matrix: q1 = L^{-1} d normalised, extended to an
        # orthonormal basis via a Householder reflection; P = L @ Q gives
        # P @ P.T = Sigma with column 0 proportional to d (positive factor,
        # so the shock raises the policy variable by construction).
        n = len(var_names)
        v = np.linalg.solve(L, d[..., np.newaxis])[..., 0]  # (C, D, n)
        q1 = v / np.linalg.norm(v, axis=-1, keepdims=True)
        e1 = np.zeros(n)
        e1[0] = 1.0
        w = q1 - e1
        w_norm2 = np.einsum("cdi,cdi->cd", w, w)[..., np.newaxis, np.newaxis]
        outer = w[..., :, np.newaxis] * w[..., np.newaxis, :]
        eye = np.broadcast_to(np.eye(n), outer.shape)
        Q = np.where(w_norm2 > 1e-14, eye - 2.0 * outer / np.where(w_norm2 > 1e-14, w_norm2, 1.0), eye)
        P = L @ Q

        if self.scale is not None:
            # Unit-effect normalisation: the identified column moves the
            # policy variable by `scale` on impact, per draw.
            P = P.copy()
            P[..., 0] = d * self.scale
        return P

    def _aligned_residuals(
        self, posterior: "xr.Dataset", data: "VARData", n_lags: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct residuals and align the instrument to them by date.

        Returns:
            Tuple `(z, u)`: the instrument values on the overlap, shape
            `(T_z,)`, and the matching residual draws, `(C, D, T_z, n)`.

        Raises:
            ValueError: If the instrument index does not overlap the
                estimation sample, or the overlap is too short.
        """
        from impulso._residuals import reduced_form_residuals

        resid = reduced_form_residuals(posterior, data, n_lags)  # (C, D, T_eff, n)

        # Inner join on dates: months missing from the instrument are dropped.
        eff_index = data.index[n_lags:]
        common = eff_index.intersection(self.instrument.index)
        n_vars = resid.shape[-1]
        if len(common) == 0:
            raise ValueError(
                "Instrument index does not overlap the estimation sample "
                f"({eff_index[0]}..{eff_index[-1]}). Check the DatetimeIndex "
                "frequency and range."
            )
        if len(common) < 3 * n_vars:
            raise ValueError(
                f"Only {len(common)} instrument observations overlap the "
                "estimation sample — too few to identify the impact column."
            )
        positions = eff_index.get_indexer(common)
        z = self.instrument.loc[common].to_numpy(dtype=float)
        return z, resid[:, :, positions, :]

    def first_stage(self, posterior: "xr.Dataset", data: "VARData", n_lags: int) -> np.ndarray:
        """Posterior draws of the first-stage F statistic.

        Regresses the policy variable's reconstructed reduced-form
        residuals on the date-aligned instrument (with a constant), per
        posterior draw. Because the residuals differ draw by draw, the
        instrument-relevance F is itself a posterior quantity.

        Args:
            posterior: Posterior Dataset with `B` and `intercept` draws (the
                fit's posterior group).
            data: The VARData used at fit time.
            n_lags: Lag order of the fitted VAR.

        Returns:
            F statistics, shape `(chains, draws)`.
        """
        policy_idx = data.endog_names.index(self.policy_variable)
        z, u = self._aligned_residuals(posterior, data, n_lags)
        z_c = z - z.mean()
        u_c = u - u.mean(axis=2, keepdims=True)
        return self._first_stage_f(z_c, u_c[:, :, :, policy_idx])

    @staticmethod
    def _first_stage_f(z_c: np.ndarray, u_policy_c: np.ndarray) -> np.ndarray:
        """Per-draw F-stat of the first stage u_policy ~ const + z.

        Args:
            z_c: Demeaned instrument, shape (T_z,).
            u_policy_c: Demeaned policy-variable residuals, (C, D, T_z).

        Returns:
            F statistics, shape (C, D).
        """
        T = len(z_c)
        szz = z_c @ z_c
        szu = np.einsum("t,cdt->cd", z_c, u_policy_c)
        slope = szu / szz
        ess = slope**2 * szz  # explained sum of squares
        tss = np.einsum("cdt,cdt->cd", u_policy_c, u_policy_c)
        rss = tss - ess
        return ess / (rss / (T - 2))

    def shock_coords(self, n_vars: int) -> list[str]:
        """Identified shock first, then rotation-arbitrary padding."""
        return pad_shock_coords([self.shock_name], n_vars)
