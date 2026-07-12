"""Identification schemes for structural VAR analysis."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field, PrivateAttr

from impulso._base import ImpulsoBaseModel, ImpulsoModel
from impulso._linalg import sigma_from_cholesky
from impulso._ma import compute_ma_phi

if TYPE_CHECKING:
    from impulso.data import VARData


class Cholesky(ImpulsoModel):
    """Cholesky identification scheme.

    Uses the lower-triangular Cholesky decomposition of the residual
    covariance matrix to identify structural shocks. Variable ordering
    determines the causal ordering.

    Attributes:
        ordering: Ordered list of variable names (most exogenous first).
    """

    ordering: list[str]

    def identify(
        self,
        L: np.ndarray,
        var_names: list[str],
        posterior: "xr.Dataset | None" = None,
        data: "VARData | None" = None,
        n_lags: int | None = None,
    ) -> np.ndarray:
        """Apply Cholesky identification.

        For default variable ordering, `identify` is a no-op and returns
        `L` unchanged. When `self.ordering` differs from `var_names`,
        the underlying covariance is permuted and re-decomposed so the
        Cholesky factor reflects the requested causal ordering.

        Args:
            L: Lower-triangular Cholesky factor, shape (chains, draws, n_vars, n_vars).
            var_names: Variable names in the data's natural order.
            posterior: Unused. Accepted for Protocol uniformity.
            data: Unused. Accepted for Protocol uniformity.
            n_lags: Unused. Accepted for Protocol uniformity.

        Returns:
            Structural shock matrix, shape (chains, draws, n_vars, n_vars).
        """
        del posterior, data, n_lags  # unused

        # Fast path: ordering matches data — identify is a no-op.
        if list(self.ordering) == list(var_names):
            return L

        # Reordering: reconstruct Sigma, permute, re-decompose.
        sigma = sigma_from_cholesky(L)
        perm = [var_names.index(v) for v in self.ordering]
        ix0, ix1 = np.ix_(perm, perm)
        sigma_ordered = sigma[:, :, ix0, ix1]
        return np.linalg.cholesky(sigma_ordered)

    def shock_coords(self, n_vars: int) -> list[str]:
        """Cholesky shock labels are simply the causal ordering."""
        del n_vars  # ordering already has the right length
        return list(self.ordering)


class SignRestriction(ImpulsoModel):
    """Sign restriction identification scheme.

    Uses random rotation matrices to find structural impact matrices
    satisfying sign restrictions on impulse responses.

    Attributes:
        restrictions: Dict mapping variable -> {shock_name: "+" or "-"}.
        n_rotations: Number of candidate rotations per draw.
        random_seed: Seed for reproducibility.
    """

    restrictions: dict[str, dict[str, str]]
    n_rotations: int = Field(default=1000, ge=1)
    restriction_horizon: int = Field(default=0, ge=0)
    random_seed: int | None = None

    # Single-call scratchpad: identify() writes the rate; the pipeline
    # (set_identification_strategy) reads it immediately afterwards and
    # attaches to idata.posterior.attrs. Not reentrant — overwritten on
    # each identify() call. Do not rely on this between calls; the
    # surviving public surface is the posterior attr.
    _last_acceptance_rate: float = PrivateAttr(default=0.0)

    def identify(
        self,
        L: np.ndarray,
        var_names: list[str],
        posterior: "xr.Dataset | None" = None,
        data: "VARData | None" = None,
        n_lags: int | None = None,
    ) -> np.ndarray:
        """Apply sign-restriction identification.

        Args:
            L: Lower-triangular Cholesky factor, shape (chains, draws, n_vars, n_vars).
            var_names: Variable names in the data's natural order.
            posterior: Required when `self.restriction_horizon > 0` because
                the multi-horizon check needs the VAR coefficients `B` from
                the posterior. Ignored for impact-only restrictions
                (`restriction_horizon == 0`).
            data: Unused. Accepted for Protocol uniformity.
            n_lags: Unused. Accepted for Protocol uniformity.

        Returns:
            Structural shock matrix, shape (chains, draws, n_vars, n_vars).
            Per-draw fallback to the supplied `L` for draws where no
            rotation satisfies the restrictions. Acceptance rate available
            via the `sign_restriction_acceptance_rate` attribute on the
            wrapping IdentifiedVAR's posterior (set by the pipeline).
        """
        del data, n_lags  # unused
        from scipy.stats import special_ortho_group

        n_chains, n_draws, n_vars, _ = L.shape
        rng = np.random.default_rng(self.random_seed)

        shock_names = list(next(iter(self.restrictions.values())).keys())

        # Multi-horizon path needs B — fail clearly if posterior wasn't provided.
        B_all: np.ndarray | None = None
        n_lags = 0
        if self.restriction_horizon > 0:
            if posterior is None or "B" not in posterior:
                raise ValueError(
                    "restriction_horizon > 0 requires the full posterior with 'B' "
                    "(VAR coefficients). Pass posterior=fitted.idata.posterior to identify()."
                )
            B_all = posterior["B"].values
            n_lags = B_all.shape[-1] // n_vars

        P = np.full((n_chains, n_draws, n_vars, n_vars), np.nan)
        accepted_count = 0
        total_count = n_chains * n_draws
        for c in range(n_chains):
            for d in range(n_draws):
                chol = L[c, d]
                found = False
                B_draw = B_all[c, d] if B_all is not None else None
                for _ in range(self.n_rotations):
                    Q = special_ortho_group.rvs(n_vars, random_state=rng)
                    candidate = chol @ Q
                    if self.restriction_horizon == 0:
                        ok = self._check_restrictions(candidate, var_names, shock_names)
                    else:
                        ok = self._check_restrictions_at_horizons(candidate, B_draw, var_names, shock_names, n_lags)
                    if ok:
                        P[c, d] = candidate
                        found = True
                        accepted_count += 1
                        break
                if not found:
                    P[c, d] = chol  # Fallback to the unrotated factor.

        fallback_count = total_count - accepted_count
        if fallback_count > 0:
            import warnings

            warnings.warn(
                f"Sign restrictions not satisfied for {fallback_count}/{total_count} draws "
                f"({fallback_count / total_count:.1%}). Those draws fell back to L (Cholesky).",
                stacklevel=2,
            )

        # Stash the acceptance rate as a side channel — the pipeline reads
        # it back to attach to the InferenceData.attrs.
        self._last_acceptance_rate = accepted_count / total_count
        return P

    @staticmethod
    def _build_shock_coords(shock_names: list[str], n_vars: int) -> list[str]:
        """Build shock coordinate labels for the structural shock matrix.

        Named shocks occupy their column positions; remaining columns
        are labeled 'unidentified_1', 'unidentified_2', etc.
        """
        if len(shock_names) == n_vars:
            return shock_names
        return shock_names + [f"unidentified_{i}" for i in range(1, n_vars - len(shock_names) + 1)]

    def shock_coords(self, n_vars: int) -> list[str]:
        """Sign-restriction shock labels: named shocks first, then padding."""
        shock_names = list(next(iter(self.restrictions.values())).keys())
        return self._build_shock_coords(shock_names, n_vars)

    def _check_restrictions_at_horizons(
        self,
        candidate: np.ndarray,
        B_draw: np.ndarray,
        var_names: list[str],
        shock_names: list[str],
        n_lags: int,
    ) -> bool:
        """Check sign restrictions at all horizons 0..restriction_horizon.

        Args:
            candidate: Candidate structural impact matrix (n_vars, n_vars).
            B_draw: VAR coefficient matrix (n_vars, n_vars * n_lags) for this draw.
            var_names: Variable names.
            shock_names: Shock names from restrictions.
            n_lags: Number of lags in the VAR.

        Returns:
            True if all restrictions satisfied at all horizons.
        """
        n_vars = candidate.shape[0]

        # Always check impact (h=0)
        if not self._check_restrictions(candidate, var_names, shock_names):
            return False

        A = [B_draw[:, j * n_vars : (j + 1) * n_vars] for j in range(n_lags)]
        Phi = compute_ma_phi(A, self.restriction_horizon)  # (H+1, n, n)

        # Phi[0] (= I) handles the impact check above; iterate h=1..H here.
        for h in range(1, self.restriction_horizon + 1):
            irf_h = Phi[h] @ candidate
            if not self._check_restrictions(irf_h, var_names, shock_names):
                return False

        return True

    def _check_restrictions(self, candidate: np.ndarray, var_names: list[str], shock_names: list[str]) -> bool:
        """Check if a candidate matrix satisfies all sign restrictions."""
        for var_name, shocks in self.restrictions.items():
            var_idx = var_names.index(var_name)
            for shock_name, sign in shocks.items():
                shock_idx = shock_names.index(shock_name)
                val = candidate[var_idx, shock_idx]
                if sign == "+" and val < 0:
                    return False
                if sign == "-" and val > 0:
                    return False
        return True


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

    # Single-call scratchpad, mirroring SignRestriction._last_acceptance_rate:
    # identify() writes first-stage diagnostics; the pipeline reads them
    # immediately afterwards and attaches to the shock matrix attrs.
    _last_diagnostics: dict[str, float] = PrivateAttr(default_factory=dict)

    # Memoised impact direction. The instrument-residual covariance (and
    # the first-stage diagnostics) depend only on (posterior, data, n_lags)
    # — not on L — so under time-varying volatility, where the pipeline
    # calls identify() once per period with the same posterior/data, the
    # expensive residual reconstruction runs once instead of T times.
    # Keyed by object identity: valid while the caller holds the same
    # posterior/data objects, which is exactly the per-t loop's lifetime.
    _impact_cache: tuple | None = PrivateAttr(default=None)

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

        cache_key = (id(posterior), id(data), n_lags)
        if self._impact_cache is not None and self._impact_cache[0] == cache_key:
            d = self._impact_cache[1]
        else:
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
            }
            self._impact_cache = (cache_key, d)
            if f_median < 10.0:
                import warnings

                warnings.warn(
                    f"Weak instrument: posterior-median first-stage F = {f_median:.2f} < 10. "
                    "The identified impact column is unreliable.",
                    UserWarning,
                    stacklevel=2,
                )

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
            posterior: Posterior Dataset with `B` and `intercept` draws
                (`fitted.idata.posterior`).
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
        return SignRestriction._build_shock_coords([self.shock_name], n_vars)
