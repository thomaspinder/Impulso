"""Sign-restriction identification (rotation rejection sampling)."""

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr
from pydantic import Field, PrivateAttr

from impulso._base import ImpulsoModel
from impulso._linalg import lag_matrices
from impulso._ma import compute_ma_phi
from impulso._posterior import COEFFICIENTS, coefficient_draws
from impulso.identification._shared import pad_shock_coords

if TYPE_CHECKING:
    from impulso.data import VARData


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

    # Rotation-sampling schemes cannot yet pin one rotation per posterior
    # draw across forecast steps; forecast-side scenario machinery checks
    # this capability flag and refuses time-varying volatility.
    _samples_rotations: ClassVar[bool] = True

    # Single-call scratchpad backing `last_diagnostics`: identify() writes
    # the acceptance rate; the pipeline (IdentifiedVAR.shock_matrix) reads
    # it back immediately afterwards and attaches it to the shock-matrix
    # DataArray's attrs. Not reentrant — overwritten on each identify() call.
    _last_diagnostics: dict[str, float] = PrivateAttr(default_factory=dict)

    @property
    def last_diagnostics(self) -> dict[str, float]:
        """Diagnostics from the most recent `identify()` call.

        Scheme-prefixed scalars (see CONTEXT.md "Identification
        diagnostics"), overwritten per call and surfaced onto
        `IdentifiedVAR.shock_matrix().attrs` by the pipeline. Returns a copy.
        """
        return dict(self._last_diagnostics)

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
            rotation satisfies the restrictions. The acceptance rate is
            available as `last_diagnostics["sign_restriction_acceptance_rate"]`
            and on the matching `IdentifiedVAR.shock_matrix()` attr.
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
            if posterior is None or COEFFICIENTS not in posterior:
                raise ValueError(
                    "restriction_horizon > 0 requires the full posterior with 'B' "
                    "(VAR coefficients). Pass the fit's posterior group as an xarray.Dataset "
                    "to identify() — FittedVAR.set_identification_strategy(...) does this for you."
                )
            B_all = coefficient_draws(posterior)
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

        # Stash the acceptance rate in the diagnostics scratchpad — the
        # pipeline reads `last_diagnostics` back to attach to the attrs.
        self._last_diagnostics = {"sign_restriction_acceptance_rate": accepted_count / total_count}
        return P

    def shock_coords(self, n_vars: int) -> list[str]:
        """Sign-restriction shock labels: named shocks first, then padding."""
        shock_names = list(next(iter(self.restrictions.values())).keys())
        return pad_shock_coords(shock_names, n_vars)

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
        # Always check impact (h=0)
        if not self._check_restrictions(candidate, var_names, shock_names):
            return False

        A = lag_matrices(B_draw, n_lags)
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
