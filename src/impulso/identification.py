"""Identification schemes for structural VAR analysis."""

import arviz as az
import numpy as np
import xarray as xr
from pydantic import Field

from impulso._base import ImpulsoModel


class Cholesky(ImpulsoModel):
    """Cholesky identification scheme.

    Uses the lower-triangular Cholesky decomposition of the residual
    covariance matrix to identify structural shocks. Variable ordering
    determines the causal ordering.

    Attributes:
        ordering: Ordered list of variable names (most exogenous first).
    """

    ordering: list[str]

    def identify(self, idata: az.InferenceData, var_names: list[str]) -> az.InferenceData:
        """Apply Cholesky identification to posterior covariance draws.

        Args:
            idata: InferenceData with 'Sigma' in posterior.
            var_names: Variable names from the VAR model.

        Returns:
            InferenceData with 'structural_shock_matrix' added to posterior.
        """
        sigma = idata.posterior["Sigma"].values  # (chains, draws, n, n)

        # Reorder if needed
        perm = [var_names.index(v) for v in self.ordering]
        sigma_ordered = sigma[:, :, np.ix_(perm, perm)[0], np.ix_(perm, perm)[1]]

        # Cholesky decompose each draw (broadcasts over leading dims)
        P = np.linalg.cholesky(sigma_ordered)

        P_da = xr.DataArray(
            P,
            dims=["chain", "draw", "shock", "response"],
            coords={"shock": self.ordering, "response": self.ordering},
        )

        new_posterior = idata.posterior.assign(structural_shock_matrix=P_da)
        return az.InferenceData(posterior=new_posterior)


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

    def identify(self, idata: az.InferenceData, var_names: list[str]) -> az.InferenceData:
        """Apply sign restriction identification.

        Args:
            idata: InferenceData with 'Sigma' in posterior.
            var_names: Variable names from the VAR model.

        Returns:
            InferenceData with 'structural_shock_matrix' added to posterior.
        """
        from scipy.stats import special_ortho_group

        sigma = idata.posterior["Sigma"].values
        n_chains, n_draws, n_vars, _ = sigma.shape
        rng = np.random.default_rng(self.random_seed)

        shock_names = list(next(iter(self.restrictions.values())).keys())

        # Extract B coefficients for multi-horizon checking
        if self.restriction_horizon > 0 and "B" not in idata.posterior:
            raise ValueError("restriction_horizon > 0 requires 'B' (VAR coefficients) in idata.posterior")
        B_all = idata.posterior["B"].values if self.restriction_horizon > 0 else None
        n_lags = B_all.shape[-1] // n_vars if B_all is not None else 0

        P = np.full((n_chains, n_draws, n_vars, n_vars), np.nan)

        accepted_count = 0
        total_count = n_chains * n_draws
        for c in range(n_chains):
            for d in range(n_draws):
                chol = np.linalg.cholesky(sigma[c, d])
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
                    P[c, d] = chol

        fallback_count = total_count - accepted_count
        if fallback_count > 0:
            import warnings

            warnings.warn(
                f"Sign restrictions not satisfied for {fallback_count}/{total_count} draws "
                f"({fallback_count / total_count:.1%}). Those draws fell back to Cholesky.",
                stacklevel=2,
            )

        coord_shocks = self._build_shock_coords(shock_names, n_vars)
        P_da = xr.DataArray(
            P,
            dims=["chain", "draw", "response", "shock"],
            coords={"response": var_names, "shock": coord_shocks},
        )

        new_posterior = idata.posterior.assign(structural_shock_matrix=P_da)
        new_posterior.attrs["sign_restriction_acceptance_rate"] = accepted_count / total_count
        return az.InferenceData(posterior=new_posterior)

    @staticmethod
    def _build_shock_coords(shock_names: list[str], n_vars: int) -> list[str]:
        """Build shock coordinate labels for the structural shock matrix.

        Named shocks occupy their column positions; remaining columns
        are labeled 'unidentified_1', 'unidentified_2', etc.
        """
        if len(shock_names) == n_vars:
            return shock_names
        return shock_names + [f"unidentified_{i}" for i in range(1, n_vars - len(shock_names) + 1)]

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

        # Extract lag coefficient matrices A_1..A_p
        A = [B_draw[:, j * n_vars : (j + 1) * n_vars] for j in range(n_lags)]

        # MA recursion: Phi_0 = I, Phi_h = sum_{j=1}^{min(h,p)} A_j @ Phi_{h-j}
        Phi_prev = [np.eye(n_vars)]
        for h in range(1, self.restriction_horizon + 1):
            phi_h = np.zeros((n_vars, n_vars))
            for j in range(min(h, n_lags)):
                phi_h += A[j] @ Phi_prev[h - j - 1]
            Phi_prev.append(phi_h)

            # IRF at horizon h = Phi_h @ candidate
            irf_h = phi_h @ candidate
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
