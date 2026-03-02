"""Identification schemes for structural VAR analysis."""

import arviz as az
import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field


class Cholesky(BaseModel):
    """Cholesky identification scheme.

    Uses the lower-triangular Cholesky decomposition of the residual
    covariance matrix to identify structural shocks. Variable ordering
    determines the causal ordering.

    Attributes:
        ordering: Ordered list of variable names (most exogenous first).
    """

    model_config = ConfigDict(frozen=True)

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
        n_chains, n_draws, _, _ = sigma.shape

        # Reorder if needed
        perm = [var_names.index(v) for v in self.ordering]
        sigma_ordered = sigma[:, :, np.ix_(perm, perm)[0], np.ix_(perm, perm)[1]]

        # Cholesky decompose each draw
        P = np.zeros_like(sigma_ordered)
        for c in range(n_chains):
            for d in range(n_draws):
                P[c, d] = np.linalg.cholesky(sigma_ordered[c, d])

        P_da = xr.DataArray(
            P,
            dims=["chain", "draw", "shock", "response"],
            coords={"shock": self.ordering, "response": self.ordering},
        )

        new_posterior = idata.posterior.assign(structural_shock_matrix=P_da)
        return az.InferenceData(posterior=new_posterior)


class SignRestriction(BaseModel):
    """Sign restriction identification scheme.

    Uses random rotation matrices to find structural impact matrices
    satisfying sign restrictions on impulse responses.

    Attributes:
        restrictions: Dict mapping variable -> {shock_name: "+" or "-"}.
        n_rotations: Number of candidate rotations per draw.
        random_seed: Seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    restrictions: dict[str, dict[str, str]]
    n_rotations: int = Field(default=1000, ge=1)
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

        P = np.full((n_chains, n_draws, n_vars, n_vars), np.nan)

        for c in range(n_chains):
            for d in range(n_draws):
                chol = np.linalg.cholesky(sigma[c, d])
                found = False
                for _ in range(self.n_rotations):
                    Q = special_ortho_group.rvs(n_vars, random_state=rng)
                    candidate = chol @ Q
                    if self._check_restrictions(candidate, var_names, shock_names):
                        P[c, d] = candidate
                        found = True
                        break
                if not found:
                    P[c, d] = chol  # fallback to Cholesky if no valid rotation found

        coord_shocks = shock_names if len(shock_names) == n_vars else var_names
        P_da = xr.DataArray(
            P,
            dims=["chain", "draw", "response", "shock"],
            coords={"response": var_names, "shock": coord_shocks},
        )

        new_posterior = idata.posterior.assign(structural_shock_matrix=P_da)
        return az.InferenceData(posterior=new_posterior)

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
