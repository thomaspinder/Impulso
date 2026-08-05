"""Recursive (Cholesky) identification."""

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from impulso._base import ImpulsoModel

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

        When `self.ordering` matches `var_names` this is a no-op and `L` is
        returned unchanged. Otherwise the factor is re-derived so that it is
        lower-triangular in the requested causal ordering, then written back
        into the data's row order.

        Concretely, with `Pi` the permutation sending data order to
        `self.ordering`, the ordered factor is the LQ factor of `Pi @ L`:
        `qr((Pi @ L).T) = Q @ R` gives `G = R.T` lower-triangular with
        `G @ G.T = Pi @ Sigma @ Pi.T`. Columns are sign-fixed so `G` has a
        positive diagonal, matching the textbook `cholesky(Pi Sigma Pi.T)`.
        `Sigma` is never formed, so the conditioning of the decomposition is
        not squared.

        Args:
            L: Lower-triangular Cholesky factor, shape (..., n_vars, n_vars).
            var_names: Variable names in the data's natural order.
            posterior: Unused. Accepted for Protocol uniformity.
            data: Unused. Accepted for Protocol uniformity.
            n_lags: Unused. Accepted for Protocol uniformity.

        Returns:
            Structural shock matrix, same shape as `L`. Rows follow
            `var_names` (the data's order, matching the `response`
            coordinate downstream); columns follow `shock_coords`, i.e.
            `self.ordering`. Triangularity therefore holds in the *ordering*
            row coordinates: permuting the rows by `self.ordering` recovers
            an exactly lower-triangular factor with a positive diagonal.
        """
        del posterior, data, n_lags  # unused

        # Fast path: ordering matches data — identify is a no-op.
        if list(self.ordering) == list(var_names):
            return L

        perm = np.array([var_names.index(v) for v in self.ordering])
        inv = np.argsort(perm)

        # (Pi L)(Pi L).T = Pi Sigma Pi.T, so any lower-triangular factor of
        # Pi L is a Cholesky factor of the permuted covariance.
        L_ord = L[..., perm, :]
        _, R = np.linalg.qr(np.swapaxes(L_ord, -1, -2))
        signs = np.sign(np.diagonal(R, axis1=-2, axis2=-1))
        signs = np.where(signs == 0.0, 1.0, signs)
        P_ord = np.swapaxes(R, -1, -2) * signs[..., np.newaxis, :]

        return P_ord[..., inv, :]  # back to data row order

    def shock_coords(self, n_vars: int) -> list[str]:
        """Cholesky shock labels are simply the causal ordering."""
        del n_vars  # ordering already has the right length
        return list(self.ordering)
