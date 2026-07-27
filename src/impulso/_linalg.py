"""Linear-algebra helpers shared across the Impulso pipeline.

These are rank-agnostic utilities that do not vary across adapters
and therefore live outside the ``VolatilityProcess`` seam.
"""

import numpy as np


def sigma_from_cholesky(L: np.ndarray) -> np.ndarray:
    """Reconstruct the covariance matrix from its lower-triangular Cholesky factor.

    Computes ``Sigma = L @ L.T`` over arbitrary leading batch dimensions
    using a single ellipsis einsum.

    Args:
        L: Lower-triangular Cholesky factor.  The last two dimensions are
            ``(n, n)``; all preceding dimensions are batch axes.  Common
            shapes include ``(chains, draws, n, n)`` (constant volatility)
            and ``(chains, draws, T, n, n)`` (time-varying volatility).

    Returns:
        Symmetric positive-(semi)definite covariance matrix with the same
        shape as *L*.
    """
    return np.einsum("...ij,...kj->...ik", L, L)


def lag_matrices(B: np.ndarray, n_lags: int) -> list[np.ndarray]:
    """Split a stacked VAR coefficient matrix into its per-lag blocks.

    Impulso stores the reduced-form coefficients as a single matrix whose
    trailing axis concatenates the lag blocks in lag order, matching the
    regressor layout built in `VAR.fit` (lag 1 first). This helper recovers
    the individual lag matrices `A_1, ..., A_p` that the moving-average
    recursion in `compute_ma_phi` consumes.

    Like `sigma_from_cholesky`, the split is rank-agnostic: only the last
    axis is partitioned, so leading batch axes pass through untouched.

    Args:
        B: Stacked coefficient matrix. The last two dimensions are
            `(n, n * n_lags)`; all preceding dimensions are batch axes.
            Common shapes are `(n, n * n_lags)` for a single draw and
            `(chains, draws, n, n * n_lags)` for a posterior tensor.
        n_lags: Number of lag blocks stacked along the trailing axis.
            Must be positive and divide the trailing axis exactly.

    Returns:
        List of `n_lags` arrays in lag order, `A_1` first. Each entry has
        the same leading batch axes as *B* and trailing shape `(n, n)`.

    Raises:
        ValueError: If `n_lags` is not positive, or if the trailing axis of
            *B* is not divisible by `n_lags`.
    """
    if n_lags < 1:
        raise ValueError(f"n_lags must be positive, got {n_lags}")
    n_coeffs = B.shape[-1]
    if n_coeffs % n_lags != 0:
        raise ValueError(f"B trailing axis {n_coeffs} is not divisible by n_lags {n_lags}")
    n_vars = n_coeffs // n_lags
    return [B[..., j * n_vars : (j + 1) * n_vars] for j in range(n_lags)]
