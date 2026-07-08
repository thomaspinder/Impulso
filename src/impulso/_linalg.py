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
