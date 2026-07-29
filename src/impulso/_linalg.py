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


def companion_spectral_radius(A: list[np.ndarray]) -> np.ndarray:
    """Largest absolute eigenvalue of the VAR companion matrix, per draw.

    The companion form stacks the lag matrices in its top block-row and
    carries an identity subdiagonal. Its spectral radius is below one
    exactly when the VAR is stable, i.e. when the moving-average sum
    `sum_h Phi_h` converges and the spectral density is finite at every
    frequency.

    Args:
        A: Lag coefficient matrices `A_1, ..., A_p` in lag order, each of
            shape `(..., n, n)` with shared leading batch axes.

    Returns:
        Spectral radii with the shared leading batch shape, e.g.
        `(chains, draws)`.
    """
    n = A[0].shape[-1]
    n_lags = len(A)
    leading = A[0].shape[:-2]
    companion = np.zeros((*leading, n * n_lags, n * n_lags))
    companion[..., :n, :] = np.concatenate(A, axis=-1)
    if n_lags > 1:
        sub = np.arange(n * (n_lags - 1))
        companion[..., n + sub, sub] = 1.0
    return np.abs(np.linalg.eigvals(companion)).max(axis=-1)


def householder_from_e1(q1: np.ndarray) -> np.ndarray:
    """Orthogonal completion of a unit vector, as a Householder reflection.

    Returns the symmetric orthogonal matrix `Q = I - 2 w w' / (w'w)` with
    `w = q1 - e1`, which maps `e1` to `q1` and therefore has `q1` as its
    first column. The remaining columns are an arbitrary but deterministic
    orthonormal completion of `q1`.

    Identification schemes that pin down a single structural column use
    this to build a full invertible `P = L Q` satisfying `P P' = Sigma`:
    column 0 is the identified direction and columns 1.. are the
    rotation-arbitrary remainder (labelled `unidentified_*` downstream).

    Args:
        q1: Unit vector(s), shape `(..., n)`. Leading axes are batch axes,
            typically `(chains, draws)`. Not renormalised — callers are
            expected to pass a normalised direction.

    Returns:
        Orthogonal matrices of shape `(..., n, n)` whose first column is
        `q1`. When `q1` is already `e1` (so `w` vanishes) the identity is
        returned, which satisfies the same contract.
    """
    n = q1.shape[-1]
    e1 = np.zeros(n)
    e1[0] = 1.0
    w = q1 - e1
    w_norm2 = np.einsum("...i,...i->...", w, w)[..., np.newaxis, np.newaxis]
    outer = w[..., :, np.newaxis] * w[..., np.newaxis, :]
    eye = np.broadcast_to(np.eye(n), outer.shape)
    usable = w_norm2 > 1e-14
    return np.where(usable, eye - 2.0 * outer / np.where(usable, w_norm2, 1.0), eye)
