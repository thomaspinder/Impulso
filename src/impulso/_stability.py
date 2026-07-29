"""Companion-form stability primitives for reduced-form VAR coefficients.

A VAR(p) is stable when every root of its companion matrix lies strictly
inside the unit circle. Stacking the lag matrices `A_1, ..., A_p` into the
first block row of a `(n*p, n*p)` matrix turns the p-th order system into a
first-order one::

    F = [[A_1, A_2, ..., A_{p-1}, A_p],
         [ I ,  0 , ...,    0   ,  0 ],
         [ 0 ,  I , ...,    0   ,  0 ],
         [ ... ...           I  ,  0 ]]

Impulso stores the lag coefficients as a single matrix `B` of shape
`(n, n*p)` whose trailing axis concatenates the lag blocks in lag order —
exactly the layout of the companion matrix's top block row — so `B` is
copied in verbatim, with no slicing. The intercept and any exogenous block
live in separate posterior variables and play no part in stability.

This module is the single source of truth for the companion form. It is
deliberately rank-agnostic in the same way as `lag_matrices` and
`sigma_from_cholesky`: only the trailing two axes are interpreted, so a
single draw `(n, n*p)` and a posterior tensor `(chain, draw, n, n*p)` both
work without branching. `companion_eigenvalues` returns the raw complex
roots rather than only their moduli, because downstream consumers (return
rates and reactivity measures borrowed from theoretical ecology) need the
imaginary parts too.
"""

from __future__ import annotations

import numpy as np


def _companion_dims(B: np.ndarray, n_lags: int) -> tuple[int, int]:
    """Validate the coefficient layout and return `(n_vars, n_vars * n_lags)`.

    Raises:
        ValueError: If `n_lags` is not positive, if the trailing axis of *B*
            is not divisible by `n_lags`, if *B* has fewer than two axes, or
            if the resulting block row is not `n_vars` rows tall.
    """
    if n_lags < 1:
        raise ValueError(f"n_lags must be positive, got {n_lags}")
    if B.ndim < 2:
        raise ValueError(f"B must have at least 2 dimensions (n, n * n_lags), got shape {B.shape}")
    n_coeffs = B.shape[-1]
    if n_coeffs % n_lags != 0:
        raise ValueError(f"B trailing axis {n_coeffs} is not divisible by n_lags {n_lags}")
    n_vars = n_coeffs // n_lags
    if B.shape[-2] != n_vars:
        raise ValueError(
            f"B must have shape (..., n, n * n_lags); trailing axis {n_coeffs} with "
            f"n_lags {n_lags} implies n = {n_vars}, but B has {B.shape[-2]} rows"
        )
    return n_vars, n_coeffs


def companion_matrix(B: np.ndarray, n_lags: int) -> np.ndarray:
    """Build the first-order companion matrix of a VAR(p).

    Args:
        B: Stacked lag coefficients with trailing shape `(n, n * n_lags)`,
            lag blocks concatenated in lag order (lag 1 first). Leading axes
            are arbitrary batch dimensions, typically `(chains, draws)`.
        n_lags: Number of lag blocks stacked along the trailing axis.

    Returns:
        Companion matrix with trailing shape `(n * n_lags, n * n_lags)` and
        the same leading batch axes as *B*. The top `n` rows are *B*
        verbatim; the sub-diagonal identity blocks shift the lag state.

    Raises:
        ValueError: If the coefficient layout is inconsistent with `n_lags`.
    """
    B = np.asarray(B)
    n_vars, m = _companion_dims(B, n_lags)
    F = np.zeros((*B.shape[:-2], m, m), dtype=np.result_type(B.dtype, np.float64))
    F[..., :n_vars, :] = B
    if n_lags > 1:
        shift = np.arange(m - n_vars)
        F[..., n_vars + shift, shift] = 1.0
    return F


def companion_eigenvalues(B: np.ndarray, n_lags: int, *, chunk_size: int = 256) -> np.ndarray:
    """Eigenvalues of the companion matrix, one set per batch element.

    Cost scales as `O(N * (n * p)**3)` for `N` batch elements, so a large
    posterior over a large system is expensive: 4000 draws of an 8x8
    companion matrix take about 0.03 s, but 200 draws of a 240x240 one take
    about 2 s and allocate roughly 90 MB. The batch is therefore processed in
    chunks, and callers holding many draws should thin them first.

    Args:
        B: Stacked lag coefficients with trailing shape `(n, n * n_lags)`.
        n_lags: Number of lag blocks stacked along the trailing axis.
        chunk_size: Number of companion matrices to materialise at once.
            Bounds peak memory; it does not change the result.

    Returns:
        Complex array with trailing axis of length `n * n_lags` and the same
        leading batch axes as *B*. Eigenvalues are unordered within each set,
        as returned by `numpy.linalg.eigvals`.

    Raises:
        ValueError: If the coefficient layout is inconsistent with `n_lags`,
            or if `chunk_size` is not positive.
    """
    B = np.asarray(B)
    n_vars, m = _companion_dims(B, n_lags)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    leading = B.shape[:-2]
    flat = B.reshape(-1, n_vars, m)
    out = np.empty((flat.shape[0], m), dtype=np.complex128)
    for start in range(0, flat.shape[0], chunk_size):
        stop = start + chunk_size
        out[start:stop] = np.linalg.eigvals(companion_matrix(flat[start:stop], n_lags))
    return out.reshape(*leading, m)


def spectral_radius(B: np.ndarray, n_lags: int, *, chunk_size: int = 256) -> np.ndarray:
    """Largest companion-matrix eigenvalue modulus, one value per batch element.

    A draw is stable when its spectral radius is strictly below 1 and
    explosive at or above it.

    Args:
        B: Stacked lag coefficients with trailing shape `(n, n * n_lags)`.
        n_lags: Number of lag blocks stacked along the trailing axis.
        chunk_size: Forwarded to `companion_eigenvalues`.

    Returns:
        Real array with the same shape as *B*'s leading batch axes — a scalar
        (0-d) array for a single draw, `(chains, draws)` for a posterior.

    Raises:
        ValueError: If the coefficient layout is inconsistent with `n_lags`,
            or if `chunk_size` is not positive.
    """
    eigenvalues = companion_eigenvalues(B, n_lags, chunk_size=chunk_size)
    return np.max(np.abs(eigenvalues), axis=-1)
