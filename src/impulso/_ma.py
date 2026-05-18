"""MA coefficient recursion shared by IRF/FEVD computation and SignRestriction.

The structural identity behind impulse-response and forecast-error-variance
analysis is the moving-average representation of a VAR:

    Phi_0 = I
    Phi_h = sum_{j=1}^{min(h, p)} A_j @ Phi_{h-j},   h >= 1

where ``A_1, ..., A_p`` are the lag coefficient matrices. `compute_ma_phi`
is the single source of truth — both the posterior-batched IRF code in
`identified.py` and the per-rotation sign-restriction check in
`identification.py` delegate here.
"""

from __future__ import annotations

import numpy as np


def compute_ma_phi(A: list[np.ndarray], horizon: int) -> np.ndarray:
    """Compute MA coefficient matrices via the structural recursion.

    Numpy's ``@`` operator broadcasts over leading dims, so the helper handles
    both a single rotation and a posterior tensor of draws without branching:
    pass each ``A_j`` with shape ``(n, n)`` for a single-draw computation, or
    with shape ``(C, D, n, n)`` for a batched one.

    Args:
        A: Lag coefficient matrices ``A_1, ..., A_p`` in lag order. Each entry
            has shape ``(..., n, n)`` where the leading dims are arbitrary but
            shared across all entries. Must be non-empty.
        horizon: Highest MA horizon to compute. The result spans
            ``Phi_0`` through ``Phi_horizon`` inclusive.

    Returns:
        A stacked array with the horizon axis third-to-last. For a single
        draw: shape ``(horizon + 1, n, n)``. For a batched posterior:
        shape ``(C, D, horizon + 1, n, n)``.

    Raises:
        ValueError: If ``A`` is empty or ``horizon`` is negative.
    """
    if not A:
        raise ValueError("A must contain at least one lag coefficient matrix")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    n = A[0].shape[-1]
    leading_shape = A[0].shape[:-2]
    n_lags = len(A)

    eye = np.broadcast_to(np.eye(n), (*leading_shape, n, n)).copy()
    Phi: list[np.ndarray] = [eye]
    for h in range(1, horizon + 1):
        phi_h = np.zeros((*leading_shape, n, n))
        for j in range(min(h, n_lags)):
            phi_h = phi_h + A[j] @ Phi[h - j - 1]
        Phi.append(phi_h)

    return np.stack(Phi, axis=-3)
