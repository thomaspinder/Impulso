"""Shared lag-recursion propagation engine for scenario analysis.

Historical decomposition, historical counterfactuals, and the forecast-side
scenario methods all reduce to one operation: push a forcing path through
the VAR's lag dynamics,

    y_t = f_t + sum_{i=1}^{p} A_i @ y_{t-i},

vectorised over posterior draws (see ADR-0005). `propagate` runs the
recursion from given initial observations — with the forcing set to
intercept + exogenous contributions + residuals it reproduces the observed
sample exactly, and with the residuals dropped it yields the deterministic
baseline. `propagate_contributions` runs the same recursion per structural
shock from zero initial conditions, which yields the propagated shock
contributions of the historical decomposition and, by linearity, of
counterfactual deltas.
"""

from __future__ import annotations

import numpy as np


def propagate(A: list[np.ndarray], forcing: np.ndarray, y_init: np.ndarray) -> np.ndarray:
    """Run the VAR lag recursion `y_t = f_t + sum_i A_i y_{t-i}`.

    Args:
        A: Lag coefficient matrices `A_1, ..., A_p` in lag order, each of
            shape `(C, D, n, n)` (from `lag_matrices`).
        forcing: Forcing path `f_t` of shape `(C, D, T, n)` — whatever
            enters the recursion additively at each `t` (intercept,
            exogenous contributions, residuals).
        y_init: Initial observations of shape `(p, n)` (broadcast across
            draws) or `(C, D, p, n)`, ordered chronologically so that
            `y_init[..., -1, :]` immediately precedes `forcing[..., 0, :]`.

    Returns:
        Propagated path of shape `(C, D, T, n)`.

    Raises:
        ValueError: If `A` is empty, `forcing` is not 4-D, or `y_init`
            does not carry `(p, n)` trailing dimensions.
    """
    if not A:
        raise ValueError("A must contain at least one lag coefficient matrix")
    n_lags = len(A)
    n_chains, n_draws, n_vars, _ = A[0].shape
    if forcing.ndim != 4:
        raise ValueError(f"forcing must be 4-D (chains, draws, T, n_vars), got {forcing.ndim}-D")
    if y_init.shape[-2:] != (n_lags, n_vars):
        raise ValueError(f"y_init trailing shape must be ({n_lags}, {n_vars}), got {y_init.shape[-2:]}")
    T = forcing.shape[2]

    ext = np.empty((n_chains, n_draws, n_lags + T, n_vars))
    ext[:, :, :n_lags, :] = np.broadcast_to(y_init, (n_chains, n_draws, n_lags, n_vars))
    for t in range(T):
        y_t = forcing[:, :, t, :].copy()
        for i, A_i in enumerate(A):
            y_t += np.einsum("cdij,cdj->cdi", A_i, ext[:, :, n_lags + t - 1 - i, :])
        ext[:, :, n_lags + t, :] = y_t
    return ext[:, :, n_lags:, :]


def propagate_contributions(A: list[np.ndarray], impact: np.ndarray) -> np.ndarray:
    """Propagate per-shock impact terms through the lag dynamics.

    Runs `c_{j,t} = impact_{j,t} + sum_i A_i c_{j,t-i}` from zero initial
    conditions, carrying the shock axis through the recursion. With
    `impact[..., t, :, j] = P_t[:, j] * eps_{j,t}` this yields the
    propagated historical-decomposition contributions: summed over shocks
    and added to the deterministic baseline they reproduce the observed
    series exactly.

    Args:
        A: Lag coefficient matrices `A_1, ..., A_p`, each `(C, D, n, n)`.
        impact: Contemporaneous impact of shape `(C, D, T, n, S)` where
            `S` is the shock axis.

    Returns:
        Propagated contributions of shape `(C, D, T, n, S)`.

    Raises:
        ValueError: If `A` is empty or `impact` is not 5-D.
    """
    if not A:
        raise ValueError("A must contain at least one lag coefficient matrix")
    if impact.ndim != 5:
        raise ValueError(f"impact must be 5-D (chains, draws, T, n_vars, n_shocks), got {impact.ndim}-D")
    n_lags = len(A)
    n_chains, n_draws, T, n_vars, n_shocks = impact.shape

    ext = np.zeros((n_chains, n_draws, n_lags + T, n_vars, n_shocks))
    for t in range(T):
        c_t = impact[:, :, t, :, :].copy()
        for i, A_i in enumerate(A):
            c_t += np.einsum("cdij,cdjs->cdis", A_i, ext[:, :, n_lags + t - 1 - i, :, :])
        ext[:, :, n_lags + t, :, :] = c_t
    return ext[:, :, n_lags:, :, :]
