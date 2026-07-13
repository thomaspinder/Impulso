"""Conditional-forecast helper for the post-March-2020 tutorial.

Imported by ``docs/tutorials/post-march-2020.py``; the leading underscore keeps
it out of the Sphinx docs build (``tutorials/_*.py`` is excluded).

The routine implements analytic conditional forecasting for a fitted
reduced-form VAR (Waggoner & Zha 1999; Lenza & Primiceri 2020): one variable's
future path is pinned to an imposed trajectory and the implied structural
shocks are propagated to every variable. Each posterior draw's forecast path is
written as a block-lower-triangular linear map from standardised shocks
``z``, ``y_flat = mean + M @ z``, and the conditioned rows are solved exactly
via a minimum-norm Gaussian projection. Pure NumPy — no PyMC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from impulso._ma import compute_ma_phi

if TYPE_CHECKING:
    from impulso.fitted import FittedVAR


def conditional_forecast(
    fitted: "FittedVAR",
    history: np.ndarray,
    unemployment_path: np.ndarray,
    *,
    index: int,
    seed: int,
) -> np.ndarray:
    """Forecast a VAR conditional on an imposed path for one variable.

    For every posterior draw the reduced-form forecast is expressed through its
    moving-average representation as ``y_flat = mean + M @ z``, where ``mean``
    is the no-shock (deterministic) path seeded from ``history``, ``z`` is a
    stack of standardised forecast shocks, and ``M`` is the block
    lower-triangular map ``M[h, s] = Phi[h - s] @ L_s`` built from the draw's MA
    coefficients ``Phi`` and its forecast Cholesky factors ``L_s``. The rows of
    ``M`` that read out the conditioned variable form the selection matrix
    ``A``; the shock stack is projected onto the constraint
    ``A @ z_cond = target - mean_sel`` by the minimum-norm update
    ``z_cond = z + A.T @ solve(A A.T, target - mean_sel - A @ z)``. The
    conditioned variable then reproduces ``unemployment_path`` exactly while the
    remaining variables respond through the same correlated shocks.

    Args:
        fitted: Fitted reduced-form VAR. Supplies posterior lag coefficients
            (``B``), intercepts, the forecast Cholesky path (via its volatility
            adapter) and the lag order.
        history: Recent observations, shape ``(>= n_lags, n_vars)``. The last
            ``fitted.n_lags`` rows seed the no-shock mean path.
        unemployment_path: Imposed trajectory for the conditioned variable,
            shape ``(horizon,)``. Its length sets the forecast horizon.
        index: Column index of the conditioned variable in the VAR.
        seed: Seed for the forecast-shock RNG (deterministic output).

    Returns:
        Conditional forecast draws of shape ``(chain, draw, horizon, n_vars)``.
    """
    horizon = int(np.asarray(unemployment_path).shape[0])
    n_lags = fitted.n_lags
    B = fitted.coefficients  # (C, D, n_vars, n_vars * n_lags)
    intercept = fitted.intercepts  # (C, D, n_vars)
    n_chains, n_draws, n_vars, _ = B.shape

    rng = np.random.default_rng(seed)

    # Forecast Cholesky factors L_0..L_{H-1} per draw: (C, D, H, n_vars, n_vars).
    L_path = fitted.volatility.forecast_cholesky_path(fitted.idata.posterior, steps=horizon, rng=rng)

    # No-shock mean path from the supplied history, propagated per draw. Mirrors
    # FittedVAR.forecast's mean mode but seeded from `history`, not fitted.data.
    hist = np.asarray(history, dtype=float)[-n_lags:]  # (n_lags, n_vars)
    buffer = np.broadcast_to(hist, (n_chains, n_draws, n_lags, n_vars)).copy()
    mean = np.zeros((n_chains, n_draws, horizon, n_vars))
    for h in range(horizon):
        x_lag = np.concatenate([buffer[:, :, -(lag + 1), :] for lag in range(n_lags)], axis=-1)
        y_new = intercept + np.einsum("cdij,cdj->cdi", B, x_lag)
        mean[:, :, h, :] = y_new
        buffer = np.concatenate([buffer[:, :, 1:, :], y_new[:, :, np.newaxis, :]], axis=2)

    # Standardised forecast shocks z ~ N(0, I), stacked over the horizon per draw.
    z_all = rng.standard_normal((n_chains, n_draws, horizon * n_vars))

    # Flat-index rows that read out the conditioned variable at each horizon.
    sel_rows = index + n_vars * np.arange(horizon)  # (H,)
    target = np.asarray(unemployment_path, dtype=float)  # (H,)

    out = np.empty((n_chains, n_draws, horizon, n_vars))
    for c in range(n_chains):
        for d in range(n_draws):
            # Lag matrices A_1..A_p -> MA coefficients Phi_0..Phi_{H-1}.
            a_lags = [B[c, d, :, j * n_vars : (j + 1) * n_vars] for j in range(n_lags)]
            phi = compute_ma_phi(a_lags, horizon - 1)  # (H, n_vars, n_vars)
            l_draw = L_path[c, d]  # (H, n_vars, n_vars)

            # Block lower-triangular map M[h, s] = Phi[h - s] @ L_s.
            m = np.zeros((horizon * n_vars, horizon * n_vars))
            for h in range(horizon):
                for s in range(h + 1):
                    m[h * n_vars : (h + 1) * n_vars, s * n_vars : (s + 1) * n_vars] = phi[h - s] @ l_draw[s]

            mean_flat = mean[c, d].reshape(-1)  # (H * n_vars,)
            z = z_all[c, d]  # (H * n_vars,)

            # Minimum-norm projection onto A @ z_cond = target - mean_sel.
            a_sel = m[sel_rows, :]  # (H, H * n_vars)
            mean_sel = mean_flat[sel_rows]  # (H,)
            gram = a_sel @ a_sel.T  # (H, H)
            gram = 0.5 * (gram + gram.T)  # symmetrise; never invert directly
            rhs = target - mean_sel - a_sel @ z
            z_cond = z + a_sel.T @ np.linalg.solve(gram, rhs)

            out[c, d] = (mean_flat + m @ z_cond).reshape(horizon, n_vars)

    return out
