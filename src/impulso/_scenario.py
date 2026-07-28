"""Scenario-analysis engine — in-sample layers.

Owns the shared machinery behind `IdentifiedVAR.counterfactual`: back out
realised structural shocks per posterior draw, apply `ShockPath` edits, and
re-propagate through the lag recursion (ADR-0005). The forecast-side
constrain/solve layers (conditional forecasts, structural scenarios) build
on the same conventions and arrive with those methods.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from impulso.fitted import FittedVAR
    from impulso.identified import IdentifiedVAR
    from impulso.scenario import ShockPath, VariablePath


def structural_shock_context(identified: IdentifiedVAR) -> tuple[np.ndarray, np.ndarray, bool]:
    """Realised structural shocks and the matrices mapping them to residuals.

    Under time-varying volatility the per-t path is used (the correct
    in-sample object); under constant volatility a single matrix. The
    shock matrices come from the instance's memoised `shock_matrix`, so
    counterfactuals share the same structural draws as the historical
    decomposition computed on the same `IdentifiedVAR`.

    Args:
        identified: The identified VAR to back shocks out of.

    Returns:
        Tuple `(P, eps, per_t)`: `P` has shape `(C, D, T, n, n)` when
        `per_t` else `(C, D, n, n)`; `eps` has shape `(C, D, T, n)` with
        `eps_t = P_t⁻¹ u_t`.
    """
    from impulso._residuals import reduced_form_residuals

    resid = reduced_form_residuals(identified.idata.posterior, identified.data, identified.n_lags)
    per_t = identified.volatility.is_time_varying
    if per_t:
        P = identified.shock_matrix(at="all").values
        eps = np.einsum("cdtij,cdtj->cdti", np.linalg.inv(P), resid)
    else:
        P = identified.shock_matrix(at=None).values
        eps = np.einsum("cdij,cdtj->cdti", np.linalg.inv(P), resid)
    return P, eps, per_t


def apply_shock_edits(
    eps: np.ndarray,
    edits: list[ShockPath],
    index: pd.DatetimeIndex,
    shock_names: list[str],
    scheme: object,
) -> np.ndarray:
    """Apply `ShockPath` edits to realised structural shocks.

    Windows resolve against the lag-trimmed index with the
    `historical_decomposition` searchsorted convention. A window that
    resolves to zero periods raises (no silent no-op counterfactuals); a
    window partially preceding the first shock period clamps forward with
    a warning. Overlapping edits of the same shock raise. `NaN` values
    raise — in-sample edits must be concrete (`NaN` marks free entries
    only on the forecast axis).

    Args:
        eps: Realised shocks `(C, D, T, n)`; not mutated.
        edits: The `ShockPath` edits to impose.
        index: Lag-trimmed time index (`data.index[n_lags:]`).
        shock_names: Shock coordinate labels from the identification scheme.
        scheme: The identification scheme (consulted for a unit-effect
            `scale` normalisation, which warns on non-zero edits).

    Returns:
        Edited copy of `eps`.

    Raises:
        ValueError: On unknown or `unidentified_*` shock names, empty
            windows, length-mismatched value arrays, `NaN` values, or
            overlapping edits.
    """
    eps = eps.copy()
    T = eps.shape[2]
    covered: dict[int, np.ndarray] = {}

    for edit in edits:
        if edit.shock not in shock_names:
            raise ValueError(f"Unknown shock {edit.shock!r}; available shocks: {shock_names}")
        if edit.shock.startswith("unidentified_"):
            raise ValueError(
                f"Cannot edit {edit.shock!r}: unidentified shock columns are rotation-arbitrary, "
                "so their individual paths carry no economic content."
            )
        j = shock_names.index(edit.shock)
        t0, t1 = _resolve_window(edit, index, T)
        window_vals = _resolve_values(edit, t1 - t0)

        mask = covered.setdefault(j, np.zeros(T, dtype=bool))
        if mask[t0:t1].any():
            raise ValueError(f"Multiple ShockPath edits cover {edit.shock!r} at overlapping periods; merge them.")
        mask[t0:t1] = True

        if np.any(window_vals != 0.0) and getattr(scheme, "scale", None) is not None:
            # stacklevel targets the public caller of IdentifiedVAR.counterfactual
            # (user -> counterfactual -> counterfactual_paths -> here).
            warnings.warn(
                "ShockPath values are in one-standard-deviation units, but the identification "
                "scheme applies a unit-effect rescaling (scale is set); non-zero edits are not "
                "invariant to that normalisation. Zero edits are safe; for custom paths "
                "re-identify with scale=None.",
                UserWarning,
                stacklevel=4,
            )

        eps[:, :, t0:t1, j] = window_vals[np.newaxis, np.newaxis, :]

    return eps


def _resolve_window(edit: ShockPath, index: pd.DatetimeIndex, T: int) -> tuple[int, int]:
    """Resolve an edit's start/end to positions on the lag-trimmed index.

    Raises:
        ValueError: If the window resolves to zero periods.
    """
    t0 = 0
    if edit.start is not None:
        t0 = int(index.searchsorted(edit.start))
        if edit.start < index[0]:
            # stacklevel targets the public caller of IdentifiedVAR.counterfactual
            # (user -> counterfactual -> counterfactual_paths -> apply_shock_edits -> here).
            warnings.warn(
                f"ShockPath start {edit.start} precedes the first structural-shock period "
                f"({index[0]}, after lag trimming); the edit window clamps forward.",
                UserWarning,
                stacklevel=5,
            )
    t1 = T if edit.end is None else int(index.searchsorted(edit.end, side="right"))
    if t1 <= t0:
        raise ValueError(
            f"ShockPath window for {edit.shock!r} resolves to zero periods "
            f"(start={edit.start}, end={edit.end} against the sample {index[0]}..{index[-1]}); "
            "refusing a silent no-op."
        )
    return t0, t1


def _resolve_values(edit: ShockPath, window_len: int) -> np.ndarray:
    """Broadcast or validate an edit's values against its resolved window.

    Raises:
        ValueError: On a length mismatch or NaN values (in-sample edits
            must be concrete).
    """
    if isinstance(edit.values, float):
        window_vals = np.full(window_len, edit.values)
    else:
        window_vals = np.asarray(edit.values, dtype=np.float64)
        if window_vals.shape[0] != window_len:
            raise ValueError(
                f"ShockPath values for {edit.shock!r} have length {window_vals.shape[0]} "
                f"but the window resolves to {window_len} periods."
            )
    if np.isnan(window_vals).any():
        raise ValueError(
            f"ShockPath values for {edit.shock!r} contain NaN: in-sample edits must be "
            "concrete (NaN marks free entries only on the forecast axis)."
        )
    return window_vals


def counterfactual_paths(identified: IdentifiedVAR, edits: list[ShockPath]) -> np.ndarray:
    """Re-simulate the estimation sample with edited structural shocks.

    Backs out realised shocks, applies the edits, rebuilds the residual
    path `u_t = P_t eps_t`, and re-runs the lag recursion from the actual
    initial observations. With no edits this reproduces the observed
    sample exactly (up to float error) — the engine's reproduction
    identity.

    Args:
        identified: The identified VAR.
        edits: `ShockPath` edits (may be empty).

    Returns:
        Counterfactual paths of shape `(C, D, T, n)`, time-aligned with
        `data.index[n_lags:]`.
    """
    from impulso._linalg import lag_matrices
    from impulso._propagate import propagate

    posterior = identified.idata.posterior
    n_lags = identified.n_lags
    P, eps, per_t = structural_shock_context(identified)
    eps_cf = apply_shock_edits(
        eps,
        edits,
        identified.data.index[n_lags:],
        identified.shock_names,
        identified.scheme,
    )
    u_cf = np.einsum("cdtij,cdtj->cdti", P, eps_cf) if per_t else np.einsum("cdij,cdtj->cdti", P, eps_cf)

    intercept = posterior["intercept"].values  # (C, D, n)
    n_chains, n_draws, n_vars = intercept.shape
    T_eff = u_cf.shape[2]
    forcing = np.broadcast_to(intercept[:, :, np.newaxis, :], (n_chains, n_draws, T_eff, n_vars)).copy()
    if identified.data.exog is not None and "B_exog" in posterior:
        forcing += np.einsum("cdij,tj->cdti", posterior["B_exog"].values, identified.data.exog[n_lags:])
    forcing += u_cf

    A = lag_matrices(posterior["B"].values, n_lags)
    return propagate(A, forcing, identified.data.endog[:n_lags])


# --- forecast-side layers (conditional forecasts; ADR-0005 layers 2-3) ---


def resolve_variable_pins(
    conditions: list[VariablePath],
    var_names: list[str],
    steps: int,
) -> list[tuple[int, int, float]]:
    """Resolve `VariablePath` conditions to `(variable, step, value)` pins.

    A scalar broadcasts to all `steps`; an array of length `L <= steps`
    pins steps `1..L` and leaves the rest free; `NaN` entries are skipped
    (unconstrained). Duplicate `(variable, step)` references raise rather
    than silently deduping — conflicting values would otherwise make the
    constraint system inconsistent.

    Args:
        conditions: The `VariablePath` conditions.
        var_names: Endogenous variable names.
        steps: Forecast horizon.

    Returns:
        List of `(variable_index, step_index, value)` with 0-based step
        indices (step 1 → index 0).

    Raises:
        TypeError: If a condition is not a `VariablePath`.
        ValueError: On unknown variables, scalar-NaN conditions,
            over-length arrays, or duplicate pins.
    """
    from impulso.scenario import VariablePath

    pins: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for cond in conditions:
        if not isinstance(cond, VariablePath):
            raise TypeError(f"conditional forecasting accepts VariablePath conditions only, got {type(cond).__name__}")
        if cond.variable not in var_names:
            raise ValueError(f"Unknown variable {cond.variable!r}; available variables: {var_names}")
        i = var_names.index(cond.variable)
        if isinstance(cond.values, float):
            if np.isnan(cond.values):
                raise ValueError(f"VariablePath for {cond.variable!r} is a scalar NaN — it pins nothing.")
            values = np.full(steps, cond.values)
        else:
            values = np.asarray(cond.values, dtype=np.float64)
            if values.shape[0] > steps:
                raise ValueError(
                    f"VariablePath values for {cond.variable!r} have length {values.shape[0]} "
                    f"but the forecast has only {steps} steps."
                )
        for h, v in enumerate(values):
            if np.isnan(v):
                continue
            if (i, h) in seen:
                raise ValueError(f"Duplicate pin for variable {cond.variable!r} at step {h + 1}; merge the conditions.")
            seen.add((i, h))
            pins.append((i, h, float(v)))
    return pins


def conditional_forecast_engine(
    fitted: FittedVAR,
    steps: int,
    conditions: list[VariablePath],
    include_shock_uncertainty: bool,
    seed: int | np.random.Generator | None,
    exog_future: np.ndarray | None,
    path_uncertainty: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Waggoner-Zha conditional forecast via the stacked-shock solve.

    The forecast stack is `Y = b + M E` with `E` the stacked future
    structural shocks; a pin on `(variable i, step h)` contributes the
    constraint row `e'_{i,h} M` with rhs `value - b_{i,h}`. `M` is never
    materialised — only the `r` constraint rows are built, from the MA
    coefficients and the volatility process's forecast Cholesky path
    (block `(h, s) = Phi_{h-s} L_{T+s}`; the observable-space answer is
    invariant to the orthogonalisation, so the raw volatility factor is
    the correct `P` here).

    Modes: hard pins (`path_uncertainty="none"`) draw
    `E = xi - C'(CC')^{-1}(C xi - cbar)` so conditions hold pathwise;
    `path_uncertainty="unconditional"` draws `E = mu* + xi` (ADPRR
    `Omega_f = DD'`: conditions restrict the mean, bands keep their
    unconditional width). Mean mode propagates `mu*`.

    RNG contract (matched-seed nesting with `forecast()`): one
    `forecast_cholesky_path` call, then per-step `standard_normal((C, D, n))`
    draws in step order.

    Plausibility per draw: `q = cbar'(CC')^{-1} cbar` — the squared
    Mahalanobis distance of the pinned values from their unconditional
    law (`chi^2_r` reference when all shocks adjust) — plus the
    ADPRR-calibrated `q_cal = (1 + sqrt(1 - exp(-q / (n*steps)))) / 2`.

    Args:
        fitted: The fitted reduced-form VAR.
        steps: Forecast horizon.
        conditions: `VariablePath` pins (may be empty).
        include_shock_uncertainty: Density mode (draw shocks) vs mean mode.
        seed: RNG seed or Generator.
        exog_future: Future exogenous values, validated by the caller.
        path_uncertainty: `"none"` (hard pins) or `"unconditional"`.

    Returns:
        Tuple `(paths, q, q_cal, r)`: forecast paths `(C, D, steps, n)`,
        per-draw plausibility `q` and calibrated `q_cal` `(C, D)`, and the
        number of binding restrictions `r`.
    """
    from impulso._linalg import lag_matrices
    from impulso._ma import compute_ma_phi
    from impulso._propagate import propagate

    posterior = fitted.idata.posterior
    B_draws = posterior["B"].values
    n_lags = fitted.n_lags
    n_chains, n_draws, n_vars, _ = B_draws.shape
    d_total = steps * n_vars

    pins = resolve_variable_pins(conditions, fitted.var_names, steps)
    r = len(pins)

    # Deterministic path b: identically forecast()'s mean mode (consumes no RNG).
    b = fitted.forecast(steps, include_shock_uncertainty=False, exog_future=exog_future)
    b_path = b.idata.posterior_predictive["forecast"].values  # (C, D, steps, n)

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    L_path = fitted.volatility.forecast_cholesky_path(posterior, steps=steps, rng=rng)  # (C, D, steps, n, n)

    E_mean = np.zeros((n_chains, n_draws, d_total))
    q = np.zeros((n_chains, n_draws))
    C_rows = None
    G = None
    cbar = None
    if r:
        Phi = compute_ma_phi(lag_matrices(B_draws, n_lags), steps - 1)  # (C, D, steps, n, n)
        C_rows = np.zeros((n_chains, n_draws, r, d_total))
        cbar = np.zeros((n_chains, n_draws, r))
        for k, (i, h, value) in enumerate(pins):
            cbar[:, :, k] = value - b_path[:, :, h, i]
            for s in range(h + 1):
                block = np.einsum("cdk,cdkl->cdl", Phi[:, :, h - s, i, :], L_path[:, :, s])
                C_rows[:, :, k, s * n_vars : (s + 1) * n_vars] = block
        G = np.einsum("cdrk,cdsk->cdrs", C_rows, C_rows)
        if np.max(np.linalg.cond(G)) > 1e8:
            warnings.warn(
                "The pinned-path constraint system is nearly redundant (condition "
                "number of CC' exceeds 1e8) — near-duplicate pins inflate the "
                "conditional adjustment and the plausibility statistic without "
                "tripping an exact-rank error.",
                UserWarning,
                stacklevel=4,
            )
        alpha = np.linalg.solve(G, cbar[..., np.newaxis])[..., 0]  # (C, D, r)
        E_mean = np.einsum("cdrk,cdr->cdk", C_rows, alpha)
        q = np.einsum("cdr,cdr->cd", cbar, alpha)

    if include_shock_uncertainty:
        xi = np.empty((n_chains, n_draws, steps, n_vars))
        for h in range(steps):  # per-step draws, forecast()'s stream order
            xi[:, :, h, :] = rng.standard_normal((n_chains, n_draws, n_vars))
        xi_flat = xi.reshape(n_chains, n_draws, d_total)
        if r and path_uncertainty == "none":
            residual = np.einsum("cdrk,cdk->cdr", C_rows, xi_flat) - cbar
            beta = np.linalg.solve(G, residual[..., np.newaxis])[..., 0]
            E = xi_flat - np.einsum("cdrk,cdr->cdk", C_rows, beta)
        elif r:
            E = E_mean + xi_flat
        else:
            E = xi_flat
    else:
        E = E_mean

    eps = E.reshape(n_chains, n_draws, steps, n_vars)
    u = np.einsum("cdhij,cdhj->cdhi", L_path, eps)
    A = lag_matrices(B_draws, n_lags)
    deviation = propagate(A, u, np.zeros((n_lags, n_vars)))
    paths = b_path + deviation

    q_cal = (1.0 + np.sqrt(1.0 - np.exp(-q / d_total))) / 2.0
    return paths, q, q_cal, r
