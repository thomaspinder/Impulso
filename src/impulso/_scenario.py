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


def _require_finite_shock_matrix(P: np.ndarray) -> None:
    """Reject `NaN` structural shock matrices at the engines' entry seam.

    Both engines are linear solves in `P`: the in-sample one inverts it
    (`eps_t = P_t⁻¹ u_t`, which silently yields an all-`NaN`
    counterfactual), the forecast one takes the numerical rank of
    constraint rows built from it (which dies inside LAPACK as `SVD did
    not converge`). Neither failure names the cause, so the check runs
    once where the matrix enters — `structural_shock_context` for
    `counterfactual`, `_forecast_shock_matrices` for
    `structural_scenario`.

    Args:
        P: Structural shock matrices, `(C, D, ..., n, n)`. The leading two
            axes are chain and draw.

    Raises:
        ValueError: If any draw carries a `NaN`.
    """
    nan_mask = np.isnan(P)
    if not nan_mask.any():
        return
    bad = nan_mask.reshape(P.shape[0], P.shape[1], -1).any(axis=-1)
    raise ValueError(
        f"The structural shock matrix is NaN for {int(bad.sum())}/{bad.size} posterior draws, "
        "and scenario analysis needs a finite P on every draw (its solves would otherwise fail "
        "opaquely inside LAPACK or return all-NaN paths). The usual cause is an identification "
        "scheme running with on_undefined='nan', which blanks the draws where its restrictions "
        "leave the shock matrix undefined — e.g. LongRunRestriction on a draw whose long-run "
        "multiplier C(1) is numerically singular. Re-identify with on_undefined='raise' to catch "
        "those draws at identification time, where the diagnostics say how many there are and why."
    )


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

    Raises:
        ValueError: If any posterior draw's shock matrix carries a `NaN`.
    """
    from impulso._residuals import reduced_form_residuals

    resid = reduced_form_residuals(identified.idata.posterior, identified.data, identified.n_lags)
    per_t = identified.volatility.is_time_varying
    P = identified.shock_matrix(at="all" if per_t else None).values
    _require_finite_shock_matrix(P)
    if per_t:
        eps = np.einsum("cdtij,cdtj->cdti", np.linalg.inv(P), resid)
    else:
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
    # Mean mode with no pins needs no volatility path and must consume no
    # randomness (forecast()'s mean-mode contract). With pins, the constraint
    # rows need L; under time-varying volatility that conditions on one
    # simulated path per draw (documented on the method).
    L_path = None
    if include_shock_uncertainty or r:
        L_path = fitted.volatility.forecast_cholesky_path(posterior, steps=steps, rng=rng)  # (C, D, steps, n, n)

    E_mean = np.zeros((n_chains, n_draws, d_total))
    q = np.zeros((n_chains, n_draws))
    C_rows = None
    G = None
    cbar = None
    if r:
        Phi = compute_ma_phi(lag_matrices(B_draws, n_lags), steps - 1)  # (C, D, steps, n, n)
        C_rows, G, cbar = _constraint_system(pins, b_path, Phi, L_path, n_vars, d_total)
        alpha = np.linalg.solve(G, cbar[..., np.newaxis])[..., 0]  # (C, D, r)
        E_mean = np.einsum("cdrk,cdr->cdk", C_rows, alpha)
        q = np.einsum("cdr,cdr->cd", cbar, alpha)

    if include_shock_uncertainty:
        xi = np.empty((n_chains, n_draws, steps, n_vars))
        # Gaussian by construction: the Waggoner-Zha / ADPRR solves below are
        # Gaussian conditioning results. Heavy-tailed error distributions are
        # rejected upstream in FittedVAR.conditional_forecast, so this does
        # NOT route through the error_dist seam the way forecast() does.
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

    if L_path is None:
        paths = b_path  # mean mode, no pins: the deterministic path itself
    else:
        eps = E.reshape(n_chains, n_draws, steps, n_vars)
        u = np.einsum("cdhij,cdhj->cdhi", L_path, eps)
        A = lag_matrices(B_draws, n_lags)
        deviation = propagate(A, u, np.zeros((n_lags, n_vars)))
        paths = b_path + deviation

    return paths, q, _calibrate_q(q, r, path_uncertainty, d_total), r


def _constraint_system(
    pins: list[tuple[int, int, float]],
    b_path: np.ndarray,
    Phi: np.ndarray,
    L_path: np.ndarray,
    n_vars: int,
    d_total: int,
    warn_on_ill_conditioning: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the constraint rows `C`, Gram matrix `CC'`, and rhs `cbar`.

    Row `k` for pin `(i, h, value)` holds block `(Phi_{h-s} L_{T+s})[i, :]`
    at stacked positions `s*n..(s+1)*n` for `s <= h`; the rhs is
    `value - b_{i,h}`. Warns when `CC'` is nearly singular unless the
    caller solves a column-restricted system and runs its own check on
    the restricted Gram (`warn_on_ill_conditioning=False`).
    """
    n_chains, n_draws = b_path.shape[:2]
    r = len(pins)
    C_rows = np.zeros((n_chains, n_draws, r, d_total))
    cbar = np.zeros((n_chains, n_draws, r))
    for k, (i, h, value) in enumerate(pins):
        cbar[:, :, k] = value - b_path[:, :, h, i]
        for s in range(h + 1):
            block = np.einsum("cdk,cdkl->cdl", Phi[:, :, h - s, i, :], L_path[:, :, s])
            C_rows[:, :, k, s * n_vars : (s + 1) * n_vars] = block
    G = np.einsum("cdrk,cdsk->cdrs", C_rows, C_rows)
    if warn_on_ill_conditioning and np.max(np.linalg.cond(G)) > 1e8:
        # stacklevel targets the public caller of conditional_forecast
        # (user -> conditional_forecast -> engine -> here -> warn).
        warnings.warn(
            "The pinned-path constraint system is nearly redundant (condition "
            "number of CC' exceeds 1e8) — near-duplicate pins inflate the "
            "conditional adjustment and the plausibility statistic without "
            "tripping an exact-rank error.",
            UserWarning,
            stacklevel=4,
        )
    return C_rows, G, cbar


def _calibrate_q(q: np.ndarray, r: int, path_uncertainty: str, d_total: int) -> np.ndarray:
    """ADPRR-calibrated plausibility `q_cal` on `[0.5, 1]`.

    Finite only under `Omega_f = DD'` (`path_uncertainty="unconditional"`,
    where the divergence collapses to `z = q/2`). Under hard pins the
    divergence is analytically infinite and ADPRR's calibrated statistic
    sits at its ceiling of 1; with no restrictions it floors at 0.5.
    """
    if r == 0:
        return np.full_like(q, 0.5)
    if path_uncertainty == "unconditional":
        return (1.0 + np.sqrt(1.0 - np.exp(-q / d_total))) / 2.0
    return np.ones_like(q)


# --- structural scenarios (three-way partition; ADR-0005 layer 3) ---


def resolve_shock_prescriptions(
    shocks: list[ShockPath],
    shock_names: list[str],
    steps: int,
) -> list[tuple[int, int, float]]:
    """Resolve forecast-side `ShockPath` prescriptions to `(shock, step, value)`.

    Forecast-side prescriptions are positional from step 1 and must not
    carry `start`/`end` (those are in-sample-only); a scalar broadcasts to
    all steps, an array of length `L <= steps` prescribes steps `1..L`,
    `NaN` entries are free. Duplicates, unknown shocks, and
    `unidentified_*` columns raise.

    Returns:
        List of `(shock_index, step_index, value)` with 0-based steps.
    """
    from impulso.scenario import ShockPath

    prescriptions: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for path in shocks:
        if not isinstance(path, ShockPath):
            raise TypeError(f"structural_scenario prescriptions must be ShockPath, got {type(path).__name__}")
        if path.start is not None or path.end is not None:
            raise ValueError(
                f"ShockPath for {path.shock!r} carries start/end, which are in-sample-only "
                "(counterfactual windows); forecast-side prescriptions are positional from step 1."
            )
        if path.shock not in shock_names:
            raise ValueError(f"Unknown shock {path.shock!r}; available shocks: {shock_names}")
        if path.shock.startswith("unidentified_"):
            raise ValueError(f"Cannot prescribe {path.shock!r}: unidentified shock columns are rotation-arbitrary.")
        j = shock_names.index(path.shock)
        for h, v in enumerate(_prescription_values(path, steps)):
            if np.isnan(v):
                continue
            if (j, h) in seen:
                raise ValueError(f"Duplicate prescription for {path.shock!r} at step {h + 1}; merge the paths.")
            seen.add((j, h))
            prescriptions.append((j, h, float(v)))
    return prescriptions


def _prescription_values(path: ShockPath, steps: int) -> np.ndarray:
    """Broadcast or validate a prescription's values against the horizon."""
    if isinstance(path.values, float):
        if np.isnan(path.values):
            raise ValueError(f"ShockPath for {path.shock!r} is a scalar NaN — it prescribes nothing.")
        return np.full(steps, path.values)
    values = np.asarray(path.values, dtype=np.float64)
    if values.shape[0] > steps:
        raise ValueError(
            f"ShockPath values for {path.shock!r} have length {values.shape[0]} "
            f"but the forecast has only {steps} steps."
        )
    return values


def _resolve_adjusting(adjusting: list[str] | None, shock_names: list[str]) -> list[int]:
    """Resolve the adjusting set to shock indices (default: all shocks).

    The set must contain none or all of the `unidentified_*` columns — a
    proper subset would make the scenario depend on the arbitrary
    orthogonal completion (the full block is completion-invariant, the
    same logic as the HD remainder collapse).
    """
    if adjusting is None:
        return list(range(len(shock_names)))
    indices: list[int] = []
    for name in adjusting:
        if name not in shock_names:
            raise ValueError(f"Unknown adjusting shock {name!r}; available shocks: {shock_names}")
        indices.append(shock_names.index(name))
    unident_all = {i for i, s in enumerate(shock_names) if s.startswith("unidentified_")}
    unident_in = unident_all & set(indices)
    if unident_in and unident_in != unident_all:
        raise ValueError(
            "The adjusting set must contain none or all of the unidentified_* columns: a proper "
            "subset makes the scenario depend on the arbitrary orthogonal completion."
        )
    return sorted(set(indices))


def _forecast_shock_matrices(identified: IdentifiedVAR, steps: int, rng: np.random.Generator) -> np.ndarray:
    """Scheme-identified structural matrices for the forecast horizon.

    Constant volatility broadcasts the memoised `shock_matrix(at=None)` —
    never a fresh `identify()` call, so rotation draws stay shared with
    counterfactual/HD on the same instance. Time-varying volatility
    identifies each forecast Cholesky slice; rotation-sampling schemes
    (the `_samples_rotations` capability flag, e.g. `SignRestriction`)
    cannot yet pin one rotation per draw across steps and error there.

    Raises:
        ValueError: On a rotation-sampling scheme under time-varying
            volatility, or if any posterior draw's matrix carries a `NaN`.
    """
    posterior = identified.idata.posterior
    if not identified.volatility.is_time_varying:
        P = identified.shock_matrix(at=None).values  # (C, D, n, n)
        P_path = np.broadcast_to(P[:, :, np.newaxis, :, :], (*P.shape[:2], steps, *P.shape[2:])).copy()
    else:
        if getattr(identified.scheme, "_samples_rotations", False):
            raise ValueError(
                f"structural_scenario under time-varying volatility is not supported for "
                f"rotation-sampling schemes ({type(identified.scheme).__name__}): rotations are "
                "re-sampled per identify() call, so no single structural coordinate system "
                "spans the forecast steps. Use a rotation-free scheme (Cholesky, ProxySVAR), "
                "or constant volatility."
            )
        L_path = identified.volatility.forecast_cholesky_path(posterior, steps=steps, rng=rng)
        P_path = np.zeros_like(L_path)
        for h in range(steps):
            P_path[:, :, h] = identified.scheme.identify(
                L_path[:, :, h],
                identified.var_names,
                posterior=posterior,
                data=identified.data,
                n_lags=identified.n_lags,
            )
    _require_finite_shock_matrix(P_path)
    return P_path


def _scenario_feasibility(
    pins: list[tuple[int, int, float]],
    entries_A: list[tuple[int, int]],
    n_prescribed: int,
    steps: int,
) -> None:
    """Draw-independent feasibility checks (design section 2, two-tier).

    Errors when the conditions outnumber the effective adjusting entries,
    globally or in any leading horizon block (conditions at step `h` load
    only on adjusting entries at steps `<= h`, by block-triangularity).
    """
    r = len(pins)
    if r > len(entries_A):
        raise ValueError(
            f"{r} condition(s) but only {len(entries_A)} effective adjusting entr(ies) "
            f"({n_prescribed} prescribed entr(ies) consumed adjusting capacity); the "
            "scenario is over-determined. Widen the adjusting set or drop conditions."
        )
    for h in range(steps):
        pins_leading = sum(1 for (_, ph, _) in pins if ph <= h)
        adjusting_leading = sum(1 for (_, ah) in entries_A if ah <= h)
        if pins_leading > adjusting_leading:
            raise ValueError(
                f"Conditions through step {h + 1} ({pins_leading}) exceed the adjusting "
                f"entries available at steps <= {h + 1} ({adjusting_leading}); conditions "
                "at a step load only on adjusting shocks at that step or earlier "
                "(block-triangularity). Widen the adjusting set or move conditions later."
            )


def structural_scenario_engine(
    identified: IdentifiedVAR,
    steps: int,
    conditions: list[VariablePath],
    shocks: list[ShockPath],
    adjusting: list[str] | None,
    include_shock_uncertainty: bool,
    seed: int | np.random.Generator | None,
    exog_future: np.ndarray | None,
    path_uncertainty: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """ADPRR structural scenario via the three-way partition solve.

    Stacked shock entries split into `D` (prescribed — substituted at
    their `ShockPath` values, never constraint rows), `A` (adjusting
    entries not in `D` — absorb the `VariablePath` conditions), and `F`
    (free — retain unconditional draws). Per free-block draw the solve is
    `C_A E_A = cbar - C_D v_S - C_F eps_F` with the conditional-Gaussian
    resolution of under-determination; feasibility is checked once at
    validation (counting) and per draw (numerical rank of `C_A`).

    Plausibility per draw: `q = ctilde'(C_A C_A')^{-1} ctilde + |v_S|^2`
    — the prescribed shocks' own magnitude registers even though they are
    substituted. The ADPRR calibration is finite only under
    `path_uncertainty="unconditional"` with no prescriptions.

    Returns:
        Tuple `(paths, q, q_cond, q_cal, r)`: as in
        `conditional_forecast_engine` plus `q_cond`, the condition-only
        part of the plausibility (the `chi^2_r`-referenced quantity —
        the prescribed `|v_S|^2` term has no chi-squared reference).
    """
    from impulso._linalg import lag_matrices
    from impulso._ma import compute_ma_phi
    from impulso._propagate import propagate

    posterior = identified.idata.posterior
    B_draws = posterior["B"].values
    n_lags = identified.n_lags
    n_chains, n_draws, n_vars, _ = B_draws.shape
    d_total = steps * n_vars
    shock_names = identified.shock_names

    pins = resolve_variable_pins(conditions, identified.var_names, steps)
    prescriptions = resolve_shock_prescriptions(shocks, shock_names, steps)
    adjusting_idx = _resolve_adjusting(adjusting, shock_names)
    r = len(pins)

    if any(v != 0.0 for (_, _, v) in prescriptions) and getattr(identified.scheme, "scale", None) is not None:
        # stacklevel targets the public caller of structural_scenario
        # (user -> structural_scenario -> engine -> warn).
        warnings.warn(
            "ShockPath values are in one-standard-deviation units, but the identification "
            "scheme applies a unit-effect rescaling (scale is set); non-zero prescriptions "
            "are not invariant to that normalisation and neither is their plausibility "
            "contribution. Zero prescriptions are safe; for custom paths re-identify with "
            "scale=None.",
            UserWarning,
            stacklevel=3,
        )

    flat = lambda j, h: h * n_vars + j
    D_entries = {(j, h) for (j, h, _) in prescriptions}
    entries_A = [(j, h) for h in range(steps) for j in adjusting_idx if (j, h) not in D_entries]
    _scenario_feasibility(pins, entries_A, len(D_entries), steps)
    A_flat = [flat(j, h) for (j, h) in entries_A]
    D_flat = [flat(j, h) for (j, h, _) in prescriptions]
    F_flat = sorted(set(range(d_total)) - set(A_flat) - set(D_flat))
    v_S = np.array([v for (_, _, v) in prescriptions])

    # Deterministic path b (mean recursion from the last observed lags).
    intercept = posterior["intercept"].values
    forcing = np.broadcast_to(intercept[:, :, np.newaxis, :], (n_chains, n_draws, steps, n_vars)).copy()
    if exog_future is not None:
        forcing += np.einsum("cdij,hj->cdhi", posterior["B_exog"].values, exog_future)
    A_lags = lag_matrices(B_draws, n_lags)
    b_path = propagate(A_lags, forcing, identified.data.endog[-n_lags:])

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    P_path = _forecast_shock_matrices(identified, steps, rng)

    # One per-step innovation grid in forecast()'s stream order — the ξ
    # source for the free block AND the adjusting-block solve, so the
    # adjusting=all / no-prescriptions case matches conditional_forecast
    # draw-for-draw under a shared seed.
    eps_flat = np.zeros((n_chains, n_draws, d_total))
    xi_flat = None
    if include_shock_uncertainty:
        xi = np.empty((n_chains, n_draws, steps, n_vars))
        # Gaussian by construction: the Waggoner-Zha / ADPRR solves are
        # Gaussian conditioning results. Heavy-tailed error distributions are
        # rejected upstream in IdentifiedVAR.structural_scenario, so this does
        # NOT route through the error_dist seam the way forecast() does.
        for h in range(steps):
            xi[:, :, h, :] = rng.standard_normal((n_chains, n_draws, n_vars))
        xi_flat = xi.reshape(n_chains, n_draws, d_total)
        if F_flat:
            eps_flat[:, :, F_flat] = xi_flat[:, :, F_flat]
    if prescriptions:
        eps_flat[:, :, D_flat] = v_S[np.newaxis, np.newaxis, :]

    q_cond = np.zeros((n_chains, n_draws))
    if r:
        Phi = compute_ma_phi(A_lags, steps - 1)
        C_rows, _, cbar = _constraint_system(pins, b_path, Phi, P_path, n_vars, d_total, warn_on_ill_conditioning=False)
        q_cond = _absorb_conditions(C_rows, cbar, eps_flat, xi_flat, A_flat, D_flat, F_flat, v_S, path_uncertainty)
    elif xi_flat is not None and A_flat:
        # No conditions: adjusting entries are simply free.
        eps_flat[:, :, A_flat] = xi_flat[:, :, A_flat]

    q = q_cond + float(v_S @ v_S) if prescriptions else q_cond

    eps = eps_flat.reshape(n_chains, n_draws, steps, n_vars)
    u = np.einsum("cdhij,cdhj->cdhi", P_path, eps)
    deviation = propagate(A_lags, u, np.zeros((n_lags, n_vars)))
    paths = b_path + deviation

    q_cal = _calibrate_scenario_q(q, r, len(prescriptions), path_uncertainty, d_total)
    return paths, q, q_cond, q_cal, r


def _absorb_conditions(
    C_rows: np.ndarray,
    cbar: np.ndarray,
    eps_flat: np.ndarray,
    xi_flat: np.ndarray | None,
    A_flat: list[int],
    D_flat: list[int],
    F_flat: list[int],
    v_S: np.ndarray,
    path_uncertainty: str,
) -> np.ndarray:
    """Solve the adjusting block and write it into `eps_flat` (in place).

    Reduces the full constraint rows to the adjusting columns, folds the
    prescribed and free contributions into the rhs, checks per-draw
    numerical rank, and fills the adjusting entries per the conditioning
    mode. Returns the per-draw condition part of the plausibility
    statistic, `ctilde'(C_A C_A')^{-1} ctilde`.
    """
    r = cbar.shape[-1]
    C_A = C_rows[:, :, :, A_flat]
    ctilde = cbar.copy()
    if D_flat:
        ctilde -= np.einsum("cdrk,k->cdr", C_rows[:, :, :, D_flat], v_S)
    if F_flat:
        ctilde -= np.einsum("cdrk,cdk->cdr", C_rows[:, :, :, F_flat], eps_flat[:, :, F_flat])

    ranks = np.linalg.matrix_rank(C_A)
    if ranks.min() < r:
        raise ValueError(
            f"The adjusting-block constraint matrix is rank-deficient for "
            f"{int((ranks < r).sum())} posterior draw(s) (rank < {r}): the adjusting "
            "shocks cannot absorb the conditions there (e.g. a Cholesky zero makes a "
            "condition load on no adjusting shock at its step). Widen the adjusting set."
        )
    G_A = np.einsum("cdrk,cdsk->cdrs", C_A, C_A)
    if np.max(np.linalg.cond(G_A)) > 1e8:
        # stacklevel targets the public caller of structural_scenario
        # (user -> structural_scenario -> engine -> here -> warn).
        warnings.warn(
            "The adjusting-block constraint system is nearly redundant (condition "
            "number of C_A C_A' exceeds 1e8) — nearly-collinear condition loadings "
            "on the adjusting shocks inflate the conditional adjustment and the "
            "plausibility statistic without tripping the rank check.",
            UserWarning,
            stacklevel=4,
        )
    alpha = np.linalg.solve(G_A, ctilde[..., np.newaxis])[..., 0]
    E_A_mean = np.einsum("cdrk,cdr->cdk", C_A, alpha)

    if xi_flat is not None and path_uncertainty == "none":
        xi_A = xi_flat[:, :, A_flat]
        residual = np.einsum("cdrk,cdk->cdr", C_A, xi_A) - ctilde
        beta = np.linalg.solve(G_A, residual[..., np.newaxis])[..., 0]
        eps_flat[:, :, A_flat] = xi_A - np.einsum("cdrk,cdr->cdk", C_A, beta)
    elif xi_flat is not None:
        eps_flat[:, :, A_flat] = E_A_mean + xi_flat[:, :, A_flat]
    else:
        eps_flat[:, :, A_flat] = E_A_mean
    return np.einsum("cdr,cdr->cd", ctilde, alpha)


def _calibrate_scenario_q(
    q: np.ndarray,
    n_conditions: int,
    n_prescribed: int,
    path_uncertainty: str,
    d_total: int,
) -> np.ndarray:
    """Scenario variant of the ADPRR calibration.

    Finite only under `path_uncertainty="unconditional"` with no
    prescriptions (hard substitutions keep the divergence infinite); the
    ceiling is 1, the no-restriction floor 0.5.
    """
    if n_conditions + n_prescribed == 0:
        return np.full_like(q, 0.5)
    if path_uncertainty == "unconditional" and n_prescribed == 0:
        return (1.0 + np.sqrt(1.0 - np.exp(-q / d_total))) / 2.0
    return np.ones_like(q)
