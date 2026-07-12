"""Frequentist proxy-SVAR benchmark, replicating Känzig (2021).

A NumPy re-implementation of the point estimate and moving-block-bootstrap
confidence bands from ``runProxyVAR.m`` in the author's replication
repository (dkaenzig/replicationOilSupplyNews). This gives the notebook a
"published bands" overlay computed from the same data by the same method,
without needing MATLAB.

Conventions follow the paper: VAR(12) in levels with a constant, sample
1974M01-2017M12; instrument sample 1975M01-2017M12 (missing months censored
to zero); a shock normalised to raise the real oil price by 10% on impact;
68% and 90% moving-block-bootstrap bands recentred on the point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def var_ols(data: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS VAR(p) with a constant, as in varxest.m.

    Args:
        data: (T_all, n) array of levels.
        p: Lag order.

    Returns:
        B: (n, 1 + n*p) coefficient matrix, constant first, then lag 1..p
           blocks (each block ordered as the variables).
        U: (T_all - p, n) residuals.
    """
    T_all, n = data.shape
    Y = data[p:]
    # Regressors: constant, then y_{t-1}, ..., y_{t-p}.
    X = np.ones((T_all - p, 1 + n * p))
    for lag in range(1, p + 1):
        X[:, 1 + (lag - 1) * n : 1 + lag * n] = data[p - lag : T_all - lag]
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    U = Y - X @ B
    return B.T, U


def first_stage_f(proxy: np.ndarray, u1: np.ndarray, robust: bool = False) -> float:
    """F-statistic of the first-stage regression u1 ~ const + proxy.

    With ``robust=True``, returns the White heteroskedasticity-robust Wald
    F, matching ``olsest.m``'s ``Frobust``.
    """
    T = len(proxy)
    X = np.column_stack([np.ones(T), proxy])
    beta, *_ = np.linalg.lstsq(X, u1, rcond=None)
    resid = u1 - X @ beta
    if robust:
        # olsest.m: Frobust = b' inv(n/(n-k) R invXpX Shat invXpX R') b
        # with R selecting the slope; k = 2 regressors.
        inv_xtx = np.linalg.inv(X.T @ X)
        shat = (X * resid[:, None] ** 2).T @ X
        var_robust = T / (T - 2) * inv_xtx @ shat @ inv_xtx
        return float(beta[1] ** 2 / var_robust[1, 1])
    rss = resid @ resid
    tss = ((u1 - u1.mean()) ** 2).sum()
    k = 1  # tested restrictions (the proxy slope)
    return ((tss - rss) / k) / (rss / (T - 2))


def impact_ratios(proxy: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Relative impact column via 2SLS, as in runProxyVAR.m.

    First stage: U[:, 0] on [const, proxy] -> fitted values uhat.
    Second stage: U[:, 1:] on [const, uhat] -> slopes b_{j1}/b_{11}.

    Returns:
        (n,) vector [1, b21/b11, ..., bn1/b11].
    """
    T = U.shape[0]
    X1 = np.column_stack([np.ones(T), proxy])
    beta1, *_ = np.linalg.lstsq(X1, U[:, 0], rcond=None)
    uhat = X1 @ beta1
    X2 = np.column_stack([np.ones(T), uhat])
    beta2, *_ = np.linalg.lstsq(X2, U[:, 1:], rcond=None)
    return np.concatenate([[1.0], beta2[1]])


def irf_ma(B_lags: np.ndarray, b1: np.ndarray, p: int, horizon: int) -> np.ndarray:
    """IRF to impact vector b1 via the MA recursion, as in varirfsingle.m.

    Args:
        B_lags: (n, n*p) lag coefficients (constant excluded), lag-1 block first.
        b1: (n,) structural impact column.
        p: Lag order.
        horizon: Maximum horizon (inclusive).

    Returns:
        (horizon + 1, n) IRF matrix, row h = response at horizon h.
    """
    n = B_lags.shape[0]
    A = [B_lags[:, lag * n : (lag + 1) * n] for lag in range(p)]
    irf = np.zeros((horizon + 1, n))
    irf[0] = b1
    for h in range(1, horizon + 1):
        acc = np.zeros(n)
        for lag in range(1, min(h, p) + 1):
            acc += A[lag - 1] @ irf[h - lag]
        irf[h] = acc
    return irf


@dataclass
class ProxyVARResult:
    """Point estimate and bootstrap bands for the proxy-SVAR IRFs."""

    irf: np.ndarray  # (H+1, n) point-estimate IRF, normalised
    bands68: tuple[np.ndarray, np.ndarray]  # lower, upper (H+1, n)
    bands90: tuple[np.ndarray, np.ndarray]
    first_stage_f: float
    boot_f: np.ndarray  # (nsim,) bootstrap first-stage F stats


def proxy_var_kaenzig(
    data: np.ndarray,
    proxy: np.ndarray,
    p: int = 12,
    horizon: int = 50,
    shock_size: float = 10.0,
    nsim: int = 1000,
    seed: int = 0,
) -> ProxyVARResult:
    """Full Känzig (2021) frequentist pipeline: point estimate + MBB bands.

    Args:
        data: (T_all, n) levels, estimation sample (1974M01-2017M12).
        proxy: (T_all - p,) instrument aligned with the VAR residuals
            (1975M01-2017M12), missing months censored to zero.
        p: Lag order.
        horizon: IRF horizon.
        shock_size: Impact normalisation for variable 0 (10 = +10%).
        nsim: Bootstrap replications.
        seed: RNG seed.

    Returns:
        ProxyVARResult with normalised IRFs and recentred quantile bands.
    """
    rng = np.random.default_rng(seed)
    T_all, n = data.shape
    B, U = var_ols(data, p)
    T_est = U.shape[0]
    if len(proxy) != T_est:
        raise ValueError(f"proxy length {len(proxy)} != residual sample {T_est}")

    ratios = impact_ratios(proxy, U)
    irf_pe = irf_ma(B[:, 1:], ratios, p, horizon)
    irf_pe = irf_pe / irf_pe[0, 0] * shock_size
    f_stat = first_stage_f(proxy, U[:, 0])

    # --- Moving block bootstrap (mbb1block, Jentsch & Lunsford) ------------
    block = round(5.03 * T_est**0.25)
    n_block = int(np.ceil(T_est / block))
    n_starts = T_est - block + 1

    # Centering terms: position-within-block means (tiled to T_est).
    idx_mat = np.arange(block)[:, None] + np.arange(n_starts)[None, :]  # (block, n_starts)
    u_center = U[idx_mat].mean(axis=1)  # (block, n) mean over starts
    u_center = np.tile(u_center, (n_block, 1))[:T_est]
    nz_all = proxy != 0
    proxy_mean_nz = proxy[nz_all].mean()
    proxy_center = np.zeros(block)
    for j in range(block):
        sub = proxy[j : n_starts + j]
        proxy_center[j] = sub[sub != 0].mean() - proxy_mean_nz
    proxy_center = np.tile(proxy_center, n_block)[:T_est]

    boot_irfs = np.full((nsim, horizon + 1, n), np.nan)
    boot_f = np.full(nsim, np.nan)
    isim = 0
    while isim < nsim:
        starts = rng.integers(0, n_starts, size=n_block)
        boot_u = np.concatenate([U[s : s + block] for s in starts])[:T_est]
        boot_proxy = np.concatenate([proxy[s : s + block] for s in starts])[:T_est]

        boot_u = boot_u - u_center
        nz = boot_proxy != 0
        if nz.sum() < 15:
            continue
        boot_proxy = boot_proxy.copy()
        boot_proxy[nz] = boot_proxy[nz] - proxy_center[nz]

        # Simulate data from the estimated VAR with bootstrapped residuals.
        boot_data = np.empty((T_all, n))
        boot_data[:p] = data[:p]
        const = B[:, 0]
        B_lags = B[:, 1:]
        for t in range(p, T_all):
            x_lag = boot_data[t - p : t][::-1].ravel()  # lag 1 first
            boot_data[t] = const + B_lags @ x_lag + boot_u[t - p]

        boot_B, boot_U = var_ols(boot_data, p)
        boot_ratios = impact_ratios(boot_proxy, boot_U)
        birf = irf_ma(boot_B[:, 1:], boot_ratios, p, horizon)
        boot_irfs[isim] = birf / birf[0, 0] * shock_size
        boot_f[isim] = first_stage_f(boot_proxy, boot_U[:, 0])
        isim += 1

    # Recentre quantiles on the point estimate (Hall-type), as in the repo.
    med = np.nanquantile(boot_irfs, 0.5, axis=0)

    def bands(alpha: float) -> tuple[np.ndarray, np.ndarray]:
        lo = np.nanquantile(boot_irfs, alpha / 2, axis=0) - med + irf_pe
        hi = np.nanquantile(boot_irfs, 1 - alpha / 2, axis=0) - med + irf_pe
        return lo, hi

    return ProxyVARResult(
        irf=irf_pe,
        bands68=bands(0.32),
        bands90=bands(0.10),
        first_stage_f=f_stat,
        boot_f=boot_f,
    )
