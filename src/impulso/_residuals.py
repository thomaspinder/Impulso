"""Reduced-form fitted values and residuals from posterior draws.

Shared by `IdentifiedVAR.historical_decomposition`, identification
schemes that need residuals (e.g. `ProxySVAR`), and
`FittedVAR.posterior_predictive` (which needs the conditional mean
only). Reconstruction from `B`/`intercept` draws is cheap and exact, so
neither fitted values nor residuals are stored in the posterior.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from impulso.data import VARData


def fitted_values(posterior: "xr.Dataset", data: "VARData", n_lags: int) -> np.ndarray:
    """Reconstruct per-draw one-step-ahead conditional means on the estimation sample.

    Computes `E[y_t | y_{t-1}, ..., y_{t-p}, theta] = intercept + B @ x_lag_t`
    for every posterior draw, where `x_lag_t` stacks lags 1..n_lags (lag-1
    block first) of the **observed** data. For models with exogenous
    variables, the `B_exog @ x_exog_t` contribution is added.

    Because the lags are the observed ones (never simulated), this is the
    one-step-ahead in-sample fit, not an iterated simulated path — the
    latter is `FittedVAR.forecast`'s job.

    Args:
        posterior: Posterior Dataset holding `B` (chains, draws, n, n*p),
            `intercept` (chains, draws, n) and, when the model has
            exogenous variables, `B_exog`.
        data: The VARData used at fit time.
        n_lags: Lag order of the fitted VAR.

    Returns:
        Conditional-mean array of shape `(chains, draws, T - n_lags, n_vars)`,
        time-aligned with `data.index[n_lags:]`.
    """
    B_draws = posterior["B"].values  # (C, D, n, n*p)
    intercept_draws = posterior["intercept"].values  # (C, D, n)

    y = data.endog  # (T, n)
    T = y.shape[0]

    x_lag = np.concatenate(
        [y[n_lags - lag : T - lag] for lag in range(1, n_lags + 1)],
        axis=1,
    )  # (T-p, n*p)
    y_hat = intercept_draws[:, :, np.newaxis, :] + np.einsum("cdij,tj->cdti", B_draws, x_lag)
    if data.exog is not None and "B_exog" in posterior:
        y_hat = y_hat + np.einsum("cdij,tj->cdti", posterior["B_exog"].values, data.exog[n_lags:])
    return y_hat


def reduced_form_residuals(posterior: "xr.Dataset", data: "VARData", n_lags: int) -> np.ndarray:
    """Reconstruct per-draw reduced-form residuals on the estimation sample.

    Computes `u_t = y_t - intercept - B @ x_lag_t` for every posterior
    draw, where `x_lag_t` stacks lags 1..n_lags (lag-1 block first). For
    models with exogenous variables, the `B_exog @ x_exog_t` contribution
    is subtracted as well.

    Args:
        posterior: Posterior Dataset holding `B` (chains, draws, n, n*p),
            `intercept` (chains, draws, n) and, when the model has
            exogenous variables, `B_exog`.
        data: The VARData used at fit time.
        n_lags: Lag order of the fitted VAR.

    Returns:
        Residual array of shape `(chains, draws, T - n_lags, n_vars)`,
        time-aligned with `data.index[n_lags:]`.
    """
    y_hat = fitted_values(posterior, data, n_lags)
    y_obs = data.endog[n_lags:]
    return y_obs[np.newaxis, np.newaxis, :, :] - y_hat
