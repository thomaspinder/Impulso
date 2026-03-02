"""OLS-based lag order selection using information criteria."""

import numpy as np
import pandas as pd

from impulso.data import VARData
from impulso.results import LagOrderResult


def select_lag_order(data: VARData, max_lags: int = 12) -> LagOrderResult:
    """Select optimal VAR lag order using AIC, BIC, and Hannan-Quinn.

    Uses OLS estimation (fast) to compute information criteria for each
    candidate lag order from 1 to max_lags.

    Args:
        data: VARData instance.
        max_lags: Maximum number of lags to evaluate.

    Returns:
        LagOrderResult with optimal lag orders and full criteria table.
    """
    y = data.endog
    T, n = y.shape

    results = []
    for p in range(1, max_lags + 1):
        # Build lagged regressor matrix
        Y = y[p:]  # (T-p, n)
        T_eff = Y.shape[0]
        X_parts = [np.ones((T_eff, 1))]  # intercept
        for lag in range(1, p + 1):
            X_parts.append(y[p - lag : T - lag])
        if data.exog is not None:
            X_parts.append(data.exog[p:])
        X = np.hstack(X_parts)  # (T_eff, 1 + n*p + k)

        # OLS
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ beta
        sigma = (resid.T @ resid) / T_eff

        # Log determinant of residual covariance
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            logdet = np.inf

        k_params = X.shape[1] * n  # total parameters
        aic = logdet + 2 * k_params / T_eff
        bic = logdet + np.log(T_eff) * k_params / T_eff
        hq = logdet + 2 * np.log(np.log(T_eff)) * k_params / T_eff

        results.append({"lag": p, "aic": aic, "bic": bic, "hq": hq})

    table = pd.DataFrame(results).set_index("lag")

    return LagOrderResult(
        aic=int(table["aic"].idxmin()),
        bic=int(table["bic"].idxmin()),
        hq=int(table["hq"].idxmin()),
        criteria_table=table,
    )
