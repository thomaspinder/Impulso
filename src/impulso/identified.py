"""IdentifiedVAR — structural VAR with identified shocks."""

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field

from impulso._base import ImpulsoBaseModel
from impulso.data import VARData
from impulso.results import FEVDResult, HistoricalDecompositionResult, IRFResult


class IdentifiedVAR(ImpulsoBaseModel):
    """Immutable structural VAR with identified shocks.

    Attributes:
        idata: InferenceData with structural_shock_matrix in posterior.
        n_lags: Lag order.
        data: Original VARData.
        var_names: Endogenous variable names.
    """

    idata: az.InferenceData = Field(repr=False)
    n_lags: int
    data: VARData
    var_names: list[str]

    def _ma_coefficients(self, B_draws: np.ndarray, n_vars: int, n_lags: int, horizon: int) -> np.ndarray:
        """Compute MA coefficient recursion, vectorised over (chains, draws).

        Returns:
            Array of shape (C, D, horizon+1, n_vars, n_vars).
        """
        # Extract A_j matrices: each (C, D, n, n)
        A = [B_draws[:, :, :, j * n_vars : (j + 1) * n_vars] for j in range(n_lags)]

        n_chains, n_draws = B_draws.shape[:2]
        Phi = [np.broadcast_to(np.eye(n_vars), (n_chains, n_draws, n_vars, n_vars)).copy()]
        for h in range(1, horizon + 1):
            phi_h = np.zeros((n_chains, n_draws, n_vars, n_vars))
            for j in range(min(h, n_lags)):
                phi_h += np.einsum("cdij,cdjk->cdik", A[j], Phi[h - j - 1])
            Phi.append(phi_h)

        return np.stack(Phi, axis=2)

    def impulse_response(self, horizon: int = 20) -> IRFResult:
        """Compute structural impulse response functions.

        Args:
            horizon: Number of periods.

        Returns:
            IRFResult with IRF posterior draws.
        """
        B_draws = self.idata.posterior["B"].values  # (C, D, n, n*p)
        P_draws = self.idata.posterior["structural_shock_matrix"].values  # (C, D, n, n)
        n_vars = B_draws.shape[2]

        Phi_arr = self._ma_coefficients(B_draws, n_vars, self.n_lags, horizon)
        # IRF = Phi @ P, vectorised over (C, D, H) via broadcasting
        irfs = Phi_arr @ P_draws[:, :, np.newaxis, :, :]

        irf_da = xr.DataArray(
            irfs,
            dims=["chain", "draw", "horizon", "response", "shock"],
            coords={
                "response": self.var_names,
                "shock": self.var_names,
                "horizon": np.arange(horizon + 1),
            },
            name="irf",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"irf": irf_da}))
        return IRFResult(idata=idata, horizon=horizon, var_names=self.var_names)

    def forecast_error_variance_decomposition(self, horizon: int = 20) -> FEVDResult:
        """Compute forecast error variance decomposition.

        Args:
            horizon: Number of periods.

        Returns:
            FEVDResult with FEVD posterior draws.
        """
        B_draws = self.idata.posterior["B"].values
        P_draws = self.idata.posterior["structural_shock_matrix"].values
        n_vars = B_draws.shape[2]

        Phi_arr = self._ma_coefficients(B_draws, n_vars, self.n_lags, horizon)
        # Theta_h = Phi_h @ P, vectorised via broadcasting
        Theta = Phi_arr @ P_draws[:, :, np.newaxis, :, :]

        # FEVD: cumulative MSE contribution
        mse_cum = np.cumsum(Theta**2, axis=2)  # (C, D, H+1, resp, shock)
        total = mse_cum.sum(axis=-1, keepdims=True)  # (C, D, H+1, resp, 1)
        fevd = np.where(total > 0, mse_cum / total, 0.0)

        fevd_da = xr.DataArray(
            fevd,
            dims=["chain", "draw", "horizon", "response", "shock"],
            coords={"response": self.var_names, "shock": self.var_names},
            name="fevd",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"fevd": fevd_da}))
        return FEVDResult(idata=idata, horizon=horizon, var_names=self.var_names)

    def fevd(self, horizon: int = 20) -> FEVDResult:
        """Alias for forecast_error_variance_decomposition.

        Args:
            horizon: Number of periods.

        Returns:
            FEVDResult.
        """
        return self.forecast_error_variance_decomposition(horizon=horizon)

    def historical_decomposition(
        self,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        cumulative: bool = False,
    ) -> HistoricalDecompositionResult:
        """Compute historical decomposition of observed series.

        Args:
            start: Optional start date to restrict decomposition.
            end: Optional end date to restrict decomposition.
            cumulative: If True, return cumulative shock contributions.

        Returns:
            HistoricalDecompositionResult.
        """
        B_draws = self.idata.posterior["B"].values  # (C, D, n, n*p)
        P_draws = self.idata.posterior["structural_shock_matrix"].values  # (C, D, n, n)
        intercept_draws = self.idata.posterior["intercept"].values  # (C, D, n)

        y = self.data.endog  # (T, n)
        T = y.shape[0]
        n_lags = self.n_lags

        # Build lag matrix: x_lag[t] = [y[t-1], y[t-2], ...] for t in [n_lags, T)
        x_lag = np.concatenate([y[n_lags - lag : T - lag] for lag in range(1, n_lags + 1)], axis=1)  # (T-p, n*p)

        # Predicted values: intercept + B @ x_lag for all (C, D, t)
        y_hat = intercept_draws[:, :, np.newaxis, :] + np.einsum("cdij,tj->cdti", B_draws, x_lag)

        # Reduced-form residuals: (C, D, T-p, n)
        y_obs = y[n_lags:]  # (T-p, n)
        resid = y_obs[np.newaxis, np.newaxis, :, :] - y_hat

        # Structural residuals: P_inv @ resid
        P_inv = np.linalg.inv(P_draws)  # (C, D, n, n)
        structural_resid = np.einsum("cdij,cdtj->cdti", P_inv, resid)  # (C, D, T-p, n)

        # hd[..., resp, shock] = P[resp, shock] * structural_resid[shock]
        hd = P_draws[:, :, np.newaxis, :, :] * structural_resid[:, :, :, np.newaxis, :]

        if cumulative:
            hd = np.cumsum(hd, axis=2)

        # Trim to date range if requested
        idx = self.data.index[n_lags:]
        t_start = 0
        t_end = len(idx)
        if start is not None:
            t_start = idx.searchsorted(start)
        if end is not None:
            t_end = idx.searchsorted(end, side="right")
        hd = hd[:, :, t_start:t_end]

        hd_da = xr.DataArray(
            hd,
            dims=["chain", "draw", "time", "response", "shock"],
            coords={"response": self.var_names, "shock": self.var_names},
            name="hd",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"hd": hd_da}))
        return HistoricalDecompositionResult(idata=idata, var_names=self.var_names)
