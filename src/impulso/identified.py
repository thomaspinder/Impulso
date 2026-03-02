"""IdentifiedVAR — structural VAR with identified shocks."""

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict

from impulso.data import VARData
from impulso.results import FEVDResult, HistoricalDecompositionResult, IRFResult


class IdentifiedVAR(BaseModel):
    """Immutable structural VAR with identified shocks.

    Attributes:
        idata: InferenceData with structural_shock_matrix in posterior.
        n_lags: Lag order.
        data: Original VARData.
        var_names: Endogenous variable names.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    idata: az.InferenceData
    n_lags: int
    data: VARData
    var_names: list[str]

    def impulse_response(self, horizon: int = 20) -> IRFResult:
        """Compute structural impulse response functions.

        Args:
            horizon: Number of periods.

        Returns:
            IRFResult with IRF posterior draws.
        """
        B_draws = self.idata.posterior["B"].values  # (chains, draws, n, n*p)
        P_draws = self.idata.posterior["structural_shock_matrix"].values  # (chains, draws, n, n)
        n_chains, n_draws, n_vars, _ = B_draws.shape
        n_lags = self.n_lags

        irfs = np.zeros((n_chains, n_draws, horizon + 1, n_vars, n_vars))
        for c in range(n_chains):
            for d in range(n_draws):
                B = B_draws[c, d]  # (n, n*p)
                P = P_draws[c, d]  # (n, n)

                # MA coefficients via recursion
                Phi = [np.eye(n_vars)]  # Phi_0 = I
                for h in range(1, horizon + 1):
                    phi_h = np.zeros((n_vars, n_vars))
                    for j in range(1, min(h, n_lags) + 1):
                        A_j = B[:, (j - 1) * n_vars : j * n_vars]
                        phi_h += A_j @ Phi[h - j]
                    Phi.append(phi_h)

                for h in range(horizon + 1):
                    irfs[c, d, h] = Phi[h] @ P

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
        n_chains, n_draws, n_vars, _ = B_draws.shape
        n_lags = self.n_lags

        fevd = np.zeros((n_chains, n_draws, horizon + 1, n_vars, n_vars))
        for c in range(n_chains):
            for d in range(n_draws):
                B = B_draws[c, d]
                P = P_draws[c, d]

                Phi = [np.eye(n_vars)]
                for h in range(1, horizon + 1):
                    phi_h = np.zeros((n_vars, n_vars))
                    for j in range(1, min(h, n_lags) + 1):
                        A_j = B[:, (j - 1) * n_vars : j * n_vars]
                        phi_h += A_j @ Phi[h - j]
                    Phi.append(phi_h)

                # Accumulate MSE contributions
                mse_total = np.zeros((n_vars, n_vars))
                for h in range(horizon + 1):
                    theta_h = Phi[h] @ P
                    mse_total += theta_h**2
                    for resp in range(n_vars):
                        total = mse_total[resp].sum()
                        if total > 0:
                            fevd[c, d, h, resp] = mse_total[resp] / total

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
        B_draws = self.idata.posterior["B"].values
        P_draws = self.idata.posterior["structural_shock_matrix"].values
        intercept_draws = self.idata.posterior["intercept"].values
        n_chains, n_draws, n_vars, _ = B_draws.shape

        y = self.data.endog
        T = y.shape[0]
        n_lags = self.n_lags

        hd = np.zeros((n_chains, n_draws, T - n_lags, n_vars, n_vars))

        for c in range(n_chains):
            for d in range(n_draws):
                B = B_draws[c, d]
                P = P_draws[c, d]
                intercept = intercept_draws[c, d]
                P_inv = np.linalg.inv(P)

                # Compute structural residuals
                for t in range(n_lags, T):
                    x_lag = np.concatenate([y[t - lag] for lag in range(1, n_lags + 1)])
                    resid = y[t] - intercept - B @ x_lag
                    structural_resid = P_inv @ resid
                    # Each shock's contribution
                    for k in range(n_vars):
                        hd[c, d, t - n_lags, :, k] = P[:, k] * structural_resid[k]

                if cumulative:
                    hd[c, d] = np.cumsum(hd[c, d], axis=0)

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

    def __repr__(self) -> str:
        n_vars = len(self.var_names)
        posterior = self.idata.posterior
        n_chains = posterior.sizes["chain"]
        n_draws = posterior.sizes["draw"]
        return f"IdentifiedVAR(n_lags={self.n_lags}, n_vars={n_vars}, n_draws={n_draws}, n_chains={n_chains})"
