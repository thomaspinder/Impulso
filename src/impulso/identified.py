"""IdentifiedVAR — structural VAR with identified shocks."""

from typing import TYPE_CHECKING

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field

from impulso._base import ImpulsoBaseModel
from impulso.data import VARData
from impulso.results import FEVDResult, HistoricalDecompositionResult, IRFResult

if TYPE_CHECKING:
    from impulso.results import ConditionalForecastResult


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

    def conditional_forecast(
        self,
        steps: int,
        conditions: list,
        shock_conditions: list | None = None,
        exog_future: np.ndarray | None = None,
    ) -> "ConditionalForecastResult":
        """Produce structural conditional forecasts.

        Extends reduced-form conditional forecasting by allowing constraints
        on structural shock paths in addition to observable variable paths.

        Args:
            steps: Number of forecast steps.
            conditions: List of ForecastCondition instances for observables.
            shock_conditions: Optional list of ForecastCondition instances for
                structural shocks.
            exog_future: Future exogenous values if model has exog.

        Returns:
            ConditionalForecastResult with constrained forecast draws.
        """
        from impulso.fitted import FittedVAR

        # If no shock conditions, delegate to the reduced-form method
        if shock_conditions is None:
            fitted = FittedVAR.model_construct(
                idata=self.idata,
                n_lags=self.n_lags,
                data=self.data,
                var_names=self.var_names,
            )
            return fitted.conditional_forecast(steps=steps, conditions=conditions, exog_future=exog_future)

        return self._structural_conditional_forecast(steps, conditions, shock_conditions)

    def _validate_structural_conditions(self, conditions: list, shock_conditions: list, steps: int) -> list[str]:
        """Validate observable and shock conditions, returning shock names.

        Args:
            conditions: Observable variable conditions.
            shock_conditions: Structural shock conditions.
            steps: Number of forecast steps.

        Returns:
            List of shock variable names from the structural matrix.
        """
        for cond in conditions:
            if cond.variable not in self.var_names:
                raise ValueError(f"Condition variable '{cond.variable}' not in var_names")
            for p in cond.periods:
                if p < 0 or p >= steps:
                    raise ValueError(f"Condition period {p} out of range for {steps} steps")

        shock_names = self.idata.posterior["structural_shock_matrix"].coords["shock"].values.tolist()
        for cond in shock_conditions:
            if cond.variable not in shock_names:
                raise ValueError(f"Shock condition variable '{cond.variable}' not in shock_names {shock_names}")
        return shock_names

    @staticmethod
    def _build_structural_constraint_system(
        conditions: list,
        shock_conditions: list,
        var_names: list[str],
        shock_names: list[str],
        ma_coefficients: list[np.ndarray],
        P: np.ndarray,
        unconditional: np.ndarray,
        steps: int,
        n_vars: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the combined observable + shock constraint system.

        Returns:
            Tuple of (R, target) arrays for the constraint system.
        """
        constraint_rows = []
        constraint_targets = []

        # Observable constraints
        for cond in conditions:
            var_idx = var_names.index(cond.variable)
            for period, value in zip(cond.periods, cond.values, strict=True):
                row = np.zeros(steps * n_vars)
                for s in range(period + 1):
                    structural_response = ma_coefficients[period - s] @ P
                    row[s * n_vars : (s + 1) * n_vars] = structural_response[var_idx, :]
                constraint_rows.append(row)
                constraint_targets.append(value - unconditional[period, var_idx])

        # Shock constraints
        for cond in shock_conditions:
            shock_idx = shock_names.index(cond.variable)
            for period, value in zip(cond.periods, cond.values, strict=True):
                row = np.zeros(steps * n_vars)
                row[period * n_vars + shock_idx] = 1.0
                constraint_rows.append(row)
                constraint_targets.append(value)

        return np.array(constraint_rows), np.array(constraint_targets)

    def _structural_conditional_forecast(
        self, steps: int, conditions: list, shock_conditions: list
    ) -> "ConditionalForecastResult":
        """Compute structural conditional forecast with shock constraints."""
        from impulso.fitted import FittedVAR
        from impulso.results import ConditionalForecastResult

        shock_names = self._validate_structural_conditions(conditions, shock_conditions, steps)

        B_draws = self.idata.posterior["B"].values
        intercept_draws = self.idata.posterior["intercept"].values
        P_draws = self.idata.posterior["structural_shock_matrix"].values
        n_chains, n_draws, n_vars, _ = B_draws.shape

        y_hist = self.data.endog[-self.n_lags :]
        forecasts = np.zeros((n_chains, n_draws, steps, n_vars))

        for c in range(n_chains):
            for d in range(n_draws):
                B = B_draws[c, d]
                P = P_draws[c, d]

                ma_coefficients = FittedVAR._ma_coefficients_single(B, n_vars, self.n_lags, steps)
                unconditional = self._unconditional_forecast_single(
                    B, intercept_draws[c, d], y_hist, steps, self.n_lags, n_vars
                )

                R, target = self._build_structural_constraint_system(
                    conditions,
                    shock_conditions,
                    self.var_names,
                    shock_names,
                    ma_coefficients,
                    P,
                    unconditional,
                    steps,
                    n_vars,
                )

                structural_shocks, _, _, _ = np.linalg.lstsq(R, target, rcond=None)
                structural_shocks = structural_shocks.reshape(steps, n_vars)

                conditional = unconditional.copy()
                for h in range(steps):
                    for s in range(h + 1):
                        conditional[h] += ma_coefficients[h - s] @ P @ structural_shocks[s]

                forecasts[c, d] = conditional

        forecast_da = xr.DataArray(
            forecasts,
            dims=["chain", "draw", "step", "variable"],
            coords={"variable": self.var_names},
            name="forecast",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"forecast": forecast_da}))
        all_conditions = conditions + (shock_conditions or [])
        return ConditionalForecastResult(idata=idata, steps=steps, var_names=self.var_names, conditions=all_conditions)

    @staticmethod
    def _unconditional_forecast_single(
        B: np.ndarray,
        intercept: np.ndarray,
        y_hist: np.ndarray,
        steps: int,
        n_lags: int,
        n_vars: int,
    ) -> np.ndarray:
        """Compute unconditional forecast for a single posterior draw.

        Returns:
            Array of shape (steps, n_vars).
        """
        y_buffer = y_hist.copy()
        unconditional = np.zeros((steps, n_vars))
        for h in range(steps):
            x_lag = np.concatenate([y_buffer[-(lag + 1)] for lag in range(n_lags)])
            unconditional[h] = intercept + B @ x_lag
            y_buffer = np.vstack([y_buffer[1:], unconditional[h].reshape(1, -1)])
        return unconditional
