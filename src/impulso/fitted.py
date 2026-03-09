"""FittedVAR — reduced-form posterior from Bayesian VAR estimation."""

from typing import TYPE_CHECKING

import arviz as az
import numpy as np
from pydantic import Field, computed_field

from impulso._base import ImpulsoBaseModel
from impulso.data import VARData
from impulso.protocols import IdentificationScheme

if TYPE_CHECKING:
    from impulso.conditions import ForecastCondition
    from impulso.identified import IdentifiedVAR
    from impulso.results import ConditionalForecastResult, ForecastResult


class FittedVAR(ImpulsoBaseModel):
    """Immutable container for a fitted (reduced-form) Bayesian VAR.

    Attributes:
        idata: ArviZ InferenceData with posterior draws.
        n_lags: Lag order used in estimation.
        data: Original VARData used for fitting.
        var_names: Names of endogenous variables.
    """

    idata: az.InferenceData = Field(repr=False)
    n_lags: int
    data: VARData
    var_names: list[str]

    @computed_field
    @property
    def has_exog(self) -> bool:
        """Whether the model includes exogenous variables."""
        return self.data.exog is not None

    @property
    def coefficients(self) -> np.ndarray:
        """Posterior draws of B coefficient matrices."""
        return self.idata.posterior["B"].values

    @property
    def intercepts(self) -> np.ndarray:
        """Posterior draws of intercept vectors."""
        return self.idata.posterior["intercept"].values

    @property
    def sigma(self) -> np.ndarray:
        """Posterior draws of residual covariance matrix."""
        return self.idata.posterior["Sigma"].values

    def forecast(
        self,
        steps: int,
        exog_future: np.ndarray | None = None,
    ) -> "ForecastResult":
        """Produce h-step-ahead forecasts from the reduced-form posterior.

        Args:
            steps: Number of forecast steps.
            exog_future: Future exogenous values, shape (steps, k). Required if model has exog.

        Returns:
            ForecastResult with posterior forecast draws.
        """
        import xarray as xr

        from impulso.results import ForecastResult

        if self.has_exog and exog_future is None:
            raise ValueError("exog_future is required when model includes exogenous variables")
        if not self.has_exog and exog_future is not None:
            raise ValueError("exog_future provided but model has no exogenous variables")

        B_draws = self.coefficients  # (C, D, n_vars, n_vars*n_lags)
        intercept_draws = self.intercepts  # (C, D, n_vars)
        n_chains, n_draws, n_vars, _ = B_draws.shape

        # Last n_lags observations — broadcast to (C, D, n_lags, n)
        y_hist = self.data.endog[-self.n_lags :]  # (p, n)
        y_buffer = np.broadcast_to(y_hist, (n_chains, n_draws, self.n_lags, n_vars)).copy()

        forecasts = np.zeros((n_chains, n_draws, steps, n_vars))

        for h in range(steps):
            # Build lag vector: concatenate y[t-1], y[t-2], ..., y[t-p]
            x_lag = np.concatenate(
                [y_buffer[:, :, -(lag + 1), :] for lag in range(self.n_lags)], axis=-1
            )  # (C, D, n*p)

            y_new = intercept_draws + np.einsum("cdij,cdj->cdi", B_draws, x_lag)

            if self.has_exog and exog_future is not None:
                B_exog = self.idata.posterior["B_exog"].values  # (C, D, n, k)
                y_new = y_new + np.einsum("cdij,j->cdi", B_exog, exog_future[h])

            forecasts[:, :, h, :] = y_new
            # Roll buffer forward
            y_buffer = np.concatenate([y_buffer[:, :, 1:, :], y_new[:, :, np.newaxis, :]], axis=2)

        # Package into InferenceData
        forecast_da = xr.DataArray(
            forecasts,
            dims=["chain", "draw", "step", "variable"],
            coords={"variable": self.var_names},
            name="forecast",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"forecast": forecast_da}))

        return ForecastResult(idata=idata, steps=steps, var_names=self.var_names)

    @staticmethod
    def _ma_coefficients_single(B: np.ndarray, n_vars: int, n_lags: int, steps: int) -> list[np.ndarray]:
        """Compute MA coefficient recursion for a single draw.

        Args:
            B: Coefficient matrix (n_vars, n_vars*n_lags).
            n_vars: Number of endogenous variables.
            n_lags: Number of lags.
            steps: Number of forecast steps.

        Returns:
            List of MA coefficient matrices [Phi_0, ..., Phi_{steps-1}].
        """
        A_matrices = [B[:, j * n_vars : (j + 1) * n_vars] for j in range(n_lags)]
        ma_coefficients: list[np.ndarray] = [np.eye(n_vars)]
        for h in range(1, steps):
            phi_h = np.zeros((n_vars, n_vars))
            for j in range(min(h, n_lags)):
                phi_h += A_matrices[j] @ ma_coefficients[h - j - 1]
            ma_coefficients.append(phi_h)
        return ma_coefficients

    @staticmethod
    def _build_constraint_system(
        conditions: "list[ForecastCondition]",
        var_names: list[str],
        ma_coefficients: list[np.ndarray],
        impact_matrix: np.ndarray,
        unconditional: np.ndarray,
        steps: int,
        n_vars: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the linear constraint system R @ shocks = target.

        Args:
            conditions: List of ForecastCondition instances.
            var_names: Variable names.
            ma_coefficients: MA coefficient matrices.
            impact_matrix: Cholesky factor of Sigma or structural matrix P.
            unconditional: Unconditional forecast array (steps, n_vars).
            steps: Number of forecast steps.
            n_vars: Number of variables.

        Returns:
            Tuple of (R, target) arrays for the constraint system.
        """
        constraint_rows = []
        constraint_targets = []
        for cond in conditions:
            var_idx = var_names.index(cond.variable)
            for period, value in zip(cond.periods, cond.values, strict=True):
                row = np.zeros(steps * n_vars)
                for s in range(period + 1):
                    response = ma_coefficients[period - s] @ impact_matrix
                    row[s * n_vars : (s + 1) * n_vars] = response[var_idx, :]
                constraint_rows.append(row)
                constraint_targets.append(value - unconditional[period, var_idx])
        return np.array(constraint_rows), np.array(constraint_targets)

    def conditional_forecast(
        self,
        steps: int,
        conditions: "list[ForecastCondition]",
        exog_future: np.ndarray | None = None,
    ) -> "ConditionalForecastResult":
        """Produce conditional forecasts subject to constraints on future paths.

        Implements the Waggoner & Zha (1999) algorithm for hard constraints.
        Computes unconditional forecasts, then solves for the shock paths
        that satisfy the constraints.

        Args:
            steps: Number of forecast steps.
            conditions: List of ForecastCondition instances specifying constraints.
            exog_future: Future exogenous values if model has exog.

        Returns:
            ConditionalForecastResult with constrained posterior forecast draws.
        """
        import xarray as xr

        from impulso.results import ConditionalForecastResult

        self._validate_conditions(conditions, steps)

        B_draws = self.coefficients  # (C, D, n, n*p)
        intercept_draws = self.intercepts  # (C, D, n)
        sigma_draws = self.sigma  # (C, D, n, n)
        n_chains, n_draws, n_vars, _ = B_draws.shape

        y_hist = self.data.endog[-self.n_lags :]
        forecasts = np.zeros((n_chains, n_draws, steps, n_vars))

        for c in range(n_chains):
            for d in range(n_draws):
                B = B_draws[c, d]
                intercept = intercept_draws[c, d]
                chol_sigma = np.linalg.cholesky(sigma_draws[c, d])

                ma_coefficients = self._ma_coefficients_single(B, n_vars, self.n_lags, steps)
                unconditional = self._unconditional_forecast_single(
                    B, intercept, y_hist, steps, self.n_lags, n_vars, c, d, exog_future
                )

                R, target = self._build_constraint_system(
                    conditions, self.var_names, ma_coefficients, chol_sigma, unconditional, steps, n_vars
                )

                shocks, _, _, _ = np.linalg.lstsq(R, target, rcond=None)
                shocks = shocks.reshape(steps, n_vars)

                conditional = unconditional.copy()
                for h in range(steps):
                    for s in range(h + 1):
                        conditional[h] += ma_coefficients[h - s] @ chol_sigma @ shocks[s]

                forecasts[c, d] = conditional

        forecast_da = xr.DataArray(
            forecasts,
            dims=["chain", "draw", "step", "variable"],
            coords={"variable": self.var_names},
            name="forecast",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"forecast": forecast_da}))
        return ConditionalForecastResult(idata=idata, steps=steps, var_names=self.var_names, conditions=conditions)

    def _validate_conditions(self, conditions: "list[ForecastCondition]", steps: int) -> None:
        """Validate forecast conditions against model variables and step range."""
        for cond in conditions:
            if cond.variable not in self.var_names:
                raise ValueError(f"Condition variable '{cond.variable}' not in var_names {self.var_names}")
            for p in cond.periods:
                if p < 0 or p >= steps:
                    raise ValueError(f"Condition period {p} out of range for {steps} forecast steps")

    def _unconditional_forecast_single(
        self,
        B: np.ndarray,
        intercept: np.ndarray,
        y_hist: np.ndarray,
        steps: int,
        n_lags: int,
        n_vars: int,
        chain_idx: int,
        draw_idx: int,
        exog_future: np.ndarray | None,
    ) -> np.ndarray:
        """Compute unconditional forecast for a single posterior draw.

        Returns:
            Array of shape (steps, n_vars).
        """
        y_buffer = y_hist.copy()
        unconditional = np.zeros((steps, n_vars))
        for h in range(steps):
            x_lag = np.concatenate([y_buffer[-(lag + 1)] for lag in range(n_lags)])
            y_new = intercept + B @ x_lag
            if self.has_exog and exog_future is not None:
                B_exog = self.idata.posterior["B_exog"].values[chain_idx, draw_idx]
                y_new = y_new + B_exog @ exog_future[h]
            unconditional[h] = y_new
            y_buffer = np.vstack([y_buffer[1:], y_new.reshape(1, -1)])
        return unconditional

    def set_identification_strategy(self, scheme: IdentificationScheme) -> "IdentifiedVAR":
        """Apply a structural identification scheme.

        Args:
            scheme: An IdentificationScheme protocol instance (e.g. Cholesky).

        Returns:
            IdentifiedVAR with structural shock matrix in the posterior.
        """
        from impulso.identified import IdentifiedVAR

        identified_idata = scheme.identify(self.idata, self.var_names)
        return IdentifiedVAR.model_construct(
            idata=identified_idata,
            n_lags=self.n_lags,
            data=self.data,
            var_names=self.var_names,
        )
