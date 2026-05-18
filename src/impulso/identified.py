"""IdentifiedVAR — structural VAR with identified shocks."""

import warnings
from typing import Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field

from impulso._base import ImpulsoBaseModel
from impulso.data import VARData
from impulso.protocols import IdentificationScheme, VolatilityProcess
from impulso.results import FEVDResult, HistoricalDecompositionResult, IRFResult

# Type alias for the `at=` parameter used by query methods.
AtParam = int | Literal["last", "all"] | None


class IdentifiedVAR(ImpulsoBaseModel):
    """Immutable structural VAR with identified shocks.

    Attributes:
        idata: InferenceData with structural_shock_matrix in posterior.
        n_lags: Lag order.
        data: Original VARData.
        var_names: Endogenous variable names.
        volatility: Volatility process carried through from the fitted VAR.
            Required for ``at=`` queries on impulse_response / fevd /
            historical_decomposition (P3), which re-call
            ``volatility.cholesky_at(at)`` for the requested time slice.
        scheme: Identification scheme used to produce the structural shock
            matrix. Required for ``at=`` queries so the scheme can be
            re-applied to a different Cholesky factor on demand.
    """

    idata: az.InferenceData = Field(repr=False)
    n_lags: int
    data: VARData
    var_names: list[str]
    volatility: VolatilityProcess  # P3: needed for at= queries
    scheme: IdentificationScheme  # P3: needed for at= queries

    @property
    def shock_names(self) -> list[str]:
        """Shock coordinate labels from the structural shock matrix."""
        return list(self.idata.posterior["structural_shock_matrix"].coords["shock"].values)

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

    def _resolve_at(self, at: AtParam) -> int | None:
        """Resolve ``at=`` to an integer ``t`` suitable for ``cholesky_at(t)``.

        Returns ``None`` when ``at`` is ``None`` or ``"last"``. ``cholesky_at``
        adapters interpret ``t=None`` as "most recent" (SV) or "ignored"
        (Constant), so passing ``None`` through is the right default in both
        cases. Integer values are returned unchanged.

        Args:
            at: Either ``None``, ``"last"``, or an integer time index.
                ``"all"`` is not handled here — callers must dispatch to the
                per-t path before calling this helper.

        Returns:
            An integer time index, or ``None`` for the most-recent default.

        Raises:
            ValueError: If ``at`` is not one of the supported forms.
        """
        if at == "last" or at is None:
            return None
        if isinstance(at, int):
            return at
        raise ValueError(
            f"Invalid at= value: {at!r}. Expected int, 'last', or None. "
            "('all' must be handled by the caller before reaching _resolve_at.)"
        )

    def _identify_per_t(self, L_path: np.ndarray) -> np.ndarray:
        """Apply ``self.scheme.identify`` per time slice.

        Iterates the per-t loop in Python — fine for Cholesky (vectorised
        internally over draws), but expensive for ``SignRestriction`` at
        large ``T`` because rotations are re-sampled per time slice. A
        future optimisation could specialise the loop for time-invariant
        schemes, but P3 does not need it.

        Args:
            L_path: ``(C, D, T, n, n)`` Cholesky factor path.

        Returns:
            ``(C, D, T, n, n)`` structural shock matrix path.
        """
        T = L_path.shape[2]
        P_path = np.zeros_like(L_path)
        for t in range(T):
            P_path[:, :, t, :, :] = self.scheme.identify(
                L_path[:, :, t, :, :], self.var_names, posterior=self.idata.posterior
            )
        return P_path

    def impulse_response(self, horizon: int = 20, at: AtParam = None) -> IRFResult:
        """Compute structural impulse response functions.

        Args:
            horizon: Number of periods.
            at: Time index for the structural shock matrix:

                - ``None`` (default): for SV, equivalent to ``"last"``;
                  for Constant, ignored entirely.
                - ``int``: query the shock matrix at this specific ``t``.
                - ``"last"``: most recent time slice.
                - ``"all"``: return IRFs for shocks at every ``t`` (adds a
                  ``time`` dim to the result).

        Returns:
            IRFResult with IRF posterior draws.
        """
        B_draws = self.idata.posterior["B"].values  # (C, D, n, n*p)
        n_vars = B_draws.shape[2]
        Phi_arr = self._ma_coefficients(B_draws, n_vars, self.n_lags, horizon)

        if at == "all":
            # Per-t IRFs. T = effective in-sample length after lag trimming.
            T = self.data.endog.shape[0] - self.n_lags
            L_path = self.volatility.cholesky_path(self.idata.posterior, T=T)
            # L_path: (C, D, T, n, n); P_path: (C, D, T, n, n) by per-t identify.
            P_path = self._identify_per_t(L_path)
            # IRF: (C, D, T, H+1, n, n). Broadcast Phi over time and P_path
            # over horizon.
            irfs = Phi_arr[:, :, np.newaxis, :, :, :] @ P_path[:, :, :, np.newaxis, :, :]

            irf_da = xr.DataArray(
                irfs,
                dims=["chain", "draw", "time", "horizon", "response", "shock"],
                coords={
                    "response": self.var_names,
                    "shock": self.shock_names,
                    "horizon": np.arange(horizon + 1),
                    # Tuple form forces the coord onto the declared `time` dim;
                    # otherwise a named DatetimeIndex (e.g. .name == "date")
                    # would be inferred as its own dim and collide.
                    "time": ("time", self.data.index[self.n_lags :]),
                },
                name="irf",
            )
        else:
            # Single-time IRF.
            t = self._resolve_at(at)
            L = self.volatility.cholesky_at(self.idata.posterior, t=t)
            P = self.scheme.identify(L, self.var_names, posterior=self.idata.posterior)
            # IRF = Phi @ P, vectorised over (C, D, H) via broadcasting
            irfs = Phi_arr @ P[:, :, np.newaxis, :, :]

            irf_da = xr.DataArray(
                irfs,
                dims=["chain", "draw", "horizon", "response", "shock"],
                coords={
                    "response": self.var_names,
                    "shock": self.shock_names,
                    "horizon": np.arange(horizon + 1),
                },
                name="irf",
            )

        idata = az.InferenceData(posterior_predictive=xr.Dataset({"irf": irf_da}))
        return IRFResult(idata=idata, horizon=horizon, var_names=self.var_names)

    def forecast_error_variance_decomposition(self, horizon: int = 20, at: AtParam = None) -> FEVDResult:
        """Compute forecast error variance decomposition.

        Args:
            horizon: Number of periods.
            at: Time index for the structural shock matrix:

                - ``None`` (default): for SV, equivalent to ``"last"``;
                  for Constant, ignored entirely.
                - ``int``: query the shock matrix at this specific ``t``.
                - ``"last"``: most recent time slice.
                - ``"all"``: return FEVDs for shocks at every ``t`` (adds a
                  ``time`` dim to the result).

        Returns:
            FEVDResult with FEVD posterior draws.
        """
        B_draws = self.idata.posterior["B"].values  # (C, D, n, n*p)
        n_vars = B_draws.shape[2]
        Phi_arr = self._ma_coefficients(B_draws, n_vars, self.n_lags, horizon)

        if at == "all":
            # Per-t FEVDs. T = effective in-sample length after lag trimming.
            T = self.data.endog.shape[0] - self.n_lags
            L_path = self.volatility.cholesky_path(self.idata.posterior, T=T)
            # L_path: (C, D, T, n, n); P_path: (C, D, T, n, n) by per-t identify.
            P_path = self._identify_per_t(L_path)
            # Theta: (C, D, T, H+1, n, n). Broadcast Phi over time and P_path
            # over horizon.
            Theta = Phi_arr[:, :, np.newaxis, :, :, :] @ P_path[:, :, :, np.newaxis, :, :]

            # FEVD: cumulative MSE contribution, summed over the horizon axis (axis=3).
            mse_cum = np.cumsum(Theta**2, axis=3)  # (C, D, T, H+1, resp, shock)
            total = mse_cum.sum(axis=-1, keepdims=True)  # (C, D, T, H+1, resp, 1)
            fevd = np.where(total > 0, mse_cum / total, 0.0)

            fevd_da = xr.DataArray(
                fevd,
                dims=["chain", "draw", "time", "horizon", "response", "shock"],
                coords={
                    "response": self.var_names,
                    "shock": self.shock_names,
                    "horizon": np.arange(horizon + 1),
                    # Tuple form: see analogous comment in impulse_response.
                    "time": ("time", self.data.index[self.n_lags :]),
                },
                name="fevd",
            )
        else:
            # Single-time FEVD.
            t = self._resolve_at(at)
            L = self.volatility.cholesky_at(self.idata.posterior, t=t)
            P = self.scheme.identify(L, self.var_names, posterior=self.idata.posterior)
            # Theta_h = Phi_h @ P, vectorised via broadcasting
            Theta = Phi_arr @ P[:, :, np.newaxis, :, :]

            # FEVD: cumulative MSE contribution
            mse_cum = np.cumsum(Theta**2, axis=2)  # (C, D, H+1, resp, shock)
            total = mse_cum.sum(axis=-1, keepdims=True)  # (C, D, H+1, resp, 1)
            fevd = np.where(total > 0, mse_cum / total, 0.0)

            fevd_da = xr.DataArray(
                fevd,
                dims=["chain", "draw", "horizon", "response", "shock"],
                coords={
                    "response": self.var_names,
                    "shock": self.shock_names,
                    "horizon": np.arange(horizon + 1),
                },
                name="fevd",
            )

        idata = az.InferenceData(posterior_predictive=xr.Dataset({"fevd": fevd_da}))
        return FEVDResult(idata=idata, horizon=horizon, var_names=self.var_names)

    def fevd(self, horizon: int = 20, at: AtParam = None) -> FEVDResult:
        """Alias for forecast_error_variance_decomposition.

        Args:
            horizon: Number of periods.
            at: See :meth:`forecast_error_variance_decomposition`.

        Returns:
            FEVDResult.
        """
        return self.forecast_error_variance_decomposition(horizon=horizon, at=at)

    def historical_decomposition(
        self,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        cumulative: bool = False,
        at: AtParam = None,
    ) -> HistoricalDecompositionResult:
        """Compute historical decomposition of observed series.

        Historical decomposition is intrinsically time-indexed: it attributes
        each in-sample observation to past structural shocks. The ``at=``
        parameter controls which Cholesky factor identifies those shocks.

        Args:
            start: Optional start date to restrict decomposition.
            end: Optional end date to restrict decomposition.
            cumulative: If True, return cumulative shock contributions.
            at: Time index for the structural shock matrix:

                - ``None`` (default) or ``"all"``: use ``cholesky_path`` —
                  identify the shock at each ``t`` with its own ``L_t``.
                  For stochastic volatility this is the correct structural
                  decomposition; for constant volatility ``L_t`` is the same
                  for all ``t`` and the result matches the legacy behaviour.
                - ``int`` or ``"last"``: identify every shock with a single
                  ``L`` queried at the supplied time index. Under stochastic
                  volatility this is a non-standard hypothetical ("what if
                  regime ``t`` had prevailed throughout?") and emits a
                  ``UserWarning``. For constant volatility the result is
                  identical to the default.

        Returns:
            HistoricalDecompositionResult.
        """
        B_draws = self.idata.posterior["B"].values  # (C, D, n, n*p)
        intercept_draws = self.idata.posterior["intercept"].values  # (C, D, n)

        y = self.data.endog  # (T, n)
        T = y.shape[0]
        n_lags = self.n_lags
        T_eff = T - n_lags

        # Build lag matrix: x_lag[t] = [y[t-1], y[t-2], ...] for t in [n_lags, T)
        x_lag = np.concatenate([y[n_lags - lag : T - lag] for lag in range(1, n_lags + 1)], axis=1)  # (T-p, n*p)

        # Predicted values: intercept + B @ x_lag for all (C, D, t)
        y_hat = intercept_draws[:, :, np.newaxis, :] + np.einsum("cdij,tj->cdti", B_draws, x_lag)

        # Reduced-form residuals: (C, D, T-p, n)
        y_obs = y[n_lags:]  # (T-p, n)
        resid = y_obs[np.newaxis, np.newaxis, :, :] - y_hat

        # Build a per-t structural shock matrix path P_path: (C, D, T_eff, n, n).
        if at is None or at == "all":
            # Per-t identification — correct for SV; broadcasts to constant L
            # for Constant volatility, matching the legacy single-P behaviour.
            L_path = self.volatility.cholesky_path(self.idata.posterior, T=T_eff)
            P_path = self._identify_per_t(L_path)
        else:
            # Single-L hypothetical applied across all t.
            if getattr(self.volatility, "name", None) == "sv":
                warnings.warn(
                    f"historical_decomposition(at={at!r}) under stochastic "
                    "volatility applies a single L across every in-sample "
                    "period — this is a non-standard hypothetical "
                    '("what if regime t had prevailed throughout?"), not '
                    "the standard structural decomposition. Pass at=None "
                    "or at='all' for the correct per-t decomposition.",
                    UserWarning,
                    stacklevel=2,
                )
            t = self._resolve_at(at)
            L = self.volatility.cholesky_at(self.idata.posterior, t=t)
            P = self.scheme.identify(L, self.var_names, posterior=self.idata.posterior)
            # Broadcast P across the time axis.
            P_path = np.broadcast_to(P[:, :, np.newaxis, :, :], (*P.shape[:2], T_eff, *P.shape[2:])).copy()

        # Structural residuals: P_t^{-1} @ resid_t  -> (C, D, T-p, n)
        P_inv_path = np.linalg.inv(P_path)  # (C, D, T_eff, n, n)
        structural_resid = np.einsum("cdtij,cdtj->cdti", P_inv_path, resid)

        # hd[..., t, resp, shock] = P_t[resp, shock] * structural_resid[t, shock]
        hd = P_path * structural_resid[:, :, :, np.newaxis, :]

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
            coords={
                "response": self.var_names,
                "shock": self.shock_names,
                # Tuple form forces the coord onto the declared `time` dim;
                # otherwise a named DatetimeIndex (e.g. .name == "date")
                # would be inferred as its own dim and collide.
                "time": ("time", idx[t_start:t_end]),
            },
            name="hd",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"hd": hd_da}))
        return HistoricalDecompositionResult(idata=idata, var_names=self.var_names)
