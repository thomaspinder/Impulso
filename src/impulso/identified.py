"""IdentifiedVAR — structural VAR with identified shocks."""

import warnings
from typing import Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field

from impulso._base import ImpulsoBaseModel
from impulso._ma import compute_ma_phi
from impulso.data import VARData
from impulso.protocols import IdentificationScheme, VolatilityProcess
from impulso.results import FEVDResult, HistoricalDecompositionResult, IRFResult

# Type alias for the `at=` parameter used by query methods.
AtParam = int | Literal["last", "all"] | None


class IdentifiedVAR(ImpulsoBaseModel):
    """Immutable structural VAR with identified shocks.

    Attributes:
        idata: InferenceData with reduced-form posterior (B, intercept, L, ...).
        n_lags: Lag order.
        data: Original VARData.
        var_names: Endogenous variable names.
        volatility: Volatility process carried through from the fitted VAR.
            Required for `at=` queries on impulse_response / fevd /
            historical_decomposition (P3), which re-call
            `volatility.cholesky_at(at)` for the requested time slice.
        scheme: Identification scheme used to produce the structural shock
            matrix. Required for `at=` queries so the scheme can be
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
        """Shock coordinate labels from the identification scheme."""
        return self.scheme.shock_coords(n_vars=len(self.var_names))

    def _ma_coefficients(self, B_draws: np.ndarray, n_vars: int, n_lags: int, horizon: int) -> np.ndarray:
        """Compute MA coefficient recursion, vectorised over (chains, draws).

        Returns:
            Array of shape (C, D, horizon+1, n_vars, n_vars).
        """
        A = [B_draws[:, :, :, j * n_vars : (j + 1) * n_vars] for j in range(n_lags)]
        return compute_ma_phi(A, horizon)

    def shock_matrix(self, at: AtParam = None) -> xr.DataArray:
        """Query the structural shock matrix at a given time index.

        This is the single pathway from the volatility process and
        identification scheme to a labelled structural shock matrix.
        IRF, FEVD, and historical decomposition all compute through it.
        Results are memoised per *at* value on this instance so that all
        quantities from one ``IdentifiedVAR`` share the same structural
        draws (deterministic per object, even under ``SignRestriction``).

        Args:
            at: Time index.  ``None`` or ``"last"`` → most recent slice.
                An integer ``t`` → that specific time index.
                ``"all"`` → full time path (adds a ``time`` dim).

        Returns:
            DataArray with dims ``(chain, draw[, time], response, shock)``.

        Raises:
            ValueError: If ``at="all"`` under constant volatility.
        """
        # Check memoisation cache.
        cache_attr = f"_shock_matrix_cache_{at!r}"
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached

        shock_coords = self.shock_names

        if at == "all":
            if not self.volatility.is_time_varying:
                raise ValueError(
                    "shock_matrix(at='all') is only meaningful for "
                    "time-varying volatility. The current volatility "
                    f"process ({type(self.volatility).__name__}) is "
                    "time-invariant — use at=None or at='last'."
                )
            T = self.data.endog.shape[0] - self.n_lags
            L_path = self.volatility.cholesky_path(self.idata.posterior, T=T)
            P_path = self._identify_per_t(L_path)
            result = xr.DataArray(
                P_path,
                dims=["chain", "draw", "time", "response", "shock"],
                coords={
                    "response": self.var_names,
                    "shock": shock_coords,
                    "time": ("time", self.data.index[self.n_lags :]),
                },
                name="structural_shock_matrix",
            )
        else:
            t = self._resolve_at(at)
            L = self.volatility.cholesky_at(self.idata.posterior, t=t)
            P = self.scheme.identify(L, self.var_names, posterior=self.idata.posterior)
            result = xr.DataArray(
                P,
                dims=["chain", "draw", "response", "shock"],
                coords={
                    "response": self.var_names,
                    "shock": shock_coords,
                },
                name="structural_shock_matrix",
            )

        # Attach sign-restriction acceptance rate if available.
        rate = getattr(self.scheme, "_last_acceptance_rate", None)
        if isinstance(rate, float) and rate < 1.0:
            result.attrs["sign_restriction_acceptance_rate"] = rate

        object.__setattr__(self, cache_attr, result)
        return result

    def _resolve_at(self, at: AtParam) -> int | None:
        """Resolve `at=` to an integer `t` suitable for `cholesky_at(t)`.

        Returns `None` when `at` is `None` or `"last"`. `cholesky_at`
        adapters interpret `t=None` as "most recent" (SV) or "ignored"
        (Constant), so passing `None` through is the right default in both
        cases. Integer values are returned unchanged.

        Args:
            at: Either `None`, `"last"`, or an integer time index.
                `"all"` is not handled here — callers must dispatch to the
                per-t path before calling this helper.

        Returns:
            An integer time index, or `None` for the most-recent default.

        Raises:
            ValueError: If `at` is not one of the supported forms.
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
        """Apply `self.scheme.identify` per time slice.

        Iterates the per-t loop in Python — fine for Cholesky (vectorised
        internally over draws), but expensive for `SignRestriction` at
        large `T` because rotations are re-sampled per time slice. A
        future optimisation could specialise the loop for time-invariant
        schemes, but P3 does not need it.

        Args:
            L_path: `(C, D, T, n, n)` Cholesky factor path.

        Returns:
            `(C, D, T, n, n)` structural shock matrix path.
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
            at: Time index for the structural shock matrix
                (see :meth:`shock_matrix` for accepted forms).

        Returns:
            IRFResult with IRF posterior draws.
        """
        B_draws = self.idata.posterior["B"].values  # (C, D, n, n*p)
        n_vars = B_draws.shape[2]
        Phi_arr = self._ma_coefficients(B_draws, n_vars, self.n_lags, horizon)
        P = self.shock_matrix(at=at)

        if "time" in P.dims:
            # P: (C, D, T, n, n) → IRF: (C, D, T, H+1, n, n)
            irfs = Phi_arr[:, :, np.newaxis, :, :, :] @ P.values[:, :, :, np.newaxis, :, :]
            irf_da = xr.DataArray(
                irfs,
                dims=["chain", "draw", "time", "horizon", "response", "shock"],
                coords={
                    "response": self.var_names,
                    "shock": self.shock_names,
                    "horizon": np.arange(horizon + 1),
                    "time": P.coords["time"],
                },
                name="irf",
            )
        else:
            # P: (C, D, n, n) → IRF: (C, D, H+1, n, n)
            irfs = Phi_arr @ P.values[:, :, np.newaxis, :, :]
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

    def fevd(self, horizon: int = 20, at: AtParam = None) -> FEVDResult:
        """Compute forecast error variance decomposition.

        Args:
            horizon: Number of periods.
            at: Time index for the structural shock matrix
                (see :meth:`shock_matrix` for accepted forms).

        Returns:
            FEVDResult with FEVD posterior draws.
        """
        B_draws = self.idata.posterior["B"].values  # (C, D, n, n*p)
        n_vars = B_draws.shape[2]
        Phi_arr = self._ma_coefficients(B_draws, n_vars, self.n_lags, horizon)
        P = self.shock_matrix(at=at)

        if "time" in P.dims:
            Theta = Phi_arr[:, :, np.newaxis, :, :, :] @ P.values[:, :, :, np.newaxis, :, :]
            mse_cum = np.cumsum(Theta**2, axis=3)
            total = mse_cum.sum(axis=-1, keepdims=True)
            fevd_arr = np.where(total > 0, mse_cum / total, 0.0)
            fevd_da = xr.DataArray(
                fevd_arr,
                dims=["chain", "draw", "time", "horizon", "response", "shock"],
                coords={
                    "response": self.var_names,
                    "shock": self.shock_names,
                    "horizon": np.arange(horizon + 1),
                    "time": P.coords["time"],
                },
                name="fevd",
            )
        else:
            Theta = Phi_arr @ P.values[:, :, np.newaxis, :, :]
            mse_cum = np.cumsum(Theta**2, axis=2)
            total = mse_cum.sum(axis=-1, keepdims=True)
            fevd_arr = np.where(total > 0, mse_cum / total, 0.0)
            fevd_da = xr.DataArray(
                fevd_arr,
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

    def historical_decomposition(
        self,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        cumulative: bool = False,
        at: AtParam = None,
    ) -> HistoricalDecompositionResult:
        """Compute historical decomposition of observed series.

        Historical decomposition is intrinsically time-indexed: it attributes
        each in-sample observation to past structural shocks.  The ``at=``
        parameter controls which Cholesky factor identifies those shocks.

        Args:
            start: Optional start date to restrict decomposition.
            end: Optional end date to restrict decomposition.
            cumulative: If True, return cumulative shock contributions.
            at: Time index for the structural shock matrix.
                ``None`` or ``"all"`` → per-t decomposition (correct for SV,
                identical to single-L under constant volatility).
                ``int`` or ``"last"`` → single-L hypothetical (warns under SV).

        Returns:
            HistoricalDecompositionResult.
        """
        B_draws = self.idata.posterior["B"].values  # (C, D, n, n*p)
        intercept_draws = self.idata.posterior["intercept"].values  # (C, D, n)

        y = self.data.endog  # (T, n)
        T = y.shape[0]
        n_lags = self.n_lags

        x_lag = np.concatenate(
            [y[n_lags - lag : T - lag] for lag in range(1, n_lags + 1)],
            axis=1,
        )  # (T-p, n*p)
        y_hat = intercept_draws[:, :, np.newaxis, :] + np.einsum("cdij,tj->cdti", B_draws, x_lag)
        y_obs = y[n_lags:]
        resid = y_obs[np.newaxis, np.newaxis, :, :] - y_hat

        use_per_t = self.volatility.is_time_varying and at in (None, "all")
        if use_per_t:
            P = self.shock_matrix(at="all").values  # (C, D, T_eff, n, n)
            P_inv = np.linalg.inv(P)
            structural_resid = np.einsum("cdtij,cdtj->cdti", P_inv, resid)
            hd = P * structural_resid[:, :, :, np.newaxis, :]
        else:
            if self.volatility.is_time_varying:
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
            # For constant vol, at='all' is equivalent to at=None (single L).
            shock_at = None if (at == "all" and not self.volatility.is_time_varying) else at
            P = self.shock_matrix(at=shock_at).values  # (C, D, n, n)
            P_inv = np.linalg.inv(P)
            structural_resid = np.einsum("cdij,cdtj->cdti", P_inv, resid)
            hd = P[:, :, np.newaxis, :, :] * structural_resid[:, :, :, np.newaxis, :]

        if cumulative:
            hd = np.cumsum(hd, axis=2)

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
                "time": ("time", idx[t_start:t_end]),
            },
            name="hd",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"hd": hd_da}))
        return HistoricalDecompositionResult(idata=idata, var_names=self.var_names)
