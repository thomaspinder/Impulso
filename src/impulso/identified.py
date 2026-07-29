"""IdentifiedVAR — structural VAR with identified shocks."""

import warnings
from typing import TYPE_CHECKING, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field

from impulso._base import ImpulsoBaseModel
from impulso._linalg import lag_matrices
from impulso._ma import compute_ma_phi
from impulso.data import VARData
from impulso.observation import Gaussian
from impulso.protocols import ErrorDistribution, IdentificationScheme, VolatilityProcess
from impulso.results import (
    CounterfactualResult,
    FEVDResult,
    HistoricalDecompositionResult,
    IRFResult,
    ScenarioResult,
)

if TYPE_CHECKING:
    from impulso.scenario import ShockPath, VariablePath

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
        error_dist: Observation error distribution carried through from the
            fitted VAR (defaults to `Gaussian()`). Identification itself is
            unaffected — every scheme factorises the scale matrix — but
            `structural_scenario` refuses to run under a heavy-tailed law.
        scheme: Identification scheme used to produce the structural shock
            matrix. Required for `at=` queries so the scheme can be
            re-applied to a different Cholesky factor on demand.
    """

    idata: az.InferenceData = Field(repr=False)
    n_lags: int
    data: VARData
    var_names: list[str]
    volatility: VolatilityProcess  # P3: needed for at= queries
    # Suppression as on FittedVAR.error_dist: the Literal discriminator
    # convention makes no adapter assignable to its own Protocol under ty's
    # invariant attribute rule.
    error_dist: ErrorDistribution = Field(default_factory=Gaussian)  # ty: ignore[invalid-assignment]
    scheme: IdentificationScheme  # P3: needed for at= queries

    @property
    def shock_names(self) -> list[str]:
        """Shock coordinate labels from the identification scheme."""
        return self.scheme.shock_coords(n_vars=len(self.var_names))

    def _ma_coefficients(self, B_draws: np.ndarray, n_lags: int, horizon: int) -> np.ndarray:
        """Compute MA coefficient recursion, vectorised over (chains, draws).

        Returns:
            Array of shape (C, D, horizon+1, n_vars, n_vars).
        """
        return compute_ma_phi(lag_matrices(B_draws, n_lags), horizon)

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
            P = self.scheme.identify(
                L, self.var_names, posterior=self.idata.posterior, data=self.data, n_lags=self.n_lags
            )
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
        if isinstance(rate, float):
            result.attrs["sign_restriction_acceptance_rate"] = rate

        # Attach any scheme-specific diagnostics (e.g. ProxySVAR first-stage
        # strength) the same way — identify() stashes them, we surface them.
        diagnostics = getattr(self.scheme, "_last_diagnostics", None)
        if diagnostics:
            result.attrs.update(diagnostics)

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
                L_path[:, :, t, :, :],
                self.var_names,
                posterior=self.idata.posterior,
                data=self.data,
                n_lags=self.n_lags,
            )
        return P_path

    def impulse_response(self, horizon: int = 20, at: AtParam = None) -> IRFResult:
        """Compute structural impulse response functions.

        Note:
            **Shock size under heavy-tailed errors.** A "unit shock" is one
            column of the structural matrix `P`, which factorises the
            *scale* matrix Ω. Under Gaussian errors that is one
            unconditional standard deviation. Under Student-t errors it is
            one *scale* unit, which is `sqrt((nu-2)/nu)` unconditional
            standard deviations — about 0.82 sd at nu = 6. Responses are
            therefore smaller by that constant factor than the
            one-standard-deviation convention; multiply by
            `sqrt(nu/(nu-2))` to restore it. Relative shapes, ratios between
            responses, and FEVD shares are unaffected. See ADR-0007.

        Args:
            horizon: Number of periods.
            at: Time index for the structural shock matrix
                (see :meth:`shock_matrix` for accepted forms).

        Returns:
            IRFResult with IRF posterior draws.
        """
        B_draws = self.idata.posterior["B"].values  # (C, D, n, n*p)
        Phi_arr = self._ma_coefficients(B_draws, self.n_lags, horizon)
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

    def _fevd_guard(self, fevd_arr: np.ndarray) -> np.ndarray:
        """Mask FEVD shares that are not identified.

        Columns labelled ``unidentified_*`` (partial identification —
        `ProxySVAR`, or `SignRestriction` naming fewer shocks than
        variables) are rotation-arbitrary: their individual variance
        shares depend on an arbitrary orthogonal completion and carry no
        economic content. They are masked to NaN rather than reported.
        The identified columns' shares remain valid — the denominator
        (total forecast-error variance) is rotation-invariant.

        Additionally warns when the scheme applies a unit-effect
        rescaling (a `scale` attribute set to a float, as in
        `ProxySVAR(scale=...)`): variance shares are only interpretable
        under the one-standard-deviation convention.
        """
        import warnings

        masked = [i for i, s in enumerate(self.shock_names) if s.startswith("unidentified_")]
        if masked:
            warnings.warn(
                f"FEVD shares for {len(masked)} unidentified shock column(s) are "
                "rotation-arbitrary and have been masked to NaN. Only the named "
                "shock columns carry identified variance shares.",
                UserWarning,
                stacklevel=3,
            )
            fevd_arr = fevd_arr.copy()
            fevd_arr[..., masked] = np.nan
        if getattr(self.scheme, "scale", None) is not None:
            warnings.warn(
                "The identification scheme applies a unit-effect rescaling "
                "(scale is set); FEVD shares assume one-standard-deviation "
                "shocks and are not interpretable under this normalisation. "
                "Re-identify with scale=None for variance decomposition.",
                UserWarning,
                stacklevel=3,
            )
        return fevd_arr

    @staticmethod
    def _shares(mse_cum: np.ndarray, total: np.ndarray) -> np.ndarray:
        """Normalise cumulative MSE contributions into variance shares.

        The zero-total guard exists for the degenerate horizon-0 case; it
        must not swallow *undefined* draws. `NaN > 0` is `False`, so a
        plain `np.where(total > 0, ...)` would report a NaN draw as a
        clean 0.0 share — indistinguishable from "this shock explains
        nothing". Schemes can legitimately return NaN draws
        (`LongRunRestriction` blanks draws whose long-run multiplier is
        numerically undefined), so NaN is re-imposed after the guard.
        """
        shares = np.where(total > 0, mse_cum / total, 0.0)
        return np.where(np.isnan(total), np.nan, shares)

    def fevd(self, horizon: int = 20, at: AtParam = None) -> FEVDResult:
        """Compute forecast error variance decomposition.

        Under partial identification (any shock column labelled
        ``unidentified_*``), the shares of the unidentified columns are
        masked to NaN — see :meth:`_fevd_guard`.

        Note:
            FEVD is **exactly invariant** to the scale-vs-covariance
            convention, so nothing here needs a Student-t correction: the
            shares are built from `Theta = Phi @ P`, and rescaling
            `P -> cP` multiplies numerator and denominator by `c^2`, which
            cancels. Do not "fix" this by inflating `P` by
            `sqrt(nu/(nu-2))` — it would change nothing except the
            rounding. See ADR-0007.

        Args:
            horizon: Number of periods.
            at: Time index for the structural shock matrix
                (see :meth:`shock_matrix` for accepted forms).

        Returns:
            FEVDResult with FEVD posterior draws.
        """
        B_draws = self.idata.posterior["B"].values  # (C, D, n, n*p)
        Phi_arr = self._ma_coefficients(B_draws, self.n_lags, horizon)
        P = self.shock_matrix(at=at)

        if "time" in P.dims:
            Theta = Phi_arr[:, :, np.newaxis, :, :, :] @ P.values[:, :, :, np.newaxis, :, :]
            mse_cum = np.cumsum(Theta**2, axis=3)
            total = mse_cum.sum(axis=-1, keepdims=True)
            fevd_arr = self._shares(mse_cum, total)
            fevd_arr = self._fevd_guard(fevd_arr)
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
            fevd_arr = self._shares(mse_cum, total)
            fevd_arr = self._fevd_guard(fevd_arr)
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
        at: AtParam = None,
    ) -> HistoricalDecompositionResult:
        """Compute the propagated historical decomposition of the observed series.

        Attributes each in-sample observation to a deterministic baseline
        (initial conditions, intercept, and any exogenous path) plus the
        *propagated* contribution of each structural shock,

            c_{j,t} = P_t[:, j] eps_{j,t} + sum_i A_i c_{j,t-i},

        so that `y_t = baseline_t + sum_j c_{j,t}` holds exactly for every
        posterior draw. Contributions carry forward through the lag
        dynamics: a shock keeps contributing beyond its impact period. The
        `at=` parameter controls which Cholesky factor identifies the
        shocks.

        Under partial identification (shock columns labelled
        `unidentified_*`), the individual contributions of the
        unidentified shocks are rotation-arbitrary, but their *sum* is
        well-defined (it is the variation the identified shocks do not
        explain). Those columns are therefore collapsed into a single
        `unidentified_remainder` column. Propagation is linear in the
        impact, so the decomposition remains exactly additive and the
        identified shocks' contributions stay invariant to both the
        orthogonal completion and any unit-effect column rescaling.

        Note:
            **Breaking change (scenario-analysis stack, 2026-07)**: earlier
            releases decomposed only the contemporaneous residual
            `u_t = sum_j P[:, j] eps_{j,t}` and offered a plain cumulative
            sum via `cumulative=`. The decomposition now propagates through
            the lag dynamics and always satisfies the additivity identity;
            the `cumulative` parameter is retired.

        Args:
            start: Optional start date to restrict the returned window.
                Contributions are always propagated from the start of the
                estimation sample; the filter only slices the output.
            end: Optional end date to restrict the returned window.
            at: Time index for the structural shock matrix.
                `None` or `"all"` → per-t identification (correct for SV,
                identical to single-L under constant volatility).
                `int` or `"last"` → single-L hypothetical (warns under SV).

        Returns:
            HistoricalDecompositionResult carrying the contribution draws
            (`"hd"`) and the deterministic baseline (`"baseline"`).
        """
        from impulso._propagate import propagate, propagate_contributions
        from impulso._residuals import reduced_form_residuals

        n_lags = self.n_lags
        posterior = self.idata.posterior
        resid = reduced_form_residuals(posterior, self.data, n_lags)

        use_per_t = self.volatility.is_time_varying and at in (None, "all")
        if use_per_t:
            P = self.shock_matrix(at="all").values  # (C, D, T_eff, n, n)
            P_inv = np.linalg.inv(P)
            structural_resid = np.einsum("cdtij,cdtj->cdti", P_inv, resid)
            impact = P * structural_resid[:, :, :, np.newaxis, :]
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
            impact = P[:, :, np.newaxis, :, :] * structural_resid[:, :, :, np.newaxis, :]

        A = lag_matrices(posterior["B"].values, n_lags)
        hd = propagate_contributions(A, impact)

        intercept = posterior["intercept"].values  # (C, D, n)
        n_chains, n_draws, n_vars = intercept.shape
        T_eff = resid.shape[2]
        forcing = np.broadcast_to(intercept[:, :, np.newaxis, :], (n_chains, n_draws, T_eff, n_vars)).copy()
        if self.data.exog is not None and "B_exog" in posterior:
            forcing += np.einsum("cdij,tj->cdti", posterior["B_exog"].values, self.data.exog[n_lags:])
        baseline = propagate(A, forcing, self.data.endog[:n_lags])

        idx = self.data.index[n_lags:]
        t_start = idx.searchsorted(start) if start is not None else 0
        t_end = idx.searchsorted(end, side="right") if end is not None else len(idx)
        hd = hd[:, :, t_start:t_end]
        baseline = baseline[:, :, t_start:t_end]

        # Partial identification: collapse rotation-arbitrary columns into
        # one well-defined remainder (their sum is completion-invariant;
        # propagation is linear in the impact, so the invariance carries
        # over to the propagated contributions).
        shock_coord = list(self.shock_names)
        unident = [i for i, s in enumerate(shock_coord) if s.startswith("unidentified_")]
        if unident:
            ident = [i for i in range(len(shock_coord)) if i not in unident]
            remainder = hd[..., unident].sum(axis=-1, keepdims=True)
            hd = np.concatenate([hd[..., ident], remainder], axis=-1)
            shock_coord = [shock_coord[i] for i in ident] + ["unidentified_remainder"]

        time_coord = ("time", idx[t_start:t_end])
        hd_da = xr.DataArray(
            hd,
            dims=["chain", "draw", "time", "response", "shock"],
            coords={
                "response": self.var_names,
                "shock": shock_coord,
                "time": time_coord,
            },
            name="hd",
        )
        baseline_da = xr.DataArray(
            baseline,
            dims=["chain", "draw", "time", "response"],
            coords={"response": self.var_names, "time": time_coord},
            name="baseline",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"hd": hd_da, "baseline": baseline_da}))
        return HistoricalDecompositionResult(idata=idata, var_names=self.var_names)

    def counterfactual(
        self,
        shocks: "list[ShockPath]",
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> CounterfactualResult:
        """Historical counterfactual: edit realised structural shocks and re-propagate.

        Backs out the realised structural shocks per posterior draw
        (`eps_t = P_t⁻¹ u_t`), overwrites the paths named by `shocks`
        (`ShockPath` values are in one-standard-deviation units; `0.0`
        switches a shock off; windows resolve against the lag-trimmed
        index), and re-runs the lag recursion from the actual initial
        conditions. Realised shocks are edited, never re-drawn, so the
        posterior spread of the counterfactual reflects parameter and
        identification uncertainty only. With `shocks=[]` the observed
        sample is reproduced exactly.

        For a shock zeroed over the *full* sample,
        `actual - counterfactual` equals that shock's historical-
        decomposition contribution exactly, per draw. For a *windowed*
        zero-edit it instead equals the propagation of the shock's
        innovations dated inside the window only — zero before the window,
        persisting (decaying under stability) after it — which is *not*
        the windowed slice of the full-sample HD contribution (that slice
        also carries earlier impulses).

        Note:
            The Lucas critique applies: fixed-path shock edits assume the
            estimated reduced-form dynamics are invariant to the
            intervention. Policy-rule replacement is a different object
            and out of scope.

        Note:
            Valid under every error distribution, and *exactly* invariant
            to the scale-vs-covariance convention for zero edits: shocks are
            backed out as `eps = P⁻¹u` and re-propagated as `P eps`, so
            rescaling `P -> cP` leaves `P eps` untouched. Non-zero
            `ShockPath` values are the exception — they are stated in units
            of one column of `P`, which under Student-t errors is one
            *scale* unit rather than one unconditional standard deviation
            (multiply by `sqrt(nu/(nu-2))` for the sd convention).

        Args:
            shocks: `ShockPath` edits to impose (may be empty).
            start: Optional start of the *returned* window. The simulation
                always runs from the sample start; `start`/`end` only
                slice the output (the `historical_decomposition`
                convention). Edit windows live on the `ShockPath` objects.
            end: Optional end of the returned window.

        Returns:
            CounterfactualResult carrying the counterfactual draws and the
            actual path over the same window.
        """
        from impulso._scenario import counterfactual_paths

        n_lags = self.n_lags
        y_cf = counterfactual_paths(self, list(shocks))

        idx = self.data.index[n_lags:]
        t_start = idx.searchsorted(start) if start is not None else 0
        t_end = idx.searchsorted(end, side="right") if end is not None else len(idx)
        y_cf = y_cf[:, :, t_start:t_end]
        actual = self.data.endog[n_lags:][t_start:t_end]

        time_coord = ("time", idx[t_start:t_end])
        cf_da = xr.DataArray(
            y_cf,
            dims=["chain", "draw", "time", "variable"],
            coords={"variable": self.var_names, "time": time_coord},
            name="counterfactual",
        )
        actual_da = xr.DataArray(
            actual,
            dims=["time", "variable"],
            coords={"variable": self.var_names, "time": time_coord},
            name="actual",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"counterfactual": cf_da, "actual": actual_da}))
        return CounterfactualResult(idata=idata, var_names=self.var_names)

    def structural_scenario(
        self,
        steps: int,
        conditions: "list[VariablePath] | None" = None,
        shocks: "list[ShockPath] | None" = None,
        adjusting: list[str] | None = None,
        include_shock_uncertainty: bool = True,
        seed: int | np.random.Generator | None = None,
        exog_future: np.ndarray | None = None,
        path_uncertainty: Literal["none", "unconditional"] = "none",
    ) -> ScenarioResult:
        """Structural scenario: conditions absorbed by named shocks, paths prescribed.

        The ADPRR structural scenario (Antolín-Díaz, Petrella &
        Rubio-Ramírez 2021), combinable in both flavours:
        *conditional-on-observables* — `VariablePath` pins that must be
        absorbed by the `adjusting` shocks while non-adjusting shocks keep
        their unconditional draws — and *conditional-on-shocks* —
        forecast-side `ShockPath` prescriptions, substituted outright
        (positional from step 1; a prescription always wins over
        adjusting membership at its steps). With `adjusting=None` all
        shocks adjust, and with no prescriptions the result reproduces
        `conditional_forecast` (exactly per draw under natural-order
        Cholesky identification with a matched `seed`).

        Feasibility is enforced twice: once at validation (conditions
        must not outnumber the effective adjusting entries, globally or
        in any leading horizon block) and per posterior draw (numerical
        rank of the adjusting-block constraint matrix — a Cholesky zero
        can make a condition load on no adjusting shock at its step).
        Infeasible draws error rather than being dropped, which would
        condition the posterior on feasibility.

        The per-draw plausibility statistic includes the prescribed
        shocks' own magnitude: `q = c̃'(C_A C_A')⁻¹c̃ + |v_S|²` in
        one-standard-deviation units — prescribing a 3-sd shock registers
        as `q += 9` even though prescriptions are substituted. The
        ADPRR-calibrated `q_cal` is finite only under
        `path_uncertainty="unconditional"` with no prescriptions.

        Note:
            Under time-varying volatility the scheme-identified forecast
            factors are built per simulated volatility path
            (conditional-on-path; see ADR-0005), and `SignRestriction` is
            not supported there — the scheme re-samples rotations per
            call, so no single structural coordinate system spans the
            forecast steps. Under constant volatility the memoised
            `shock_matrix` is broadcast, sharing rotation draws with
            `counterfactual` and the historical decomposition on this
            instance.

        Note:
            Gaussian errors only. The ADPRR three-way partition solves a
            Gaussian conditioning problem for the adjusting block, and the
            plausibility statistic's `chi^2_r` reference assumes Gaussian
            shocks. Under `error_dist="student_t"` this method raises
            `NotImplementedError` rather than returning a half-valid
            answer.

        Args:
            steps: Number of forecast steps.
            conditions: `VariablePath` pins to be absorbed by the
                adjusting shocks.
            shocks: Forecast-side `ShockPath` prescriptions (no
                `start`/`end`; positional from step 1; `NaN` = free).
            adjusting: Names of the shocks permitted to absorb the
                conditions. `None` (default) lets every shock adjust.
                Must contain none or all of any `unidentified_*` columns.
            include_shock_uncertainty: Density mode (default) vs mean
                mode (free block zeroed, conditional mean propagated).
            seed: RNG seed (int) or Generator.
            exog_future: Future exogenous values, shape `(steps, k)`.
                Required if the posterior carries `B_exog`.
            path_uncertainty: `"none"` (hard pins) or `"unconditional"`
                (pins restrict the mean; bands keep unconditional width).

        Returns:
            ScenarioResult with forecast draws, the scenario ingredients
            echoed, and the plausibility statistics.

        Raises:
            NotImplementedError: If the model was fitted with a
                heavy-tailed error distribution.
            ValueError: On unknown shocks/variables, `unidentified_*`
                references, in-sample windows on prescriptions, duplicate
                pins or prescriptions, over-determination, per-draw rank
                failure, `SignRestriction` under time-varying volatility,
                an invalid `path_uncertainty`, or exogenous-data
                mismatches.
        """
        from scipy.stats import chi2

        from impulso._scenario import structural_scenario_engine

        if self.error_dist.is_heavy_tailed:
            raise NotImplementedError(
                "structural_scenario is Gaussian-only: the ADPRR three-way "
                "partition draws the adjusting block from its Gaussian "
                "conditional law and the plausibility statistic's chi-squared "
                "reference assumes Gaussian shocks. Under "
                f"{type(self.error_dist).__name__} errors the conditional law "
                "has updated degrees of freedom and a Mahalanobis-inflated "
                "scale, and the plausibility reference becomes an F / "
                "Hotelling statistic, so both would be wrong. Use "
                "counterfactual() for in-sample shock edits (exactly valid "
                "under any error law), or refit with error_dist='gaussian'."
            )
        if path_uncertainty not in ("none", "unconditional"):
            raise ValueError(f"path_uncertainty must be 'none' or 'unconditional', got {path_uncertainty!r}")
        posterior = self.idata.posterior
        if self.data.exog is not None and "B_exog" not in posterior:
            raise ValueError(
                "This IdentifiedVAR's data carries exogenous regressors the estimator "
                "never consumed (no B_exog in the posterior); refit with an estimator "
                "that supports them before scenario analysis."
            )
        if exog_future is not None:
            if "B_exog" not in posterior:
                raise ValueError("exog_future provided but the posterior carries no B_exog.")
            exog_future = np.asarray(exog_future, dtype=float)
            n_exog = posterior["B_exog"].shape[-1]
            if exog_future.shape != (steps, n_exog):
                raise ValueError(f"exog_future must have shape ({steps}, {n_exog}), got {exog_future.shape}.")
        if self.data.exog is not None and exog_future is None:
            raise ValueError("exog_future is required when the model includes exogenous variables")

        paths, q, q_cond, q_cal, r = structural_scenario_engine(
            self,
            steps=steps,
            conditions=list(conditions or []),
            shocks=list(shocks or []),
            adjusting=adjusting,
            include_shock_uncertainty=include_shock_uncertainty,
            seed=seed,
            exog_future=exog_future,
            path_uncertainty=path_uncertainty,
        )

        forecast_da = xr.DataArray(
            paths,
            dims=["chain", "draw", "step", "variable"],
            coords={"variable": self.var_names},
            name="forecast",
        )
        ds = xr.Dataset({
            "forecast": forecast_da,
            "plausibility": xr.DataArray(q, dims=["chain", "draw"], name="plausibility"),
            "plausibility_calibrated": xr.DataArray(q_cal, dims=["chain", "draw"], name="plausibility_calibrated"),
        })
        ds.attrs["n_restrictions"] = r
        # The chi^2_r reference applies to the condition-only part of q;
        # the prescribed |v_S|^2 term carries no chi-squared law.
        ds.attrs["chi2_tail_of_median"] = float(chi2.sf(float(np.median(q_cond)), df=r)) if r else 1.0
        return ScenarioResult(
            idata=az.InferenceData(posterior_predictive=ds),
            steps=steps,
            var_names=self.var_names,
            mode="density" if include_shock_uncertainty else "mean",
            path_uncertainty=path_uncertainty,
            conditions=list(conditions or []),
            adjusting=adjusting if adjusting is None else list(adjusting),
            shocks=list(shocks or []),
        )
