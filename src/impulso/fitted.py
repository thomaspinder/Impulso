"""FittedVAR — reduced-form posterior from Bayesian VAR estimation."""

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr
from pydantic import Field

from impulso._arviz_compat import InferenceDataLike, make_idata
from impulso._base import ImpulsoBaseModel
from impulso._linalg import lag_matrices, sigma_from_cholesky
from impulso._ma import compute_ma_phi
from impulso._posterior import (
    COEFFICIENTS,
    EXOG_COEFFICIENTS,
    coefficient_draws,
    has_exog_block,
    intercept_draws,
    posterior_dataset,
)
from impulso.data import VARData
from impulso.evidence import ModelEvidence
from impulso.observation import Gaussian
from impulso.protocols import ErrorDistribution, IdentificationScheme, VolatilityProcess

if TYPE_CHECKING:
    from impulso.identified import IdentifiedVAR
    from impulso.results import (
        ConditionalForecastResult,
        DynamicMultiplierResult,
        ForecastResult,
        GrangerCausalityResult,
    )
    from impulso.scenario import VariablePath


class FittedVAR(ImpulsoBaseModel):
    """Immutable container for a fitted (reduced-form) Bayesian VAR.

    Attributes:
        idata: InferenceData-schema container with posterior draws
            (`arviz.InferenceData` on ArviZ 0, `xarray.DataTree` on ArviZ 1).
            Mutating it follows the installed upstream API.
        n_lags: Lag order used in estimation.
        data: Original VARData used for fitting.
        var_names: Names of endogenous variables.
        volatility: Volatility process used at fit time. Required;
            populated by VAR.fit from VAR.volatility (default at the
            spec level is "constant", which resolves to Constant()).
        error_dist: Observation error distribution used at fit time.
            Defaults to `Gaussian()`, which is what every estimator that
            predates the seam (notably `ConjugateVAR`) produces. `VAR.fit`
            populates it from `VAR.error_dist`. It drives the forecast
            innovation law and `innovation_covariance`, so a Student-t fit
            whose `error_dist` was lost would silently forecast Gaussian
            tails.
        pymc_model: The `pymc.Model` built during estimation, or None when
            the estimator never constructs one (`ConjugateVAR` draws in
            closed form and builds no PyMC graph). `VAR.fit` populates it so
            callers can inspect the graph (`pm.model_to_graphviz`), draw
            prior/posterior predictive samples, compute log-likelihoods, or
            apply `pm.do` / `pm.observe` transformations without refitting.
            The design matrices are baked into the graph as constants, so
            the model cannot be re-conditioned on new data. Typed as Any so
            that importing `impulso.fitted` does not pull in PyMC (see the
            lazy-import convention).
        evidence: The closed-form log marginal likelihood of the fitted model
            plus the metadata needed to compare it, or None when the estimator
            has no closed-form evidence (the PyMC/NUTS `VAR` path). Populated
            by `ConjugateVAR.fit`; pass fits carrying it to
            `impulso.compare_evidence` for Bayes factors.
    """

    idata: InferenceDataLike = Field(repr=False)
    n_lags: int
    data: VARData
    var_names: list[str]
    volatility: VolatilityProcess
    # The suppression below is a ty limitation, not a design smell: concrete
    # adapters declare `name: Literal["gaussian"]` per the discriminator
    # convention, and ty treats the Protocol's `name: str` as invariant, so
    # no adapter is assignable to its own Protocol. The same holds for
    # `Constant` vs `VolatilityProcess`; runtime `isinstance` succeeds.
    error_dist: ErrorDistribution = Field(default_factory=Gaussian)  # ty: ignore[invalid-assignment]
    pymc_model: Any = Field(default=None, repr=False)
    evidence: ModelEvidence | None = Field(default=None, repr=False)

    def _posterior(self) -> xr.Dataset:
        """The `posterior` group as an `xarray.Dataset`; see `impulso._posterior.posterior_dataset`.

        A fresh Dataset is built on each call, so methods that use the
        posterior more than once — or hand it to a memoising helper — must
        bind this to a local first.
        """
        return posterior_dataset(self.idata)

    @property
    def has_exog(self) -> bool:
        """Whether the model includes exogenous variables."""
        return self.data.exog is not None

    @property
    def coefficients(self) -> np.ndarray:
        """Posterior draws of B coefficient matrices."""
        return coefficient_draws(self._posterior())

    @property
    def intercepts(self) -> np.ndarray:
        """Posterior draws of intercept vectors."""
        return intercept_draws(self._posterior())

    def sigma(self) -> np.ndarray:
        """Posterior draws of the structural-shock scale matrix Σ.

        Σ = L Lᵀ, straight from the volatility process. Under the default
        Gaussian errors this is the innovation covariance. Under Student-t
        errors it is the **scale** matrix only — the covariance is
        `nu/(nu-2)·Σ`, available from `innovation_covariance`. Identification
        factorises the scale matrix in both cases, which is why IRFs and
        FEVD keep reading `sigma()` (see ADR-0007).

        Dispatches to the configured volatility adapter so the returned
        shape depends on whether Σ is time-invariant or time-varying:

        * Constant volatility — Σ is shared across time, so the result
          has shape `(chains, draws, n_vars, n_vars)`.
        * Stochastic volatility — Σ_t evolves, so the result has shape
          `(chains, draws, T, n_vars, n_vars)` where `T` is the
          in-sample length after lag trimming. Callers needing a single
          slice should call `volatility.cholesky_at(posterior, t)` and
          square the factor themselves.

        Note:
            **Breaking change vs. v0.0.4 and earlier**: `sigma` is now
            a method, not a property. Call sites that used `fitted.sigma`
            must be updated to `fitted.sigma()`.

        Returns:
            Posterior draws of Σ (or Σ_t for SV) computed from the
            volatility adapter's Cholesky factor as `L @ L.T`.
        """
        posterior = self._posterior()
        if self.volatility.is_time_varying:
            T = self.data.endog.shape[0] - self.n_lags
            L_path = self.volatility.cholesky_path(posterior, T=T)
            return sigma_from_cholesky(L_path)
        L = self.volatility.cholesky_at(posterior, t=None)
        return sigma_from_cholesky(L)

    def innovation_covariance(self) -> np.ndarray:
        """Posterior draws of the reduced-form innovation covariance.

        The second moment of the observation error, as opposed to `sigma()`,
        which returns the scale matrix Σ = L Lᵀ the volatility process
        builds. The two differ only under heavy-tailed errors:

        * Gaussian — returns `sigma()` unchanged.
        * Student-t — returns `nu/(nu-2)·Σ`, finite because nu > 2 is enforced
          at both the fixed and the inferred parameterisation. nu is read per
          posterior draw, so the inflation varies draw by draw when the
          degrees of freedom are inferred.

        Reach for this when the number has to be a variance: reporting
        innovation standard deviations, comparing against an OLS residual
        covariance, or sizing a shock in unconditional-standard-deviation
        units. Reach for `sigma()` when the number feeds identification.

        Returns:
            Draws of shape `(chains, draws, n_vars, n_vars)` under constant
            volatility, or `(chains, draws, T, n_vars, n_vars)` under
            stochastic volatility — the shape of `sigma()`, unchanged.
        """
        sigma = self.sigma()
        inflation = self.error_dist.variance_inflation(self._posterior())
        if np.isscalar(inflation):
            return sigma * inflation
        # (C, D) -> broadcast over the trailing matrix (and time) axes.
        inflation = np.asarray(inflation)
        extra_dims = sigma.ndim - inflation.ndim
        return sigma * inflation.reshape(inflation.shape + (1,) * extra_dims)

    def posterior_predictive(
        self,
        *,
        simulate_innovations: bool = True,
        seed: int | np.random.Generator | None = None,
    ) -> InferenceDataLike:
        """Replicate the estimation sample from the posterior.

        For every posterior draw and every in-sample date `t`,

            y_rep[t] = intercept + B x_t^obs (+ B_exog z_t) + L_t eps_t,
            eps_t ~ N(0, I)

        where `x_t^obs` stacks the **observed** lags. The replicates are
        therefore *one-step-ahead conditioned on the observed history*, the
        same object `pymc.sample_posterior_predictive` returns on the fitted
        graph and the one `arviz.plot_ppc` expects — not an iterated
        simulated path from initial conditions, which is `forecast`'s job.

        `L_t` comes from the volatility seam
        (`volatility.cholesky_path(posterior, T=T)`), so the innovations use
        the *model's own* Σ: a single Σ under constant volatility, and a
        genuinely per-draw, per-`t` Σ_t under stochastic volatility (an
        empirical residual covariance would flatten exactly the
        heteroscedasticity an SV fit exists to capture).

        With `simulate_innovations=False` the conditional means come back
        unperturbed — the in-sample fit under parameter uncertainty, useful
        for residual diagnostics (`observed - mean` is exactly the
        reduced-form residual). Mean mode consumes no randomness at all, so
        `seed` is irrelevant there.

        Computed in NumPy rather than on a PyMC graph, so it works for
        `ConjugateVAR`-derived posteriors too (no graph exists there); see
        ADR-0011.

        Note:
            `self.idata` is never mutated — a fresh container comes back.
            To attach the result to the fit, merge it yourself using the
            installed ArviZ line's own API:

                ppc = fitted.posterior_predictive(seed=0)
                fitted.idata.extend(ppc)   # ArviZ 0 (arviz.InferenceData)
                fitted.idata.update(ppc)   # ArviZ 1 (xarray.DataTree)

            The two differ in conflict precedence: `extend` keeps the
            existing group unless `join="right"`, while `update` overwrites
            it. Drop the conflicting group first if that matters.

        Note:
            The replicate array is dense: `chains * draws * T * n_vars`
            float64 values, roughly 19 MB at 4 chains, 1000 draws, 200
            dates and 3 variables. Thin the posterior before calling if
            that is too large.

        Args:
            simulate_innovations: If `True` (default), add shock
                innovations drawn from the volatility process's in-sample
                Cholesky path — a true posterior-predictive density. If
                `False`, return the conditional mean only.
            seed: RNG seed (int) or Generator for reproducible replicates.
                Ignored when `simulate_innovations=False`.

        Returns:
            InferenceData-schema container with a `posterior_predictive`
            group holding `obs` with dims `(chain, draw, time, var)` and an
            `observed_data` group holding the realised `obs` with dims
            `(time, var)`. Both are time-aligned with
            `data.index[n_lags:]`.

        Raises:
            ValueError: If the data carries exogenous regressors the
                estimator never consumed (no `B_exog` in the posterior).
        """
        from impulso._residuals import fitted_values

        posterior = self._posterior()
        if self.data.exog is not None and not has_exog_block(posterior):
            raise ValueError(
                "This FittedVAR's data carries exogenous regressors the estimator "
                "never consumed (no B_exog in the posterior); refit with an "
                "estimator that supports them before replicating the sample."
            )

        mu = fitted_values(posterior, self.data, self.n_lags)  # (C, D, T, n)
        y_rep = mu
        if simulate_innovations:
            rng = np.random.default_rng(seed) if not isinstance(seed, np.random.Generator) else seed
            L_path = self.volatility.cholesky_path(posterior, T=mu.shape[2])  # (C, D, T, n, n)
            eps = rng.standard_normal(mu.shape)
            y_rep = mu + np.einsum("cdtij,cdtj->cdti", L_path, eps)

        # Explicit-dim tuple form, as in `identified.py`: a `from_df` index may
        # carry a name (e.g. "date") that xarray would otherwise adopt as the
        # coordinate's dimension and reject against the explicit "time" dim.
        time = ("time", self.data.index[self.n_lags :])
        coords = {"time": time, "var": self.var_names}
        return make_idata(
            posterior_predictive=xr.Dataset({
                "obs": xr.DataArray(y_rep, dims=["chain", "draw", "time", "var"], coords=coords, name="obs")
            }),
            observed_data=xr.Dataset({
                "obs": xr.DataArray(self.data.endog[self.n_lags :], dims=["time", "var"], coords=coords, name="obs")
            }),
        )

    def forecast(
        self,
        steps: int,
        include_shock_uncertainty: bool = True,
        seed: int | np.random.Generator | None = None,
        exog_future: np.ndarray | None = None,
    ) -> "ForecastResult":
        """Produce h-step-ahead forecasts from the reduced-form posterior.

        Args:
            steps: Number of forecast steps.
            include_shock_uncertainty: If ``True`` (default), draw shock
                innovations from the volatility process's forecast Cholesky
                path, giving true posterior-predictive intervals (density
                forecast).  If ``False``, propagate only the conditional
                mean — intervals reflect parameter uncertainty only (mean
                forecast).
            seed: RNG seed (int) or Generator for reproducible density
                forecasts.  Ignored when ``include_shock_uncertainty=False``.
            exog_future: Future exogenous values, shape ``(steps, k)``.
                Required if model has exog.

        Returns:
            ForecastResult with posterior forecast draws.

        Note:
            The innovation law comes from `error_dist`. Under Student-t
            errors the per-step innovation is `L_h ξ` with
            `ξ = z / sqrt(g/nu)`, `z ~ N(0, I)` and one `g ~ χ²_nu` **per
            forecast draw and step** — the same shared-mixing construction
            PyMC's own `MvStudentT` sampler uses, so the density forecast
            matches the fitted likelihood rather than approximating it.
            Mean mode consumes no randomness and is therefore identical
            across error distributions.
        """
        from impulso._propagate import propagate
        from impulso._scenario import deterministic_forecast_path, resolve_exog_future
        from impulso.results import ForecastResult

        posterior = self._posterior()
        exog_arr = resolve_exog_future(posterior, self.data, steps, exog_future)
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

        # Density mode: get the forecast Cholesky path FIRST, then the
        # per-step innovations — the seam's RNG stream order (ADR-0007).
        L_path = None
        if include_shock_uncertainty:
            L_path = self.volatility.forecast_cholesky_path(posterior, steps=steps, rng=rng)  # (C, D, steps, n, n)

        forecasts = deterministic_forecast_path(posterior, self.data, self.n_lags, steps, exog_arr)

        if L_path is not None:
            n_chains, n_draws, _, n_vars = forecasts.shape
            # Innovations from the error-distribution seam (standardised:
            # identity scale under Gaussian, nu/(nu-2)·I under Student-t),
            # drawn per step in stream order, then propagated through the
            # shared lag recursion as a deviation from the baseline.
            eps = np.empty((n_chains, n_draws, steps, n_vars))
            for h in range(steps):
                eps[:, :, h, :] = self.error_dist.draw_standardised_innovations(
                    (n_chains, n_draws, n_vars), rng, posterior
                )
            u = np.einsum("cdhij,cdhj->cdhi", L_path, eps)
            A = lag_matrices(coefficient_draws(posterior), self.n_lags)
            forecasts = forecasts + propagate(A, u, np.zeros((self.n_lags, n_vars)))

        mode = "density" if include_shock_uncertainty else "mean"
        forecast_da = xr.DataArray(
            forecasts,
            dims=["chain", "draw", "step", "variable"],
            coords={"variable": self.var_names},
            name="forecast",
        )
        idata = make_idata(posterior_predictive=xr.Dataset({"forecast": forecast_da}))
        return ForecastResult(idata=idata, steps=steps, var_names=self.var_names, mode=mode)

    def conditional_forecast(
        self,
        steps: int,
        conditions: "list[VariablePath] | None" = None,
        include_shock_uncertainty: bool = True,
        seed: int | np.random.Generator | None = None,
        exog_future: np.ndarray | None = None,
        path_uncertainty: Literal["none", "unconditional"] = "none",
    ) -> "ConditionalForecastResult":
        """Forecast constrained so chosen variables follow pinned future paths.

        Hard conditional forecasting in the sense of Waggoner and Zha
        (1999): all structural shocks adjust to absorb the pins, and the
        observable-space answer is invariant to the identification scheme
        — no scheme is required, which is why this lives on the
        reduced-form object (the dynamic-multiplier placement logic).
        Under the default mode every draw satisfies the pins *pathwise*;
        with `path_uncertainty="unconditional"` (Antolín-Díaz, Petrella &
        Rubio-Ramírez 2021) the pins restrict the forecast mean only and
        the bands keep their unconditional width — the mode behind that
        paper's headline plausibility numbers.

        Each `VariablePath` pins values from step 1: a scalar broadcasts
        to all steps, an array of length `L <= steps` pins steps `1..L`,
        and `NaN` entries are free. With no conditions the result is
        distributionally `forecast()` — and exactly equal per draw under a
        matched `seed`, because the engine consumes the generator in
        `forecast()`'s order (one forecast-Cholesky-path call, then
        per-step innovation draws).

        The result carries the per-draw plausibility statistic
        `q = c̄'(CC')⁻¹c̄` (squared Mahalanobis distance of the pinned
        values from their unconditional law; `chi^2_r` reference) and its
        ADPRR-calibrated companion `q_cal ∈ [0.5, 1]` — large values mean
        the scenario demands incredible shocks and the model's answer
        should not be trusted. `q_cal` is finite only under
        `path_uncertainty="unconditional"` (where ADPRR's divergence
        collapses to `z = q/2`); under hard pins the underlying
        divergence is infinite and `q_cal` sits at its ceiling of 1
        (floor 0.5 with no conditions).

        Note:
            Under time-varying volatility the conditioning is per
            simulated volatility path — the conditions never reweight the
            volatility-path law, so the result is conditional-on-path
            rather than the full Bayesian conditional (standard practice;
            see ADR-0005). Consequently, mean mode with pins under a
            stochastic-volatility adapter conditions on one simulated
            path per draw and therefore depends on `seed`; with no
            conditions, mean mode consumes no randomness at all.

        Note:
            Gaussian errors only. The Waggoner-Zha constrained draw *is*
            the Gaussian conditional-law formula, and the plausibility
            statistic's `chi^2_r` reference assumes Gaussian shocks;
            neither survives a heavy-tailed error law. Under
            `error_dist="student_t"` this method raises
            `NotImplementedError` rather than returning a half-valid
            answer.

        Args:
            steps: Number of forecast steps.
            conditions: `VariablePath` pins (may be empty or omitted).
            include_shock_uncertainty: If `True` (default), draw the
                unconstrained shock dimensions (density forecast). If
                `False`, propagate the conditional mean — pins still hold,
                intervals reflect parameter uncertainty only.
            seed: RNG seed (int) or Generator for reproducible density
                forecasts.
            exog_future: Future exogenous values, shape `(steps, k)`.
                Required if the posterior carries `B_exog`.
            path_uncertainty: `"none"` (default — hard pins, variance
                collapses at the pins) or `"unconditional"` (pins
                restrict the mean; bands keep unconditional width).

        Returns:
            ConditionalForecastResult with forecast draws, the pinned
            conditions echoed, and the plausibility statistics.

        Raises:
            NotImplementedError: If the model was fitted with a
                heavy-tailed error distribution.
            ValueError: On unknown variables, duplicate or over-length
                pins, an invalid `path_uncertainty`, exogenous data the
                estimator never consumed, or a mis-shaped `exog_future`.
        """
        from scipy.stats import chi2

        from impulso._scenario import conditional_forecast_engine, resolve_exog_future
        from impulso.results import ConditionalForecastResult

        if self.error_dist.is_heavy_tailed:
            raise NotImplementedError(
                "conditional_forecast is Gaussian-only: the Waggoner-Zha "
                "constrained draw is the Gaussian conditional-law formula and "
                "the plausibility statistic's chi-squared reference assumes "
                "Gaussian shocks. Under "
                f"{type(self.error_dist).__name__} errors the conditional law "
                "has updated degrees of freedom and a Mahalanobis-inflated "
                "scale, so both would be wrong. Use forecast() for "
                "unconditional density forecasts, or refit with "
                "error_dist='gaussian'."
            )
        if path_uncertainty not in ("none", "unconditional"):
            raise ValueError(f"path_uncertainty must be 'none' or 'unconditional', got {path_uncertainty!r}")
        posterior = self._posterior()
        exog_future = resolve_exog_future(posterior, self.data, steps, exog_future)

        paths, q, q_cal, r = conditional_forecast_engine(
            self,
            steps=steps,
            conditions=list(conditions or []),
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
        ds.attrs["chi2_tail_of_median"] = float(chi2.sf(float(np.median(q)), df=r)) if r else 1.0
        return ConditionalForecastResult(
            idata=make_idata(posterior_predictive=ds),
            steps=steps,
            var_names=self.var_names,
            mode="density" if include_shock_uncertainty else "mean",
            path_uncertainty=path_uncertainty,
            conditions=list(conditions or []),
        )

    def dynamic_multiplier(self, horizon: int = 20, cumulative: bool = False) -> "DynamicMultiplierResult":
        """Response of the endogenous variables to a unit exogenous impulse.

        The exogenous term enters the VAR contemporaneously
        (`mu = intercept + X_lag @ B.T + X_exog @ B_exog.T`), so it acts as a
        forcing term in the same position as the reduced-form shock. The
        horizon-`h` dynamic multiplier is therefore

            Psi_h = Phi_h @ B_exog

        where `Phi_h` is the moving-average coefficient matrix already used by
        `IdentifiedVAR.impulse_response`. All dynamics come from the
        endogenous lag structure; `B_exog` itself carries no lags.

        No structural identification is involved: exogenous regressors are
        exogenous by assumption, so this lives on the reduced-form posterior
        and needs neither an `IdentificationScheme` nor an `at=` time slice
        (`B` and `B_exog` are time-invariant under every volatility process).

        Draws are read from the posterior as `(chain, draw, var, coeff)` and
        `(chain, draw, var, exog)`. Hand-built posteriors that use those
        dimension names are realigned automatically; posteriors without them
        are trusted positionally in that order.

        Args:
            horizon: Highest horizon to compute. The result spans horizon
                `0` through `horizon` inclusive. Must be non-negative.
            cumulative: If True, return the cumulative (step-response)
                multiplier — the response to a permanent unit step in the
                exogenous variable from time 0 onward. If False (default),
                return the per-horizon response to a one-off unit impulse.

        Returns:
            DynamicMultiplierResult with draws of shape
            `(chains, draws, horizon + 1, n_vars, n_exog)`.

        Raises:
            ValueError: If `horizon` is negative, if the fitted posterior
                carries no `B_exog` (the model was fitted without exogenous
                regressors, or by an estimator that ignores them), or if
                `data.exog_names` disagrees with the posterior's `B_exog`
                column count.
        """
        from impulso.results import DynamicMultiplierResult

        if horizon < 0:
            raise ValueError(f"horizon must be non-negative, got {horizon}")
        posterior = self._posterior()
        # Guard on the posterior, not on `has_exog`: an estimator may carry
        # exogenous data it never actually consumed.
        if not has_exog_block(posterior):
            raise ValueError(
                "This FittedVAR has no B_exog in its posterior, so no dynamic "
                "multiplier is defined. Fit a VAR with exogenous regressors "
                "(VARData(..., exog=...)) using an estimator that supports them."
            )

        B_da = posterior[COEFFICIENTS]
        B_exog_da = posterior[EXOG_COEFFICIENTS]
        # Hand-built posteriors may order dims arbitrarily. Realign by name
        # when the canonical labels are present; otherwise trust the
        # positional (chain, draw, var, coeff/exog) convention.
        if set(B_da.dims) == {"chain", "draw", "var", "coeff"}:
            B_da = B_da.transpose("chain", "draw", "var", "coeff")
        if set(B_exog_da.dims) == {"chain", "draw", "var", "exog"}:
            B_exog_da = B_exog_da.transpose("chain", "draw", "var", "exog")
        B_draws = B_da.values  # (C, D, n, n*p)
        B_exog_draws = B_exog_da.values  # (C, D, n, k)

        n_exog = B_exog_draws.shape[-1]
        exog_names = self.data.exog_names or [f"exog_{i}" for i in range(n_exog)]
        if len(exog_names) != n_exog:
            raise ValueError(
                f"data.exog_names carries {len(exog_names)} names but the "
                f"posterior's B_exog has {n_exog} exogenous columns; this "
                "FittedVAR's data and posterior disagree."
            )

        Phi = compute_ma_phi(lag_matrices(B_draws, self.n_lags), horizon)  # (C, D, H+1, n, n)
        psi = Phi @ B_exog_draws[:, :, np.newaxis, :, :]  # (C, D, H+1, n, k)
        if cumulative:
            psi = np.cumsum(psi, axis=2)

        psi_da = xr.DataArray(
            psi,
            dims=["chain", "draw", "horizon", "response", "exog"],
            coords={
                "response": self.var_names,
                "exog": exog_names,
                "horizon": np.arange(horizon + 1),
            },
            name="dynamic_multiplier",
        )
        idata = make_idata(posterior_predictive=xr.Dataset({"dynamic_multiplier": psi_da}))
        return DynamicMultiplierResult(
            idata=idata,
            horizon=horizon,
            var_names=self.var_names,
            exog_names=list(exog_names),
            cumulative=cumulative,
        )

    def granger_causality(
        self,
        cause: str,
        effect: str,
        *,
        rope: float | None = None,
        standardize: bool = True,
        test_lags: int | None = None,
    ) -> "GrangerCausalityResult":
        """Posterior strength of one variable's lags in another's equation.

        Reports the posterior of `‖b‖`, the Euclidean norm of the tested
        lag coefficients of `cause` in the `effect` equation, together with
        the per-lag posteriors behind it. A magnitude, not a test
        statistic: nothing here divides by the posterior covariance, so a
        small effect and an imprecise one stay distinguishable.

        Supply a `rope` — a region of practical equivalence, in the
        reporting units — to also get `p_rope = P(‖b‖ < rope | data)`.
        That is a statement about practical negligibility, **not** the
        probability of no causality: under continuous coefficient priors
        `b = 0` has probability zero regardless of the data. See
        `GrangerCausalityResult` for the full statement.

        Granger causality is conditional predictive precedence within this
        set of variables, not intervention. An omitted common driver is
        enough to manufacture it.

        No identification scheme is involved — this reads the reduced-form
        coefficients directly — and it works under any volatility process,
        because `B` is time-invariant under all of them.

        Args:
            cause: Variable whose lags are tested.
            effect: Variable whose equation they are tested in. Must differ
                from `cause`.
            rope: Region of practical equivalence for `p_rope`. Must be
                positive when given; there is deliberately no default.
            standardize: If True (default), scale the draws by
                `sd(cause) / sd(effect)` so magnitudes read as standard
                deviations of the effect per standard deviation of the
                cause. The factor is recorded on the result as `scale`.
            test_lags: Number of lags to test, counting from lag 1.
                Defaults to every fitted lag. Pass `p` after fitting `p + d`
                lags to run the Toda-Yamamoto test by hand; the untested
                lags are recorded as `augmentation`, never dropped from the
                fit.

        Returns:
            GrangerCausalityResult for the ordered `cause -> effect` pair.

        Raises:
            ValueError: If either name is unknown, if `cause == effect`, if
                `test_lags` is outside `[1, n_lags]`, or if `rope` is not
                positive.
        """
        from impulso._granger import granger_causality

        return granger_causality(
            self,
            cause,
            effect,
            rope=rope,
            standardize=standardize,
            test_lags=test_lags,
        )

    def set_identification_strategy(self, scheme: IdentificationScheme) -> "IdentifiedVAR":
        """Apply a structural identification scheme.

        The structural shock matrix is no longer eagerly computed or stored
        in the posterior.  Instead, :meth:`IdentifiedVAR.shock_matrix` lazily
        queries and memoises it on first access.  This method constructs
        the ``IdentifiedVAR`` through normal Pydantic validation.

        Args:
            scheme: An IdentificationScheme protocol instance (e.g. Cholesky,
                SignRestriction).

        Returns:
            IdentifiedVAR ready for structural queries.
        """
        from impulso.identified import IdentifiedVAR

        return IdentifiedVAR(
            idata=self.idata,
            n_lags=self.n_lags,
            data=self.data,
            var_names=self.var_names,
            volatility=self.volatility,
            error_dist=self.error_dist,
            scheme=scheme,
        )
