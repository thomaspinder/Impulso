"""Tests for VAR specification."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from impulso._arviz_compat import make_idata
from impulso.data import VARData
from impulso.priors import MinnesotaPrior
from impulso.spec import VAR, _exog_prior_sigma
from impulso.volatility import Constant


def _exog_data(endog: np.ndarray, exog: np.ndarray, exog_names: list[str]) -> VARData:
    """VARData wrapper for the exog-prior tests."""
    return VARData(
        endog=endog,
        endog_names=[f"y{i + 1}" for i in range(endog.shape[1])],
        exog=exog,
        exog_names=exog_names,
        index=pd.date_range("2000-01-01", periods=endog.shape[0], freq="QS"),
    )


def _captured_sigma(model, name: str) -> np.ndarray:
    """Evaluate the sigma tensor a `pm.Normal` RV was built with.

    `dist_params` maps the RV's op back onto its distribution parameters, so
    this survives the positional reshuffles PyMC makes to `owner.inputs`.
    """
    rv = model[name]
    _mu, sigma = rv.owner.op.dist_params(rv.owner)
    return np.asarray(sigma.eval())


class TestVARSpec:
    def test_fixed_lags(self):
        spec = VAR(lags=4, prior="minnesota")
        assert spec.lags == 4
        assert spec.max_lags is None

    @pytest.mark.parametrize("criterion", ["aic", "bic", "hq"])
    def test_string_lags(self, criterion):
        spec = VAR(lags=criterion, prior="minnesota")
        assert spec.lags == criterion

    def test_string_lags_with_max(self):
        spec = VAR(lags="aic", max_lags=12, prior="minnesota")
        assert spec.max_lags == 12

    def test_rejects_max_lags_with_fixed(self):
        with pytest.raises(ValueError, match="max_lags is only valid"):
            VAR(lags=4, max_lags=12, prior="minnesota")

    def test_prior_string_resolves(self):
        spec = VAR(lags=2, prior="minnesota")
        assert isinstance(spec.resolved_prior, MinnesotaPrior)

    def test_prior_object(self):
        prior = MinnesotaPrior(tightness=0.2)
        spec = VAR(lags=2, prior=prior)
        assert spec.resolved_prior.tightness == 0.2

    def test_frozen(self):
        spec = VAR(lags=2, prior="minnesota")
        with pytest.raises(ValidationError):
            spec.lags = 3

    def test_rejects_zero_lags(self):
        with pytest.raises(ValidationError):
            VAR(lags=0, prior="minnesota")

    def test_default_sampler_is_safe(self):
        """The sampler `VAR.fit` falls back to must pin cores=1 to dodge the segfault."""
        from impulso.samplers import NUTSSampler

        sampler = VAR._default_sampler()
        assert isinstance(sampler, NUTSSampler)
        assert sampler.cores == 1
        assert sampler.chains == 4
        assert sampler.target_accept == 0.8


class TestVolatilityParameter:
    def test_default_volatility_is_constant_string(self):
        spec = VAR(lags=2)
        assert spec.volatility == "constant"

    def test_volatility_string_resolves_to_constant(self):
        spec = VAR(lags=2)
        assert isinstance(spec.resolved_volatility, Constant)

    def test_volatility_object_pass_through(self):
        adapter = Constant(sigma_sd_beta=3.0)
        spec = VAR(lags=2, volatility=adapter)
        assert spec.resolved_volatility is adapter
        assert spec.resolved_volatility.sigma_sd_beta == 3.0

    def test_unknown_string_raises(self):
        with pytest.raises(ValidationError):
            VAR(lags=2, volatility="nonexistent")


class TestPyMCModelBuild:
    """Verify the PyMC model graph composition after the seam refactor."""

    def test_model_has_expected_unobserved_rvs(self, var_data_2v):
        """The same set of RVs must exist in the PyMC graph as before the seam."""
        # We don't run MCMC here; we just build the model and inspect.
        # Rebuild logic mirrors VAR.fit but stops before sampler.sample().
        import pymc as pm

        from impulso._lag_selection import select_lag_order  # noqa: F401 — import-side-effect parity

        spec = VAR(lags=1)
        prior_params = spec.resolved_prior.build_priors(n_vars=2, n_lags=1)
        volatility = spec.resolved_volatility

        y = var_data_2v.endog
        Y = y[1:]
        X_lag = y[:-1]

        with pm.Model() as model:
            pm.Normal("intercept", mu=0, sigma=1, shape=2)
            pm.Normal("B", mu=prior_params["B_mu"], sigma=prior_params["B_sigma"], shape=(2, 2))
            L = volatility.build_pymc_latent(n_vars=2, T=Y.shape[0])
            pm.Deterministic("Sigma", pm.math.dot(L, L.T))
            mu = pm.math.dot(X_lag, model.named_vars["B"].T)
            pm.MvNormal("obs", mu=mu, chol=L, observed=Y)

        rv_names = {v.name for v in model.unobserved_RVs}
        det_names = {v.name for v in model.deterministics}
        assert "intercept" in rv_names
        assert "B" in rv_names
        assert "sigma_sd" in rv_names
        assert "tril_offdiag" in rv_names
        assert "Sigma" in det_names

    def test_var_fit_routes_through_volatility_seam(self, var_data_2v):
        """VAR.fit (intercepted before sampling) must register the canonical RV set.

        The companion test above mirrors the model graph by hand, so it would
        keep passing even if VAR.fit silently skipped the volatility delegation.
        This one captures the model from the production codepath via a sampler
        that aborts before MCMC, then asserts the graph shape.
        """
        import pymc as pm

        captured: dict[str, pm.Model] = {}

        class CapturingSampler:
            name = "capture"

            def sample(self, model: pm.Model):
                captured["model"] = model
                raise RuntimeError("stop before sampling")

        spec = VAR(lags=1)
        with pytest.raises(RuntimeError, match="stop before sampling"):
            spec.fit(var_data_2v, sampler=CapturingSampler())

        model = captured["model"]
        rv_names = {v.name for v in model.unobserved_RVs}
        det_names = {v.name for v in model.deterministics}
        assert {"intercept", "B", "sigma_sd", "tril_offdiag"} <= rv_names
        assert "Sigma" in det_names
        assert {v.name for v in model.observed_RVs} == {"obs"}


class TestVarFitWithSV:
    def test_var_fit_with_sv_builds_3d_chol(self, var_data_2v):
        """VAR(volatility=StochasticVolatility(...)).fit(...) builds a model
        whose obs likelihood uses a 3D chol factor (per-t)."""
        import pymc as pm

        from impulso.spec import VAR
        from impulso.sv.spec import StochasticVolatility

        captured = {}

        class CapturingSampler:
            name = "capture"

            def sample(self, model: pm.Model):
                captured["model"] = model
                raise RuntimeError("stop before sampling")

        with pytest.raises(RuntimeError, match="stop before sampling"):
            VAR(lags=1, volatility=StochasticVolatility()).fit(var_data_2v, sampler=CapturingSampler())

        model = captured["model"]
        rv_names = {v.name for v in model.unobserved_RVs}
        det_names = {v.name for v in model.deterministics}

        # Per-variable log-vol paths registered.
        assert "v0_mu" in rv_names and "v1_mu" in rv_names
        assert "v0_sigma_eta" in rv_names and "v1_sigma_eta" in rv_names
        # Shared correlation factor.
        assert "R_chol" in det_names or "R_chol" in rv_names
        # h is the stacked log-vol path deterministic.
        assert "h" in det_names
        # No Sigma deterministic for SV (skipped — too memory-heavy).
        assert "Sigma" not in det_names


class TestVarPassesResidualsToVolatility:
    """VAR computes OLS residuals once and passes them as `data` to
    volatility.build_pymc_latent — the multivariate SV adapter relies on
    this to seed per-variable priors (closes #65)."""

    def test_var_threads_ols_residuals_through_volatility_adapter(self, var_data_2v):
        import numpy as np
        import pymc as pm
        import pytensor.tensor as pt

        from impulso.protocols import VolatilityProcess

        captured = {}

        class CapturingVolatility:
            """VolatilityProcess stand-in that records the `data` it receives
            and returns a trivial constant L so the model still builds."""

            name = "capturing"
            is_time_varying = False

            def build_pymc_latent(self, n_vars, T, data=None):
                captured["n_vars"] = n_vars
                captured["T"] = T
                captured["data"] = None if data is None else data.copy()
                sd = pm.HalfCauchy("capturing_sd", beta=1.0, shape=n_vars)
                L = pt.zeros((n_vars, n_vars))
                L = pt.set_subtensor(L[np.diag_indices(n_vars)], sd)
                return pm.Deterministic("L", L)

            def cholesky_at(self, posterior, t):
                return posterior["L"].values

            def cholesky_path(self, posterior, T):
                L = self.cholesky_at(posterior, t=None)
                return np.broadcast_to(L[:, :, None, :, :], (*L.shape[:2], T, *L.shape[-2:])).copy()

            def forecast_cholesky_path(self, posterior, steps, rng):
                return self.cholesky_path(posterior, steps)

        assert isinstance(CapturingVolatility(), VolatilityProcess)

        class CapturingSampler:
            name = "capture"

            def sample(self, model):
                raise RuntimeError("stop before sampling")

        with pytest.raises(RuntimeError, match="stop before sampling"):
            VAR(lags=1, volatility=CapturingVolatility()).fit(var_data_2v, sampler=CapturingSampler())

        n_lags = 1
        T_eff = var_data_2v.endog.shape[0] - n_lags
        n_vars = var_data_2v.endog.shape[1]

        assert captured["n_vars"] == n_vars
        assert captured["T"] == T_eff
        assert captured["data"] is not None, "VAR did not pass `data` to volatility"
        assert captured["data"].shape == (T_eff, n_vars)
        # OLS residuals are demeaned by construction (intercept in X_full).
        assert np.allclose(captured["data"].mean(axis=0), 0.0, atol=1e-8)


class TestExogPriorScaleField:
    def test_default_is_loose(self):
        assert VAR(lags=1).exog_prior_scale == 100.0

    def test_custom_value_round_trips(self):
        assert VAR(lags=1, exog_prior_scale=2.5).exog_prior_scale == 2.5

    @pytest.mark.parametrize("bad", [0.0, -1.0, -1e-9])
    def test_rejects_non_positive(self, bad):
        with pytest.raises(ValidationError):
            VAR(lags=1, exog_prior_scale=bad)

    def test_frozen(self):
        spec = VAR(lags=1)
        with pytest.raises(ValidationError):
            spec.exog_prior_scale = 5.0


class TestExogPriorSigma:
    """`_exog_prior_sigma` puts the B_exog prior in contribution space (#192)."""

    def test_matches_independent_formula(self, rng):
        from impulso._conjugate import ar1_residual_sd

        endog = rng.standard_normal((120, 3))
        exog = rng.standard_normal((120, 2)) * np.array([0.01, 500.0])

        got = _exog_prior_sigma(endog, exog, 100.0)
        expected = 100.0 * np.outer(ar1_residual_sd(endog), 1.0 / exog.std(axis=0, ddof=1))

        assert got.shape == (3, 2)
        np.testing.assert_allclose(got, expected)

    def test_scale_is_linear(self, rng):
        endog = rng.standard_normal((120, 2))
        exog = rng.standard_normal((120, 3))

        np.testing.assert_allclose(
            _exog_prior_sigma(endog, exog, 5.0),
            0.05 * _exog_prior_sigma(endog, exog, 100.0),
        )

    def test_tiny_scale_regressor_gets_a_huge_prior_sd(self, rng):
        """The bug: sd 0.01 regressor needs a prior ~100x looser than sd 1.0."""
        endog = rng.standard_normal((120, 2))
        tiny = rng.standard_normal((120, 1)) * 0.01
        ordinary = tiny / 0.01

        ratio = _exog_prior_sigma(endog, tiny, 100.0) / _exog_prior_sigma(endog, ordinary, 100.0)
        np.testing.assert_allclose(ratio, 100.0)

    def test_floor_fires_on_a_numerically_flat_column(self, rng):
        """A column with negligible spread falls back to a level-derived scale."""
        from impulso._conjugate import ar1_residual_sd
        from impulso.spec import _EXOG_SD_FLOOR_FRACTION

        endog = rng.standard_normal((120, 2))
        col = (1.0 + 1e-13 * rng.standard_normal(120))[:, None]

        got = _exog_prior_sigma(endog, col, 100.0)
        floor = _EXOG_SD_FLOOR_FRACTION * np.abs(col).max()
        expected = 100.0 * np.outer(ar1_residual_sd(endog), 1.0 / np.array([floor]))

        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, expected)
        # The floor is doing real work: the raw sd would give a prior ~1e10 wider.
        unfloored = 100.0 * np.outer(ar1_residual_sd(endog), 1.0 / col.std(axis=0, ddof=1))
        assert (got < unfloored / 1e9).all()

    def test_floor_does_not_fire_on_a_step_dummy(self, rng):
        """A 0/1 break at 75% has real spread, so the raw sd is used."""
        from impulso._conjugate import ar1_residual_sd

        endog = rng.standard_normal((120, 2))
        dummy = np.zeros((120, 1))
        dummy[90:] = 1.0

        raw_sd = dummy.std(axis=0, ddof=1)
        floor = 1e-3 * np.abs(dummy).max(axis=0)
        assert raw_sd[0] > floor[0], "fixture no longer exercises the non-floored path"

        expected = 100.0 * np.outer(ar1_residual_sd(endog), 1.0 / raw_sd)
        np.testing.assert_allclose(_exog_prior_sigma(endog, dummy, 100.0), expected)

    def test_rejects_column_that_is_all_zero_after_lag_trimming(self, rng):
        endog = rng.standard_normal((120, 2))
        pulse = np.zeros((120, 1))

        with pytest.raises(ValueError, match=r"constant over the estimation sample: 'pulse'"):
            _exog_prior_sigma(endog, pulse, 100.0, ["pulse"])

    def test_rejects_column_that_is_constant_nonzero_after_lag_trimming(self, rng):
        """The floor must not rescue a column that is flat at a non-zero level.

        A 0/1 dummy that switches inside the initial conditions passes VARData's
        whole-sample check, then arrives here as a column of ones — collinear
        with the intercept. Keying the floor off its level would hand it a wide
        prior on an unidentified coefficient, which is the failure this guard
        exists to stop.
        """
        endog = rng.standard_normal((120, 2))
        flat = np.ones((120, 1))

        with pytest.raises(ValueError, match=r"constant over the estimation sample: 'early_break'") as exc:
            _exog_prior_sigma(endog, flat, 100.0, ["early_break"])
        assert "reduce `lags`" in str(exc.value)

    def test_constant_column_error_falls_back_to_column_index(self, rng):
        endog = rng.standard_normal((120, 2))
        with pytest.raises(ValueError, match=r"'column 0'"):
            _exog_prior_sigma(endog, np.zeros((120, 1)), 100.0)


class TestExogPriorWiredIntoModel:
    """VAR.fit must build B_exog with the scale-adaptive sigma, not sigma=1."""

    @staticmethod
    def _capture(data, spec):
        import pymc as pm

        captured: dict[str, pm.Model] = {}

        class CapturingSampler:
            name = "capture"

            def sample(self, model: pm.Model):
                captured["model"] = model
                raise RuntimeError("stop before sampling")

        with pytest.raises(RuntimeError, match="stop before sampling"):
            spec.fit(data, sampler=CapturingSampler())
        return captured["model"]

    def test_b_exog_sigma_matches_the_formula(self, rng):
        endog = rng.standard_normal((150, 2))
        exog = rng.standard_normal((150, 2)) * np.array([0.01, 250.0])
        data = _exog_data(endog, exog, ["tiny", "huge"])

        model = self._capture(data, VAR(lags=1))

        expected = _exog_prior_sigma(endog, exog[1:], 100.0)
        np.testing.assert_allclose(_captured_sigma(model, "B_exog"), expected)
        # The old bug: a flat unit prior regardless of the regressor's units.
        assert not np.allclose(_captured_sigma(model, "B_exog"), 1.0)

    def test_knob_reaches_the_graph(self, rng):
        endog = rng.standard_normal((150, 2))
        exog = rng.standard_normal((150, 1)) * 0.01
        data = _exog_data(endog, exog, ["tiny"])

        model = self._capture(data, VAR(lags=1, exog_prior_scale=5.0))

        np.testing.assert_allclose(
            _captured_sigma(model, "B_exog"),
            _exog_prior_sigma(endog, exog[1:], 5.0),
        )

    def test_sigma_uses_lag_trimmed_rows(self, rng):
        """The prior keys off the rows the likelihood sees, not the raw column."""
        endog = rng.standard_normal((150, 2))
        exog = rng.standard_normal((150, 1))
        exog[:4] *= 500.0  # outlying head, dropped by lags=4
        data = _exog_data(endog, exog, ["x"])

        model = self._capture(data, VAR(lags=4))

        got = _captured_sigma(model, "B_exog")
        np.testing.assert_allclose(got, _exog_prior_sigma(endog, exog[4:], 100.0))
        assert not np.allclose(got, _exog_prior_sigma(endog, exog, 100.0))

    def test_no_b_exog_without_exog(self, var_data_2v):
        model = self._capture(var_data_2v, VAR(lags=1))
        assert "B_exog" not in {v.name for v in model.unobserved_RVs}

    def test_fit_rejects_a_dummy_that_only_switches_inside_the_initial_conditions(self, rng):
        """VARData sees variation; the estimation sample does not (#192)."""
        endog = rng.standard_normal((150, 2))
        dummy = np.ones((150, 1))
        dummy[:2] = 0.0  # switches at t=2, but lags=4 trims rows 0-3 away
        data = _exog_data(endog, dummy, ["early_break"])

        with pytest.raises(ValueError, match=r"constant over the estimation sample: 'early_break'"):
            VAR(lags=4).fit(data)

        # Same column with lags=1 keeps the switch, so it is estimable.
        self._capture(data, VAR(lags=1))

    def test_prior_predictive_graph_carries_the_same_scaled_sigma(self, rng):
        """The graph `prior_predictive` draws from is the graph `fit` samples (#56, #192).

        `VAR.prior_predictive` builds its graph through `_build_pymc_model`,
        the same seam `fit` uses. If the scale-adaptive exog prior lived in
        `fit` instead, a prior-predictive check would silently describe a
        unit-scale model that was never fitted.
        """
        endog = rng.standard_normal((150, 2))
        exog = rng.standard_normal((150, 2)) * np.array([0.01, 250.0])
        data = _exog_data(endog, exog, ["tiny", "huge"])
        spec = VAR(lags=1)

        prior_predictive_model, _ = spec._build_pymc_model(data)
        got = _captured_sigma(prior_predictive_model, "B_exog")

        np.testing.assert_allclose(got, _exog_prior_sigma(endog, exog[1:], 100.0))
        assert not np.allclose(got, 1.0)
        # Belt and braces: identical to what the fit path builds, so the two
        # cannot drift apart.
        np.testing.assert_allclose(got, _captured_sigma(self._capture(data, spec), "B_exog"))

    def test_prior_predictive_graph_honours_the_knob(self, rng):
        endog = rng.standard_normal((150, 2))
        exog = rng.standard_normal((150, 1)) * 0.01
        data = _exog_data(endog, exog, ["tiny"])

        model, _ = VAR(lags=1, exog_prior_scale=5.0)._build_pymc_model(data)

        np.testing.assert_allclose(
            _captured_sigma(model, "B_exog"),
            _exog_prior_sigma(endog, exog[1:], 5.0),
        )


def _capture_model(var_data, **var_kwargs):
    """Build the production PyMC graph via VAR.fit, aborting before MCMC."""
    import pymc as pm

    captured: dict[str, pm.Model] = {}

    class CapturingSampler:
        name = "capture"

        def sample(self, model: pm.Model):
            captured["model"] = model
            raise RuntimeError("stop before sampling")

    with pytest.raises(RuntimeError, match="stop before sampling"):
        VAR(**var_kwargs).fit(var_data, sampler=CapturingSampler())
    return captured["model"]


class TestErrorDistributionParameter:
    """The `error_dist=` seam on the VAR spec (issue #152)."""

    def test_default_is_gaussian_string(self):
        assert VAR(lags=2).error_dist == "gaussian"

    def test_string_resolves_to_adapter(self):
        from impulso.observation import Gaussian, StudentT

        assert isinstance(VAR(lags=2).resolved_error_dist, Gaussian)
        resolved = VAR(lags=2, error_dist="student_t").resolved_error_dist
        assert isinstance(resolved, StudentT)
        # The string shorthand takes the adapter's defaults, i.e. inferred nu.
        assert resolved.nu == "infer"

    def test_object_pass_through_is_identity(self):
        from impulso.observation import StudentT

        adapter = StudentT(nu=7.0)
        assert VAR(lags=2, error_dist=adapter).resolved_error_dist is adapter

    def test_unknown_string_raises(self):
        with pytest.raises(ValidationError):
            VAR(lags=2, error_dist="nonexistent")

    def test_volatility_adapter_rejected_as_error_dist(self):
        """Cross-seam accident: a VolatilityProcess is not an ErrorDistribution."""
        with pytest.raises(ValidationError):
            VAR(lags=2, error_dist=Constant())

    def test_error_dist_adapter_rejected_as_volatility(self):
        from impulso.observation import StudentT

        with pytest.raises(ValidationError):
            VAR(lags=2, volatility=StudentT())


class TestStudentTRejectedWithStochasticVolatility:
    """SV x Student-t is rejected at construction, not at fit (issue #152)."""

    def test_string_forms_raise(self):
        with pytest.raises(ValidationError, match="time-varying volatility"):
            VAR(lags=1, volatility="sv", error_dist="student_t")

    def test_object_forms_raise(self):
        from impulso.observation import StudentT
        from impulso.sv.spec import StochasticVolatility

        with pytest.raises(ValidationError, match="time-varying volatility"):
            VAR(lags=1, volatility=StochasticVolatility(), error_dist=StudentT(nu=5.0))

    def test_error_names_the_alternatives(self):
        with pytest.raises(ValidationError, match="volatility='constant'"):
            VAR(lags=1, volatility="sv", error_dist="student_t")

    def test_gaussian_with_sv_still_allowed(self):
        assert VAR(lags=1, volatility="sv").resolved_volatility.is_time_varying


class TestStudentTModelBuild:
    """The graph VAR.fit builds under `error_dist='student_t'`."""

    def test_gaussian_branch_is_unchanged(self, var_data_2v):
        import pymc as pm

        model = _capture_model(var_data_2v, lags=1)
        obs = model.observed_RVs[0]
        assert isinstance(obs.owner.op, pm.MvNormal.rv_type)
        assert "nu" not in {v.name for v in model.deterministics}
        assert "nu_excess" not in {v.name for v in model.free_RVs}

    def test_fixed_nu_registers_a_deterministic(self, var_data_2v):
        import pymc as pm

        from impulso.observation import StudentT

        model = _capture_model(var_data_2v, lags=1, error_dist=StudentT(nu=5.0))
        obs = model.observed_RVs[0]
        assert isinstance(obs.owner.op, pm.MvStudentT.rv_type)
        dets = {v.name: v for v in model.deterministics}
        assert "nu" in dets
        assert "nu_excess" not in {v.name for v in model.free_RVs}
        assert float(dets["nu"].eval()) == 5.0

    def test_inferred_nu_registers_a_free_excess_rv(self, var_data_2v):
        import pymc as pm

        model = _capture_model(var_data_2v, lags=1, error_dist="student_t")
        obs = model.observed_RVs[0]
        assert isinstance(obs.owner.op, pm.MvStudentT.rv_type)
        assert "nu_excess" in {v.name for v in model.free_RVs}
        dets = {v.name: v for v in model.deterministics}
        assert "nu" in dets
        # The prior is a *shift*, not a truncation: every prior draw of nu
        # exceeds 2, so nu/(nu-2) never diverges.
        nu_prior_draws = np.asarray(pm.draw(dets["nu"], draws=50, random_seed=0))
        assert (nu_prior_draws > 2.0).all()

    @pytest.mark.parametrize("error_dist", ["gaussian", "student_t"])
    def test_manual_cholesky_parameterisation_survives(self, var_data_2v, error_dist):
        """LKJCholeskyCov is broken upstream; the manual factor must persist."""
        model = _capture_model(var_data_2v, lags=1, error_dist=error_dist)
        rv_names = {v.name for v in model.free_RVs}
        det_names = {v.name for v in model.deterministics}
        assert {"sigma_sd", "tril_offdiag"} <= rv_names
        assert "Sigma" in det_names
        op_names = {type(v.owner.op).__name__ for v in model.free_RVs}
        assert not any("LKJ" in op for op in op_names), op_names

    @pytest.mark.parametrize("error_dist", ["gaussian", "student_t"])
    def test_logp_compiles_and_is_finite(self, var_data_2v, error_dist):
        model = _capture_model(var_data_2v, lags=1, error_dist=error_dist)
        logp = model.compile_logp()(model.initial_point())
        assert np.isfinite(logp)

    def test_exogenous_regressors_survive_the_t_likelihood(self, var_data_2v):
        import pymc as pm

        from impulso.data import VARData

        data = VARData(
            endog=var_data_2v.endog,
            endog_names=var_data_2v.endog_names,
            index=var_data_2v.index,
            exog=np.arange(len(var_data_2v.endog), dtype=float)[:, None],
            exog_names=["trend"],
        )
        model = _capture_model(data, lags=1, error_dist="student_t")
        assert "B_exog" in {v.name for v in model.free_RVs}
        assert isinstance(model.observed_RVs[0].owner.op, pm.MvStudentT.rv_type)


class TestErrorDistThreadsToFittedVAR:
    """The model_construct threading guard (issue #152, risk 5).

    Forgetting `error_dist=` in `FittedVAR.model_construct` is a *silent*
    wrong-numbers bug: a t-fitted model would forecast Gaussian tails with
    no error anywhere.
    """

    def test_fit_passes_the_resolved_adapter_to_fitted_var(self, var_data_2v):
        import xarray as xr

        from impulso.observation import StudentT

        adapter = StudentT(nu=6.0)

        class StubSampler:
            name = "stub"

            def sample(self, model):
                # Minimal posterior; only the threading is under test here.
                return make_idata(
                    posterior=xr.Dataset({
                        "B": (("chain", "draw", "var", "coeff"), np.zeros((1, 2, 2, 2))),
                        "intercept": (("chain", "draw", "var"), np.zeros((1, 2, 2))),
                        "L": (("chain", "draw", "var1", "var2"), np.tile(np.eye(2), (1, 2, 1, 1))),
                        "nu": (("chain", "draw"), np.full((1, 2), 6.0)),
                    })
                )

        fitted = VAR(lags=1, error_dist=adapter).fit(var_data_2v, sampler=StubSampler())
        assert fitted.error_dist is adapter
        assert fitted.error_dist.is_heavy_tailed

    def test_gaussian_fit_keeps_a_gaussian_adapter(self, var_data_2v):
        import xarray as xr

        from impulso.observation import Gaussian

        class StubSampler:
            name = "stub"

            def sample(self, model):
                return make_idata(
                    posterior=xr.Dataset({
                        "B": (("chain", "draw", "var", "coeff"), np.zeros((1, 2, 2, 2))),
                        "intercept": (("chain", "draw", "var"), np.zeros((1, 2, 2))),
                        "L": (("chain", "draw", "var1", "var2"), np.tile(np.eye(2), (1, 2, 1, 1))),
                    })
                )

        fitted = VAR(lags=1).fit(var_data_2v, sampler=StubSampler())
        assert isinstance(fitted.error_dist, Gaussian)


class TestVolatilityShorthandSV:
    def test_sv_string_resolves_to_stochastic_volatility(self):
        from impulso.spec import VAR
        from impulso.sv.spec import StochasticVolatility

        spec = VAR(lags=2, volatility="sv")
        assert isinstance(spec.resolved_volatility, StochasticVolatility)

    def test_unknown_string_still_raises(self):
        from pydantic import ValidationError

        from impulso.spec import VAR

        with pytest.raises(ValidationError):
            VAR(lags=2, volatility="not_a_real_adapter")
