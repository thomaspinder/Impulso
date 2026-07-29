"""Invariant tests for FittedVAR.conditional_forecast (PR 3 of the scenario stack).

Core identities (design + ADR-0005): matched-seed nesting with forecast();
pins hold pathwise under hard conditions; full pins collapse variance;
dense-M cross-check of the constraint-row construction; rotation invariance
of the observable-space answer; the ADPRR path_uncertainty mode; and the
corrected plausibility statistic q with its calibration.
"""

import matplotlib

matplotlib.use("Agg")

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.figure import Figure

from impulso._linalg import lag_matrices
from impulso._ma import compute_ma_phi
from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.scenario import ShockPath, VariablePath
from impulso.volatility import Constant


@pytest.fixture
def fitted_2v(synthetic_idata_2v, var_data_2v):
    """Reduced-form 2-var VAR(1) from the synthetic posterior."""
    return FittedVAR(
        idata=synthetic_idata_2v,
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )


def _single_draw_fitted():
    """A 1-chain, 1-draw posterior for deterministic algebra tests."""
    rng = np.random.default_rng(7)
    B = np.array([[[[0.5, 0.1], [-0.2, 0.3]]]])
    intercept = np.array([[[0.1, -0.05]]])
    L = np.array([[np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 0.8]]))]])
    posterior = xr.Dataset({
        "B": (("chain", "draw", "var", "coeff"), B),
        "intercept": (("chain", "draw", "var"), intercept),
        "L": (("chain", "draw", "var1", "var2"), L),
    })
    data = VARData(
        endog=rng.standard_normal((12, 2)),
        endog_names=["y1", "y2"],
        index=pd.date_range("2000-01-01", periods=12, freq="QS"),
    )
    return FittedVAR(
        idata=az.InferenceData(posterior=posterior),
        n_lags=1,
        data=data,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )


class TestForecastNesting:
    def test_no_conditions_matches_forecast_with_matched_seed(self, fitted_2v):
        """Invariant 4: the RNG contract makes the nesting exact per draw."""
        cf = fitted_2v.conditional_forecast(steps=8, seed=123)
        fc = fitted_2v.forecast(steps=8, seed=123)
        np.testing.assert_allclose(
            cf.idata.posterior_predictive["forecast"].values,
            fc.idata.posterior_predictive["forecast"].values,
            atol=1e-8,
        )

    def test_mean_mode_matches_forecast_mean_mode(self, fitted_2v):
        cf = fitted_2v.conditional_forecast(steps=8, include_shock_uncertainty=False)
        fc = fitted_2v.forecast(steps=8, include_shock_uncertainty=False)
        np.testing.assert_allclose(
            cf.idata.posterior_predictive["forecast"].values,
            fc.idata.posterior_predictive["forecast"].values,
            atol=1e-8,
        )

    def test_no_conditions_plausibility_is_zero(self, fitted_2v):
        cf = fitted_2v.conditional_forecast(steps=4, seed=0)
        pp = cf.idata.posterior_predictive
        np.testing.assert_array_equal(pp["plausibility"].values, 0.0)
        np.testing.assert_allclose(pp["plausibility_calibrated"].values, 0.5)
        assert pp.attrs["n_restrictions"] == 0
        assert pp.attrs["chi2_tail_of_median"] == 1.0


class TestHardConditions:
    def test_pins_hold_pathwise(self, fitted_2v):
        """Every density draw satisfies the pinned entries exactly."""
        values = np.array([0.4, np.nan, -0.2])  # steps 1 and 3 pinned, 2 free
        cf = fitted_2v.conditional_forecast(steps=5, conditions=[VariablePath(variable="y1", values=values)], seed=11)
        draws = cf.idata.posterior_predictive["forecast"].values
        np.testing.assert_allclose(draws[:, :, 0, 0], 0.4, atol=1e-8)
        np.testing.assert_allclose(draws[:, :, 2, 0], -0.2, atol=1e-8)
        # The NaN step and the other variable keep genuine draw variation.
        assert draws[:, :, 1, 0].std() > 1e-4
        assert draws[:, :, 0, 1].std() > 1e-4

    def test_full_pins_collapse_variance(self, fitted_2v):
        """Invariant 7: pinning every variable at every step is deterministic."""
        steps = 4
        target = {"y1": 0.3, "y2": -0.1}
        cf = fitted_2v.conditional_forecast(
            steps=steps,
            conditions=[VariablePath(variable=v, values=t) for v, t in target.items()],
            seed=5,
        )
        draws = cf.idata.posterior_predictive["forecast"].values
        np.testing.assert_allclose(draws[:, :, :, 0], 0.3, atol=1e-8)
        np.testing.assert_allclose(draws[:, :, :, 1], -0.1, atol=1e-8)

    def test_matches_dense_stacked_solve(self, fitted_2v, synthetic_idata_2v):
        """The row construction equals an independently materialised M."""
        steps = 3
        pins = [(0, 0, 0.5), (1, 2, -0.3)]  # (variable, step0, value)
        conditions = [
            VariablePath(variable="y1", values=np.array([0.5, np.nan, np.nan])),
            VariablePath(variable="y2", values=np.array([np.nan, np.nan, -0.3])),
        ]
        cf = fitted_2v.conditional_forecast(steps=steps, conditions=conditions, include_shock_uncertainty=False)
        engine_paths = cf.idata.posterior_predictive["forecast"].values
        engine_q = cf.idata.posterior_predictive["plausibility"].values

        B = synthetic_idata_2v.posterior["B"].values
        n_chains, n_draws, n, _ = B.shape
        b = fitted_2v.forecast(steps, include_shock_uncertainty=False).idata.posterior_predictive["forecast"].values
        L = synthetic_idata_2v.posterior["L"].values
        Phi = compute_ma_phi(lag_matrices(B, 1), steps - 1)
        d_tot = steps * n
        dense_paths = np.zeros_like(engine_paths)
        dense_q = np.zeros((n_chains, n_draws))
        for c in range(n_chains):
            for d in range(n_draws):
                M = np.zeros((d_tot, d_tot))
                for h in range(steps):
                    for s in range(h + 1):
                        M[h * n : (h + 1) * n, s * n : (s + 1) * n] = Phi[c, d, h - s] @ L[c, d]
                rows = np.array([M[h * n + i] for (i, h, _) in pins])
                cbar = np.array([v - b[c, d, h, i] for (i, h, v) in pins])
                G_inv = np.linalg.inv(rows @ rows.T)
                mu = rows.T @ G_inv @ cbar
                dense_paths[c, d] = b[c, d] + (M @ mu).reshape(steps, n)
                dense_q[c, d] = cbar @ G_inv @ cbar
        np.testing.assert_allclose(engine_paths, dense_paths, atol=1e-8)
        np.testing.assert_allclose(engine_q, dense_q, atol=1e-8)


class TestRotationInvariance:
    def test_mean_mode_invariant_to_orthogonal_refactorisation(self, synthetic_idata_2v, var_data_2v):
        """Invariant 5: replacing L by LQ (same Sigma) leaves the answer unchanged."""
        rng = np.random.default_rng(3)
        Q, _ = np.linalg.qr(rng.standard_normal((2, 2)))
        rotated = synthetic_idata_2v.posterior.copy(deep=True)
        rotated["L"].values[:] = np.einsum("cdij,jk->cdik", synthetic_idata_2v.posterior["L"].values, Q)

        conditions = [VariablePath(variable="y1", values=0.7)]
        kwargs = {"steps": 5, "conditions": conditions, "include_shock_uncertainty": False}
        base = FittedVAR(
            idata=synthetic_idata_2v, n_lags=1, data=var_data_2v, var_names=["y1", "y2"], volatility=Constant()
        ).conditional_forecast(**kwargs)
        rot = FittedVAR(
            idata=az.InferenceData(posterior=rotated),
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        ).conditional_forecast(**kwargs)

        np.testing.assert_allclose(
            base.idata.posterior_predictive["forecast"].values,
            rot.idata.posterior_predictive["forecast"].values,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            base.idata.posterior_predictive["plausibility"].values,
            rot.idata.posterior_predictive["plausibility"].values,
            atol=1e-8,
        )


class TestPathUncertainty:
    def test_mean_mode_identical_across_modes(self, fitted_2v):
        conditions = [VariablePath(variable="y1", values=0.4)]
        hard = fitted_2v.conditional_forecast(steps=4, conditions=conditions, include_shock_uncertainty=False)
        soft = fitted_2v.conditional_forecast(
            steps=4, conditions=conditions, include_shock_uncertainty=False, path_uncertainty="unconditional"
        )
        np.testing.assert_allclose(
            hard.idata.posterior_predictive["forecast"].values,
            soft.idata.posterior_predictive["forecast"].values,
            atol=1e-12,
        )

    def test_unconditional_mode_keeps_band_width(self, fitted_2v):
        """ADPRR Omega_f = DD': pins restrict the mean; draws keep spread."""
        conditions = [VariablePath(variable="y1", values=0.4)]
        hard = fitted_2v.conditional_forecast(steps=4, conditions=conditions, seed=9)
        soft = fitted_2v.conditional_forecast(steps=4, conditions=conditions, seed=9, path_uncertainty="unconditional")
        hard_draws = hard.idata.posterior_predictive["forecast"].values
        soft_draws = soft.idata.posterior_predictive["forecast"].values
        # Hard pins are exact pathwise; the soft mode keeps genuine variation...
        np.testing.assert_allclose(hard_draws[:, :, :, 0], 0.4, atol=1e-8)
        assert soft_draws[:, :, 0, 0].std() > 0.1 * hard_draws[:, :, 0, 1].std()
        # ...with strictly wider spread at the pinned coordinates.
        assert soft_draws[:, :, 0, 0].std() > 10 * hard_draws[:, :, 0, 0].std()

    def test_invalid_mode_errors(self, fitted_2v):
        with pytest.raises(ValueError, match="path_uncertainty"):
            fitted_2v.conditional_forecast(steps=4, path_uncertainty="soft")


class TestPlausibility:
    def test_ray_monotonicity_quadratic(self):
        """q is exactly quadratic along a ray scaling the pin deviations."""
        fitted = _single_draw_fitted()
        b = fitted.forecast(3, include_shock_uncertainty=False).idata.posterior_predictive["forecast"].values
        delta = np.array([1.0, -0.5, 0.7])

        def q_at(t):
            conditions = [VariablePath(variable="y1", values=b[0, 0, :, 0] + t * delta)]
            cf = fitted.conditional_forecast(steps=3, conditions=conditions, include_shock_uncertainty=False)
            return float(cf.idata.posterior_predictive["plausibility"].values[0, 0])

        q1, q2 = q_at(1.0), q_at(2.0)
        assert q1 > 0
        np.testing.assert_allclose(q2, 4.0 * q1, rtol=1e-8)

    def test_pinning_the_unconditional_path_gives_zero_q(self):
        fitted = _single_draw_fitted()
        b = fitted.forecast(3, include_shock_uncertainty=False).idata.posterior_predictive["forecast"].values
        conditions = [VariablePath(variable="y1", values=b[0, 0, :, 0])]
        hard = fitted.conditional_forecast(steps=3, conditions=conditions, include_shock_uncertainty=False)
        np.testing.assert_allclose(hard.idata.posterior_predictive["plausibility"].values, 0.0, atol=1e-16)
        # Hard pins peg q_cal at the ADPRR ceiling regardless of the mean shift...
        np.testing.assert_array_equal(hard.idata.posterior_predictive["plausibility_calibrated"].values, 1.0)
        # ...while the unconditional-variance mode reaches the q = 0 floor.
        soft = fitted.conditional_forecast(
            steps=3, conditions=conditions, include_shock_uncertainty=False, path_uncertainty="unconditional"
        )
        np.testing.assert_allclose(soft.idata.posterior_predictive["plausibility_calibrated"].values, 0.5, atol=1e-8)

    def test_calibration_formula_unconditional_mode(self, fitted_2v):
        cf = fitted_2v.conditional_forecast(
            steps=4,
            conditions=[VariablePath(variable="y1", values=1.5)],
            include_shock_uncertainty=False,
            path_uncertainty="unconditional",
        )
        pp = cf.idata.posterior_predictive
        q = pp["plausibility"].values
        expected = (1.0 + np.sqrt(1.0 - np.exp(-q / (4 * 2)))) / 2.0
        np.testing.assert_allclose(pp["plausibility_calibrated"].values, expected, atol=1e-12)

    def test_hard_mode_calibration_pegs_at_one(self, fitted_2v):
        """Under hard pins ADPRR's divergence is infinite; q_cal sits at its ceiling."""
        cf = fitted_2v.conditional_forecast(
            steps=4, conditions=[VariablePath(variable="y1", values=1.5)], include_shock_uncertainty=False
        )
        np.testing.assert_array_equal(cf.idata.posterior_predictive["plausibility_calibrated"].values, 1.0)

    def test_q_follows_chi_squared_under_model_drawn_pins(self):
        """Invariant 9 (chi^2 part): pins drawn from the model's own law give q ~ chi^2_1."""
        fitted = _single_draw_fitted()
        qs = []
        for k in range(200):
            path = fitted.forecast(steps=2, seed=k).idata.posterior_predictive["forecast"].values
            pin_value = float(path[0, 0, 1, 0])  # the drawn (y1, step 2) realisation
            cf = fitted.conditional_forecast(
                steps=2,
                conditions=[VariablePath(variable="y1", values=np.array([np.nan, pin_value]))],
                include_shock_uncertainty=False,
            )
            qs.append(float(cf.idata.posterior_predictive["plausibility"].values[0, 0]))
        qs = np.asarray(qs)
        assert 0.7 < qs.mean() < 1.3  # E[chi^2_1] = 1
        assert 1.0 < qs.var() < 3.4  # Var[chi^2_1] = 2

    def test_summary_accessor(self, fitted_2v):
        from scipy.stats import chi2

        cf = fitted_2v.conditional_forecast(steps=4, conditions=[VariablePath(variable="y1", values=1.0)], seed=2)
        summary = cf.plausibility()
        assert summary["n_restrictions"] == 4
        assert 0.0 <= summary["tail_probability"] <= 1.0
        assert 0.5 <= summary["q_calibrated_median"] <= 1.0
        assert summary["q_hdi_lower"] <= summary["q_median"] <= summary["q_hdi_upper"]
        pp = cf.idata.posterior_predictive
        expected_tail = float(chi2.sf(float(np.median(pp["plausibility"].values)), df=4))
        assert summary["tail_probability"] == pytest.approx(expected_tail)


class _RngConsumingVol:
    """Time-varying fake whose forecast Cholesky path consumes the generator.

    The matched-seed nesting invariant is only falsifiable under an adapter
    that actually draws from the rng — Constant ignores it, so a wrong
    stream order would pass every Constant-based test.
    """

    name = "fake-sv"
    is_time_varying = True

    def build_pymc_latent(self, n_vars, T):  # pragma: no cover
        raise NotImplementedError

    def cholesky_at(self, posterior, t):
        return posterior["L"].values

    def cholesky_path(self, posterior, T):  # pragma: no cover
        raise NotImplementedError

    def forecast_cholesky_path(self, posterior, steps, rng):
        L = posterior["L"].values
        n_chains, n_draws = L.shape[:2]
        scales = 1.0 + 0.2 * np.abs(rng.standard_normal((n_chains, n_draws, steps)))
        return L[:, :, np.newaxis, :, :] * scales[:, :, :, np.newaxis, np.newaxis]


class TestRngConsumingVolatility:
    @pytest.fixture
    def fitted_sv(self, synthetic_idata_2v, var_data_2v):
        return FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=_RngConsumingVol(),
        )

    def test_matched_seed_nesting_when_volatility_consumes_rng(self, fitted_sv):
        """The RNG contract is falsifiable only when the adapter draws from rng."""
        cf = fitted_sv.conditional_forecast(steps=6, seed=77)
        fc = fitted_sv.forecast(steps=6, seed=77)
        np.testing.assert_allclose(
            cf.idata.posterior_predictive["forecast"].values,
            fc.idata.posterior_predictive["forecast"].values,
            atol=1e-8,
        )

    def test_pins_hold_pathwise_under_time_varying_volatility(self, fitted_sv):
        cf = fitted_sv.conditional_forecast(steps=4, conditions=[VariablePath(variable="y1", values=0.4)], seed=3)
        draws = cf.idata.posterior_predictive["forecast"].values
        np.testing.assert_allclose(draws[:, :, :, 0], 0.4, atol=1e-8)


def _single_draw_fitted_exog():
    """The single-draw posterior extended with one exogenous regressor."""
    rng = np.random.default_rng(7)
    posterior = xr.Dataset({
        "B": (("chain", "draw", "var", "coeff"), np.array([[[[0.5, 0.1], [-0.2, 0.3]]]])),
        "B_exog": (("chain", "draw", "var", "exog"), np.array([[[[0.8], [-0.3]]]])),
        "intercept": (("chain", "draw", "var"), np.array([[[0.1, -0.05]]])),
        "L": (("chain", "draw", "var1", "var2"), np.array([[np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 0.8]]))]])),
    })
    data = VARData(
        endog=rng.standard_normal((12, 2)),
        endog_names=["y1", "y2"],
        exog=rng.standard_normal((12, 1)),
        exog_names=["z"],
        index=pd.date_range("2000-01-01", periods=12, freq="QS"),
    )
    return FittedVAR(
        idata=az.InferenceData(posterior=posterior),
        n_lags=1,
        data=data,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )


class TestExogHappyPath:
    def test_matched_seed_nesting_with_exog(self):
        fitted = _single_draw_fitted_exog()
        exog_future = np.ones((5, 1))
        cf = fitted.conditional_forecast(steps=5, seed=9, exog_future=exog_future)
        fc = fitted.forecast(steps=5, seed=9, exog_future=exog_future)
        np.testing.assert_allclose(
            cf.idata.posterior_predictive["forecast"].values,
            fc.idata.posterior_predictive["forecast"].values,
            atol=1e-8,
        )

    def test_exog_shifts_unpinned_path_while_pin_holds(self):
        fitted = _single_draw_fitted_exog()
        pin = [VariablePath(variable="y1", values=0.3)]
        low = fitted.conditional_forecast(
            steps=4, conditions=pin, include_shock_uncertainty=False, exog_future=np.zeros((4, 1))
        )
        high = fitted.conditional_forecast(
            steps=4, conditions=pin, include_shock_uncertainty=False, exog_future=np.ones((4, 1))
        )
        low_draws = low.idata.posterior_predictive["forecast"].values
        high_draws = high.idata.posterior_predictive["forecast"].values
        np.testing.assert_allclose(low_draws[:, :, :, 0], 0.3, atol=1e-8)
        np.testing.assert_allclose(high_draws[:, :, :, 0], 0.3, atol=1e-8)
        assert np.abs(low_draws[:, :, :, 1] - high_draws[:, :, :, 1]).max() > 1e-3


class TestGuards:
    def test_unknown_variable_errors(self, fitted_2v):
        with pytest.raises(ValueError, match="Unknown variable"):
            fitted_2v.conditional_forecast(steps=4, conditions=[VariablePath(variable="oil", values=1.0)])

    def test_shock_path_condition_rejected(self, fitted_2v):
        with pytest.raises(TypeError, match="VariablePath"):
            fitted_2v.conditional_forecast(steps=4, conditions=[ShockPath(shock="y1", values=0.0)])

    def test_duplicate_pins_error(self, fitted_2v):
        with pytest.raises(ValueError, match="Duplicate pin"):
            fitted_2v.conditional_forecast(
                steps=4,
                conditions=[
                    VariablePath(variable="y1", values=1.0),
                    VariablePath(variable="y1", values=np.array([0.5])),
                ],
            )

    def test_over_length_values_error(self, fitted_2v):
        with pytest.raises(ValueError, match="length"):
            fitted_2v.conditional_forecast(steps=2, conditions=[VariablePath(variable="y1", values=np.zeros(5))])

    def test_scalar_nan_condition_errors(self, fitted_2v):
        with pytest.raises(ValueError, match="pins nothing"):
            fitted_2v.conditional_forecast(steps=4, conditions=[VariablePath(variable="y1", values=float("nan"))])

    def test_exog_data_without_posterior_b_exog_errors(self, synthetic_idata_2v):
        rng = np.random.default_rng(0)
        data = VARData(
            endog=rng.standard_normal((30, 2)),
            endog_names=["y1", "y2"],
            exog=rng.standard_normal((30, 1)),
            exog_names=["z"],
            index=pd.date_range("2000-01-01", periods=30, freq="QS"),
        )
        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=data,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        with pytest.raises(ValueError, match="never consumed"):
            fitted.conditional_forecast(steps=4)

    def test_exog_future_shape_validated(self, fitted_2v):
        with pytest.raises(ValueError, match="no B_exog"):
            fitted_2v.conditional_forecast(steps=4, exog_future=np.zeros((4, 1)))


class TestReproducibility:
    def test_same_seed_same_result(self, fitted_2v):
        conditions = [VariablePath(variable="y1", values=0.4)]
        a = fitted_2v.conditional_forecast(steps=6, conditions=conditions, seed=42)
        b = fitted_2v.conditional_forecast(steps=6, conditions=conditions, seed=42)
        np.testing.assert_array_equal(
            a.idata.posterior_predictive["forecast"].values,
            b.idata.posterior_predictive["forecast"].values,
        )


class TestResultSurface:
    def test_median_and_hdi_shapes(self, fitted_2v):
        cf = fitted_2v.conditional_forecast(steps=6, conditions=[VariablePath(variable="y1", values=0.4)], seed=1)
        med = cf.median()
        assert med.shape == (6, 2)
        assert med.index.name == "step"
        hdi = cf.hdi(prob=0.9)
        assert hdi.lower.shape == med.shape
        assert (hdi.upper.values >= hdi.lower.values).all()

    def test_plot_marks_pins(self, fitted_2v):
        cf = fitted_2v.conditional_forecast(steps=6, conditions=[VariablePath(variable="y1", values=0.4)], seed=1)
        fig = cf.plot()
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        assert "Conditional Forecast" in fig._suptitle.get_text()
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert "pinned" in labels


class TestHeavyTailedErrorsRejected:
    """conditional_forecast is Gaussian-only (issue #152).

    The Waggoner-Zha constrained draw *is* the Gaussian conditional-law
    formula and the plausibility statistic's chi-squared reference assumes
    Gaussian shocks, so a heavy-tailed fit gets a clear error rather than a
    half-valid answer.
    """

    @pytest.fixture
    def fitted_2v_t(self, synthetic_idata_2v_t, var_data_2v):
        from impulso.observation import StudentT

        return FittedVAR(
            idata=synthetic_idata_2v_t,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            error_dist=StudentT(nu=5.0),
        )

    @pytest.mark.parametrize("include_shock_uncertainty", [True, False])
    @pytest.mark.parametrize("conditions", [None, [VariablePath(variable="y1", values=0.4)]])
    def test_raises_not_implemented(self, fitted_2v_t, include_shock_uncertainty, conditions):
        with pytest.raises(NotImplementedError, match="Gaussian-only"):
            fitted_2v_t.conditional_forecast(
                steps=4,
                conditions=conditions,
                include_shock_uncertainty=include_shock_uncertainty,
                seed=1,
            )

    def test_error_names_the_alternatives(self, fitted_2v_t):
        with pytest.raises(NotImplementedError, match=r"forecast\(\)"):
            fitted_2v_t.conditional_forecast(steps=4)
        with pytest.raises(NotImplementedError, match="error_dist='gaussian'"):
            fitted_2v_t.conditional_forecast(steps=4)

    def test_guard_fires_before_any_validation(self, fitted_2v_t):
        """Heavy tails win over the path_uncertainty ValueError."""
        with pytest.raises(NotImplementedError):
            fitted_2v_t.conditional_forecast(steps=4, path_uncertainty="bogus")

    def test_gaussian_sibling_still_works(self, fitted_2v):
        result = fitted_2v.conditional_forecast(steps=4, seed=1)
        assert result.median().shape == (4, 2)
