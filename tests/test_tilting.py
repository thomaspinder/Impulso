"""Invariant tests for entropic tilting (issue #150).

The solver half is checked against analytic oracles: the closed-form
two-mass tilt and its ESS/KL, the Gaussian mean tilt whose dual solution
is known in closed form, and the primal problem solved independently by
SLSQP. The result half checks the guards, the weighted summaries, and the
theorem that hard pins survive any reweighting.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from impulso._tilting import (
    build_moments,
    ess,
    kl_divergence,
    solve_tilt,
    weighted_hdi,
    weighted_quantile,
)
from impulso.scenario import MomentTarget, ProbabilityTarget


def _forecast_from_series(x: np.ndarray, n_chains: int = 1) -> np.ndarray:
    """Wrap a 1-D sample as a `(C, D, 1, 1)` forecast array."""
    return x.reshape(n_chains, -1, 1, 1)


class TestVocabulary:
    def test_horizon_must_be_positive(self):
        with pytest.raises(ValueError, match="1-based"):
            ProbabilityTarget(variable="y1", horizon=0, threshold=0.0, probability=0.5)

    def test_probability_bounds(self):
        with pytest.raises(ValueError, match="0 < p <= 1"):
            ProbabilityTarget(variable="y1", horizon=1, threshold=0.0, probability=0.0)
        with pytest.raises(ValueError, match="0 < p <= 1"):
            ProbabilityTarget(variable="y1", horizon=1, threshold=0.0, probability=1.5)

    def test_non_finite_threshold_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            ProbabilityTarget(variable="y1", horizon=1, threshold=np.inf, probability=0.5)

    def test_moment_target_requires_finite_mean(self):
        with pytest.raises(ValueError, match="finite"):
            MomentTarget(variable="y1", horizon=1, mean=np.nan)

    def test_targets_are_frozen(self):
        target = MomentTarget(variable="y1", horizon=2, mean=0.5)
        with pytest.raises(ValueError, match="frozen"):
            target.mean = 1.0


class TestClosedForm:
    """Single probability target: the two-mass solution and its oracles."""

    @staticmethod
    def _setup(n_total=100, n_event=20, probability=0.3):
        x = np.arange(float(n_total))
        # The first n_event draws sit below the threshold.
        forecast = _forecast_from_series(x)
        target = ProbabilityTarget(variable="y1", horizon=1, threshold=float(n_event) - 0.5, probability=probability)
        G, t = build_moments(forecast, [target], ["y1"], steps=1)
        return G, t, n_total, n_event, probability

    def test_weights_match_the_analytic_two_mass_solution(self):
        G, t, n_total, n_event, p = self._setup()
        w, achieved = solve_tilt(G, t)
        expected = np.where(G[:, 0] > 0.5, p / n_event, (1.0 - p) / (n_total - n_event))
        assert np.allclose(w, expected, atol=1e-15, rtol=0.0)
        assert achieved[0] == pytest.approx(p, abs=1e-15)
        assert w.sum() == pytest.approx(1.0, abs=1e-14)

    def test_ess_matches_the_analytic_value(self):
        G, t, n_total, n_event, p = self._setup()
        w, _ = solve_tilt(G, t)
        oracle = 1.0 / (p**2 / n_event + (1.0 - p) ** 2 / (n_total - n_event))
        assert ess(w) == pytest.approx(oracle, rel=1e-13)

    def test_kl_matches_the_analytic_value(self):
        G, t, n_total, n_event, p = self._setup()
        w, _ = solve_tilt(G, t)
        p_hat = n_event / n_total
        oracle = p * np.log(p / p_hat) + (1.0 - p) * np.log((1.0 - p) / (1.0 - p_hat))
        assert kl_divergence(w) == pytest.approx(oracle, rel=1e-13)

    def test_probability_one_is_pure_conditioning(self):
        G, t, _, n_event, _ = self._setup(probability=1.0)
        w, achieved = solve_tilt(G, t)
        in_event = G[:, 0] > 0.5
        assert np.allclose(w[in_event], 1.0 / n_event, atol=1e-15, rtol=0.0)
        assert np.all(w[~in_event] == 0.0)
        assert achieved[0] == pytest.approx(1.0, abs=1e-15)
        assert ess(w) == pytest.approx(float(n_event), rel=1e-13)

    def test_direction_above_flips_the_event(self):
        x = np.arange(100.0)
        target = ProbabilityTarget(variable="y1", horizon=1, threshold=79.5, probability=0.5, direction="above")
        G, _ = build_moments(_forecast_from_series(x), [target], ["y1"], steps=1)
        assert G[:, 0].sum() == 20.0
        assert np.all(G[80:, 0] == 1.0)

    def test_uniform_weights_when_the_target_matches_the_sample(self):
        G, t, _, _, _ = self._setup(probability=0.2)
        w, _ = solve_tilt(G, t)
        assert np.allclose(w, 1.0 / 100.0, atol=1e-15, rtol=0.0)
        assert kl_divergence(w) == pytest.approx(0.0, abs=1e-15)


class TestGaussianMeanTilt:
    """Tilting N(0,1) draws to a target mean — the RTW textbook case."""

    @staticmethod
    def _solve(mean=0.5, n_total=50_000):
        x = np.random.default_rng(0).standard_normal(n_total)
        target = MomentTarget(variable="x", horizon=1, mean=mean)
        G, t = build_moments(_forecast_from_series(x), [target], ["x"], steps=1)
        w, achieved = solve_tilt(G, t)
        return x, w, achieved

    def test_achieved_mean_hits_the_target(self):
        _, _, achieved = self._solve()
        assert achieved[0] == pytest.approx(0.5, abs=1e-8)

    def test_log_weights_are_affine_in_the_variable(self):
        x, w, _ = self._solve()
        # log w = lambda x - log Z exactly, so a two-point fit reproduces
        # every other draw.
        design = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(design, np.log(w), rcond=None)
        residual = np.max(np.abs(design @ coef - np.log(w)))
        assert residual < 1e-10

    def test_dual_solution_matches_an_independent_root_find(self):
        from scipy.optimize import brentq

        x, w, _ = self._solve()
        design = np.column_stack([x, np.ones_like(x)])
        lam_hat = np.linalg.lstsq(design, np.log(w), rcond=None)[0][0]

        def gap(lam):
            z = np.exp(lam * (x - x.max()))
            return float((x * z).sum() / z.sum() - 0.5)

        lam_star = brentq(gap, -5.0, 5.0, xtol=1e-14, rtol=1e-15)
        assert lam_hat == pytest.approx(lam_star, abs=1e-8)

    def test_dual_solution_is_near_the_population_value(self):
        # Tilting a standard normal to mean m has population lambda = m.
        x, w, _ = self._solve()
        design = np.column_stack([x, np.ones_like(x)])
        lam_hat = np.linalg.lstsq(design, np.log(w), rcond=None)[0][0]
        assert lam_hat == pytest.approx(0.5, abs=3.0 / np.sqrt(x.size))


class TestMultiTargetDual:
    def test_dual_matches_the_primal_slsqp_solution(self):
        from scipy.optimize import minimize

        rng = np.random.default_rng(3)
        n_total = 200
        y = rng.standard_normal((1, n_total, 2, 2))
        targets = [
            MomentTarget(variable="y1", horizon=1, mean=0.3),
            ProbabilityTarget(variable="y2", horizon=2, threshold=0.0, probability=0.7),
        ]
        G, t = build_moments(y, targets, ["y1", "y2"], steps=2)
        w, achieved = solve_tilt(G, t)
        assert np.allclose(achieved, t, atol=1e-6)

        def entropy(v):
            v = np.clip(v, 1e-300, None)
            return float(np.sum(v * np.log(n_total * v)))

        primal = minimize(
            entropy,
            np.full(n_total, 1.0 / n_total),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_total,
            constraints=[
                {"type": "eq", "fun": lambda v: v.sum() - 1.0},
                {"type": "eq", "fun": lambda v, G=G, t=t: G.T @ v - t},
            ],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        assert np.allclose(w, primal.x, atol=1e-5)

    def test_multiple_probability_targets_take_the_dual_path(self):
        rng = np.random.default_rng(11)
        y = rng.standard_normal((2, 500, 3, 1))
        targets = [
            ProbabilityTarget(variable="y1", horizon=1, threshold=0.0, probability=0.8),
            ProbabilityTarget(variable="y1", horizon=3, threshold=0.5, probability=0.6),
        ]
        G, t = build_moments(y, targets, ["y1"], steps=3)
        w, achieved = solve_tilt(G, t)
        assert np.allclose(achieved, t, atol=1e-8)
        assert w.min() >= 0.0


class TestInfeasibility:
    def test_empty_event_names_the_draw_counts(self):
        x = np.arange(100.0)
        target = ProbabilityTarget(variable="y1", horizon=1, threshold=-5.0, probability=0.5)
        with pytest.raises(ValueError, match=r"0 of 100 draws satisfy"):
            build_moments(_forecast_from_series(x), [target], ["y1"], steps=1)

    def test_full_event_with_partial_probability_raises(self):
        x = np.arange(100.0)
        target = ProbabilityTarget(variable="y1", horizon=1, threshold=1e6, probability=0.5)
        with pytest.raises(ValueError, match=r"only achievable probability is 1\.0"):
            build_moments(_forecast_from_series(x), [target], ["y1"], steps=1)

    def test_moment_target_beyond_the_sample_range_raises(self):
        x = np.arange(100.0)
        target = MomentTarget(variable="y1", horizon=1, mean=500.0)
        with pytest.raises(ValueError, match="outside the range spanned"):
            build_moments(_forecast_from_series(x), [target], ["y1"], steps=1)

    def test_jointly_infeasible_targets_report_achieved_values(self):
        # Two disjoint events cannot both carry probability 0.7.
        x = np.arange(100.0)
        forecast = _forecast_from_series(x)
        targets = [
            ProbabilityTarget(variable="y1", horizon=1, threshold=49.5, probability=0.7),
            ProbabilityTarget(variable="y1", horizon=1, threshold=49.5, probability=0.7, direction="above"),
        ]
        G, t = build_moments(forecast, targets, ["y1"], steps=1)
        with pytest.raises(ValueError, match="not jointly achievable"):
            solve_tilt(G, t)

    def test_unknown_variable_raises(self):
        x = np.arange(10.0)
        target = MomentTarget(variable="nope", horizon=1, mean=1.0)
        with pytest.raises(ValueError, match="Unknown variable"):
            build_moments(_forecast_from_series(x), [target], ["y1"], steps=1)

    def test_horizon_beyond_the_forecast_raises(self):
        x = np.arange(10.0)
        target = MomentTarget(variable="y1", horizon=4, mean=1.0)
        with pytest.raises(ValueError, match="only 1 steps"):
            build_moments(_forecast_from_series(x), [target], ["y1"], steps=1)

    def test_empty_target_list_raises(self):
        with pytest.raises(ValueError, match="at least one target"):
            build_moments(_forecast_from_series(np.arange(10.0)), [], ["y1"], steps=1)

    def test_wrong_target_type_raises(self):
        from impulso.scenario import VariablePath

        with pytest.raises(TypeError, match="ProbabilityTarget or MomentTarget"):
            build_moments(
                _forecast_from_series(np.arange(10.0)),
                [VariablePath(variable="y1", values=0.0)],
                ["y1"],
                steps=1,
            )


class TestWeightedSummaries:
    def test_uniform_weights_reproduce_numpy_hazen_quantiles(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(501)
        w = np.full(x.size, 1.0 / x.size)
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            assert weighted_quantile(x, w, q) == pytest.approx(float(np.quantile(x, q, method="hazen")), abs=1e-12)

    def test_two_mass_quantile_matches_the_closed_form(self):
        x = np.array([0.0, 1.0])
        w = np.array([0.25, 0.75])
        # Knots at c = [0.125, 0.625]; q = 0.5 interpolates between them.
        expected = (0.5 - 0.125) / (0.625 - 0.125)
        assert weighted_quantile(x, w, 0.5) == pytest.approx(expected, abs=1e-14)
        assert weighted_quantile(x, w, 0.05) == pytest.approx(0.0, abs=1e-14)
        assert weighted_quantile(x, w, 0.99) == pytest.approx(1.0, abs=1e-14)

    def test_quantile_ignores_weight_normalisation_and_ordering(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal(200)
        w = rng.random(200)
        base = weighted_quantile(x, w / w.sum(), 0.4)
        assert weighted_quantile(x, 7.0 * w, 0.4) == pytest.approx(base, abs=1e-12)
        perm = rng.permutation(200)
        assert weighted_quantile(x[perm], w[perm], 0.4) == pytest.approx(base, abs=1e-12)

    def test_quantile_is_monotone_in_q(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(300)
        w = rng.random(300)
        levels = np.linspace(0.01, 0.99, 25)
        values = [weighted_quantile(x, w, q) for q in levels]
        assert np.all(np.diff(values) >= -1e-12)

    def test_quantile_vectorises_over_trailing_dimensions(self):
        rng = np.random.default_rng(8)
        x = rng.standard_normal((400, 3, 2))
        w = rng.random(400)
        out = weighted_quantile(x, w, 0.5)
        assert out.shape == (3, 2)
        assert out[1, 1] == pytest.approx(weighted_quantile(x[:, 1, 1], w, 0.5), abs=1e-12)

    def test_uniform_weighted_hdi_matches_arviz(self):
        import arviz as az

        rng = np.random.default_rng(9)
        x = rng.standard_normal(501)  # prob * n is not an integer
        w = np.full(x.size, 1.0 / x.size)
        lower, upper = weighted_hdi(x, w, prob=0.89)
        expected = az.hdi(x, hdi_prob=0.89)
        assert float(lower) == pytest.approx(float(expected[0]), abs=1e-12)
        assert float(upper) == pytest.approx(float(expected[1]), abs=1e-12)

    def test_hdi_concentrates_when_the_weights_concentrate(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal(1000)
        uniform = np.full(x.size, 1.0 / x.size)
        concentrated = np.where(np.abs(x) < 0.25, 1.0, 1e-8)
        concentrated = concentrated / concentrated.sum()
        wide = weighted_hdi(x, uniform, prob=0.89)
        narrow = weighted_hdi(x, concentrated, prob=0.89)
        assert float(narrow[1] - narrow[0]) < float(wide[1] - wide[0])

    def test_hdi_rejects_an_out_of_range_probability(self):
        with pytest.raises(ValueError, match="0 < prob <= 1"):
            weighted_hdi(np.arange(10.0), np.full(10, 0.1), prob=1.5)

    def test_summaries_reject_mismatched_draw_counts(self):
        with pytest.raises(ValueError, match="draws on its leading axis"):
            weighted_quantile(np.arange(10.0), np.full(5, 0.2), 0.5)


@pytest.fixture
def fitted_2v(synthetic_idata_2v, var_data_2v):
    """Reduced-form 2-var VAR(1) from the synthetic posterior."""
    from impulso.fitted import FittedVAR
    from impulso.volatility import Constant

    return FittedVAR(
        idata=synthetic_idata_2v,
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )


def _median_target(forecast, variable="y1", horizon=3, probability=0.8):
    """A probability target at the forecast's own empirical median."""
    da = forecast.idata.posterior_predictive["forecast"]
    threshold = float(np.median(da.sel(variable=variable).isel(step=horizon - 1).values))
    return ProbabilityTarget(variable=variable, horizon=horizon, threshold=threshold, probability=probability)


class TestTiltEntryPoints:
    def test_mean_mode_forecast_is_refused(self, fitted_2v):
        forecast = fitted_2v.forecast(4, include_shock_uncertainty=False)
        target = ProbabilityTarget(variable="y1", horizon=2, threshold=0.0, probability=0.5)
        with pytest.raises(ValueError, match="needs a density forecast"):
            forecast.tilt([target])

    def test_mean_mode_conditional_forecast_is_refused(self, fitted_2v):
        from impulso.scenario import VariablePath

        result = fitted_2v.conditional_forecast(
            4,
            conditions=[VariablePath(variable="y1", values=np.array([0.1]))],
            include_shock_uncertainty=False,
        )
        with pytest.raises(ValueError, match="needs a density forecast"):
            result.tilt([MomentTarget(variable="y2", horizon=2, mean=0.0)])

    def test_unknown_variable_raises_through_tilt(self, fitted_2v):
        forecast = fitted_2v.forecast(4, seed=0)
        with pytest.raises(ValueError, match="Unknown variable"):
            forecast.tilt([MomentTarget(variable="nope", horizon=1, mean=0.0)])

    def test_horizon_beyond_the_forecast_raises_through_tilt(self, fitted_2v):
        forecast = fitted_2v.forecast(4, seed=0)
        with pytest.raises(ValueError, match="only 4 steps"):
            forecast.tilt([MomentTarget(variable="y1", horizon=9, mean=0.0)])

    def test_ess_warn_fraction_must_be_a_fraction(self, fitted_2v):
        forecast = fitted_2v.forecast(4, seed=0)
        with pytest.raises(ValueError, match=r"ess_warn_fraction must lie in \[0, 1\]"):
            forecast.tilt([_median_target(forecast)], ess_warn_fraction=2.0)

    def test_scenario_result_inherits_tilt(self, fitted_2v):
        from impulso.identification import Cholesky

        identified = fitted_2v.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        scenario = identified.structural_scenario(4, seed=2)
        tilted = scenario.tilt([_median_target(scenario, horizon=2)])
        assert tilted.summary()["targets"][0]["achieved"] == pytest.approx(0.8, abs=1e-12)


class TestHardConditioningSurvivesTilting:
    def test_pins_hold_on_every_draw_after_reweighting(self, fitted_2v):
        from impulso.scenario import VariablePath

        pinned_path = np.array([0.2, 0.15, np.nan, np.nan])
        conditional = fitted_2v.conditional_forecast(
            4,
            conditions=[VariablePath(variable="y1", values=pinned_path)],
            seed=11,
        )
        before = conditional.idata.posterior_predictive["forecast"].sel(variable="y1").values
        assert np.allclose(before[:, :, 0], 0.2, atol=1e-10)

        tilted = conditional.tilt([_median_target(conditional, variable="y2", horizon=4, probability=0.75)])
        after = tilted.idata.posterior_predictive["forecast"].sel(variable="y1").values
        # Reweighting never moves a draw, so the pins are untouched.
        assert np.allclose(after[:, :, 0], 0.2, atol=1e-10)
        assert np.allclose(after[:, :, 1], 0.15, atol=1e-10)
        assert np.array_equal(before, after)
        # The tilted median at a pinned step is the pin itself.
        assert tilted.median()["y1"].iloc[0] == pytest.approx(0.2, abs=1e-10)

    def test_tilted_result_shares_the_parent_draws_by_reference(self, fitted_2v):
        forecast = fitted_2v.forecast(4, seed=3)
        tilted = forecast.tilt([_median_target(forecast)])
        parent = forecast.idata.posterior_predictive["forecast"].values
        child = tilted.idata.posterior_predictive["forecast"].values
        assert np.shares_memory(parent, child)


class TestDegeneracy:
    def test_far_tail_target_warns_and_reports_a_small_ess(self, fitted_2v):
        forecast = fitted_2v.forecast(4, seed=4)
        da = forecast.idata.posterior_predictive["forecast"].sel(variable="y1").isel(step=3).values
        # Keep only the most extreme few per cent of draws.
        threshold = float(np.quantile(da, 0.03))
        target = ProbabilityTarget(variable="y1", horizon=4, threshold=threshold, probability=1.0)
        with pytest.warns(UserWarning, match="tilt is concentrated"):
            tilted = forecast.tilt([target])
        assert tilted.summary()["ess_fraction"] < 0.1

    def test_no_warning_when_the_tilt_is_mild(self, fitted_2v):
        import warnings

        forecast = fitted_2v.forecast(4, seed=4)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            forecast.tilt([_median_target(forecast, probability=0.55)])


class TestTiltedResultSurface:
    def test_median_and_hdi_mirror_the_forecast_result_shapes(self, fitted_2v):
        forecast = fitted_2v.forecast(5, seed=5)
        tilted = forecast.tilt([_median_target(forecast, horizon=2)])
        assert tilted.median().shape == forecast.median().shape
        assert list(tilted.median().columns) == ["y1", "y2"]
        hdi = tilted.hdi(prob=0.5)
        assert hdi.lower.shape == (5, 2)
        assert hdi.upper.shape == (5, 2)
        assert np.all(hdi.upper.values >= hdi.lower.values)
        assert tilted.to_dataframe().equals(tilted.median())

    def test_uniform_tilt_reproduces_the_untilted_median(self, fitted_2v):
        forecast = fitted_2v.forecast(4, seed=6)
        da = forecast.idata.posterior_predictive["forecast"].sel(variable="y1").isel(step=1).values
        n_below = int((da < 0.0).sum())
        # Requesting exactly the empirical probability leaves weights uniform.
        target = ProbabilityTarget(variable="y1", horizon=2, threshold=0.0, probability=n_below / da.size)
        tilted = forecast.tilt([target])
        assert tilted.summary()["kl_divergence"] == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(tilted.median().values, tilted.base_median().values, atol=1e-12)

    def test_summary_surfaces_requested_achieved_and_event_counts(self, fitted_2v):
        forecast = fitted_2v.forecast(4, seed=7)
        prob_target = _median_target(forecast, horizon=2, probability=0.65)
        moment_target = MomentTarget(
            variable="y2",
            horizon=3,
            mean=float(np.mean(forecast.idata.posterior_predictive["forecast"].sel(variable="y2").isel(step=2))),
        )
        tilted = forecast.tilt([prob_target, moment_target])
        summary = tilted.summary()
        assert set(summary) == {"ess", "ess_fraction", "kl_divergence", "n_draws", "targets"}
        assert summary["n_draws"] == 100
        rows = summary["targets"]
        assert rows[0]["requested"] == pytest.approx(0.65, abs=1e-12)
        assert rows[0]["achieved"] == pytest.approx(0.65, abs=1e-8)
        assert rows[0]["draws_in_event"] == 50
        assert rows[1]["draws_in_event"] is None
        assert tilted.targets == [prob_target, moment_target]

    def test_weights_are_chain_draw_shaped_and_normalised(self, fitted_2v):
        forecast = fitted_2v.forecast(4, seed=8)
        tilted = forecast.tilt([_median_target(forecast)])
        assert tilted.weights.shape == (2, 50)
        assert tilted.weights.sum() == pytest.approx(1.0, abs=1e-12)

    def test_plot_returns_a_figure(self, fitted_2v):
        from matplotlib.figure import Figure

        forecast = fitted_2v.forecast(4, seed=9)
        tilted = forecast.tilt([_median_target(forecast)])
        assert isinstance(tilted.plot(), Figure)

    def test_result_is_frozen(self, fitted_2v):
        forecast = fitted_2v.forecast(4, seed=10)
        tilted = forecast.tilt([_median_target(forecast)])
        with pytest.raises(ValueError, match="frozen"):
            tilted.steps = 9
