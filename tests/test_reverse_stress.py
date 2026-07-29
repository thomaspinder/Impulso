"""Invariant tests for IdentifiedVAR.reverse_stress (issue #150).

Core identities: the base draws nest inside `structural_scenario` under a
matched seed (the RNG stream contract of `structural_forecast_draws`); the
cocktail at `probability=1.0` is exactly the plain mean of the retained
shocks; and with a deterministic posterior the cocktail reproduces the
textbook truncated-normal mean.
"""

import matplotlib

matplotlib.use("Agg")

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.figure import Figure

from impulso._scenario import structural_forecast_draws
from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.identification import Cholesky
from impulso.volatility import Constant


@pytest.fixture
def identified_2v(synthetic_idata_2v, var_data_2v):
    """Cholesky-identified 2-var VAR(1) from the synthetic posterior."""
    fitted = FittedVAR(
        idata=synthetic_idata_2v,
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )
    return fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))


def _deterministic_fitted(n_draws=4000):
    """A posterior whose draws all carry the same parameters.

    Every draw shares one `B`, `intercept` and `L`, so the only randomness
    left in a forecast is the structural shocks themselves — which turns
    the reverse-stress cocktail into a textbook truncated-normal mean.
    """
    n_vars = 2
    B = np.tile(np.array([[0.5, 0.1], [-0.2, 0.3]]), (1, n_draws, 1, 1))
    intercept = np.tile(np.array([0.1, -0.05]), (1, n_draws, 1))
    L_single = np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 0.8]]))
    L = np.tile(L_single, (1, n_draws, 1, 1))
    posterior = xr.Dataset({
        "B": (("chain", "draw", "var", "coeff"), B),
        "intercept": (("chain", "draw", "var"), intercept),
        "L": (("chain", "draw", "var1", "var2"), L),
    })
    data = VARData(
        endog=np.random.default_rng(1).standard_normal((12, n_vars)),
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


def _median_threshold(identified, steps, horizon, seed, variable_index=0):
    """Empirical median of the matched-seed forecast at `horizon`."""
    paths, _ = structural_forecast_draws(identified, steps, seed=seed)
    return float(np.median(paths[:, :, horizon - 1, variable_index])), paths


class TestMatchedSeedNesting:
    def test_base_draws_match_structural_scenario_under_a_shared_seed(self, identified_2v):
        paths, eps = structural_forecast_draws(identified_2v, 6, seed=123)
        scenario = identified_2v.structural_scenario(6, seed=123)
        assert np.allclose(paths, scenario.idata.posterior_predictive["forecast"].values, atol=1e-8)
        assert eps.shape == paths.shape

    def test_reverse_stress_carries_those_same_draws(self, identified_2v):
        threshold, paths = _median_threshold(identified_2v, 6, 6, seed=123)
        result = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=6, seed=123)
        assert np.allclose(result.idata.posterior_predictive["forecast"].values, paths, atol=1e-8)

    def test_shocks_reproduce_the_paths_through_the_scenario_engine(self, identified_2v):
        # The scenario engine with every shock prescribed at the drawn
        # values must reproduce the same paths, which pins the returned
        # eps to the ones that actually generated the forecast.
        from impulso.scenario import ShockPath

        paths, eps = structural_forecast_draws(identified_2v, 3, seed=7)
        prescribed = identified_2v.structural_scenario(
            3,
            shocks=[ShockPath(shock=name, values=eps[0, 0, :, j]) for j, name in enumerate(identified_2v.shock_names)],
            seed=7,
        )
        # Draw (0, 0) had exactly those shocks prescribed, so its path is unchanged.
        assert np.allclose(prescribed.idata.posterior_predictive["forecast"].values[0, 0], paths[0, 0], atol=1e-10)


class TestCocktail:
    def test_cocktail_is_the_plain_mean_of_the_retained_shocks(self, identified_2v):
        threshold, paths = _median_threshold(identified_2v, 5, 5, seed=42)
        _, eps = structural_forecast_draws(identified_2v, 5, seed=42)
        result = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=5, seed=42)

        n_total = paths.shape[0] * paths.shape[1]
        retained = paths[:, :, 4, 0].reshape(n_total) < threshold
        oracle = eps.reshape(n_total, 5, 2)[retained].mean(axis=0)
        assert np.allclose(result.shock_cocktail().values, oracle, atol=1e-14)

    def test_q_is_the_squared_norm_of_the_cocktail(self, identified_2v):
        threshold, _ = _median_threshold(identified_2v, 5, 5, seed=42)
        result = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=5, seed=42)
        cocktail = result.shock_cocktail().values
        assert result.summary()["q"] == pytest.approx(float(np.sum(cocktail**2)), rel=1e-13)

    def test_cocktail_is_labelled_by_shock_names(self, identified_2v):
        threshold, _ = _median_threshold(identified_2v, 4, 4, seed=5)
        result = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=4, seed=5)
        cocktail = result.shock_cocktail()
        assert list(cocktail.columns) == identified_2v.shock_names
        assert list(cocktail.index) == [1, 2, 3, 4]
        assert cocktail.index.name == "step"

    def test_soft_conditioning_shrinks_the_cocktail_toward_zero(self, identified_2v):
        threshold, _ = _median_threshold(identified_2v, 4, 4, seed=8)
        hard = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=4, seed=8)
        soft = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=4, probability=0.6, seed=8)
        assert soft.summary()["q"] < hard.summary()["q"]
        assert soft.summary()["ess"] > hard.summary()["ess"]


class TestProbabilities:
    def test_threshold_at_the_empirical_median_gives_a_half_baseline(self, identified_2v):
        threshold, _ = _median_threshold(identified_2v, 5, 5, seed=13)
        result = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=5, seed=13)
        summary = result.summary()
        assert summary["baseline_probability"] == pytest.approx(0.5, abs=0.02)
        assert summary["achieved_probability"] == pytest.approx(1.0, abs=1e-12)

    def test_a_certain_event_leaves_the_weights_uniform(self, identified_2v):
        result = identified_2v.reverse_stress(variable="y1", threshold=-1e6, steps=4, direction="above", seed=14)
        summary = result.summary()
        assert summary["baseline_probability"] == pytest.approx(1.0, abs=1e-12)
        assert summary["kl_divergence"] == pytest.approx(0.0, abs=1e-12)
        assert summary["q_cal"] == pytest.approx(0.5, abs=1e-12)
        assert summary["ess"] == pytest.approx(float(result.weights.size), rel=1e-12)
        # Uniform weights average all the shocks, so the cocktail is Monte
        # Carlo noise around zero rather than a stress configuration.
        _, eps = structural_forecast_draws(identified_2v, 4, seed=14)
        assert np.allclose(result.shock_cocktail().values, eps.mean(axis=(0, 1)), atol=1e-14)
        assert summary["q"] < 0.5

    def test_horizon_defaults_to_the_final_step(self, identified_2v):
        threshold, _ = _median_threshold(identified_2v, 5, 5, seed=15)
        default = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=5, seed=15)
        explicit = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=5, horizon=5, seed=15)
        assert default.horizon == 5
        assert np.allclose(default.shock_cocktail().values, explicit.shock_cocktail().values)


class TestSignSanity:
    """A deterministic posterior turns the cocktail into a textbook oracle."""

    def test_impact_shock_matches_the_truncated_normal_mean(self):
        fitted = _deterministic_fitted()
        identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        # With identical parameters across draws, y1 at step 1 is
        # b1 + P[0, 0] * eps_1[y1]; thresholding at b1 selects the draws
        # with a negative y1 shock on impact.
        b1 = float(fitted.forecast(4, include_shock_uncertainty=False).median()["y1"].iloc[0])
        result = identified.reverse_stress(variable="y1", threshold=b1, steps=4, horizon=1, seed=99)
        cocktail = result.shock_cocktail()

        assert cocktail.loc[1, "y1"] == pytest.approx(-np.sqrt(2.0 / np.pi), abs=0.05)
        # Under a natural-order Cholesky, y1 on impact loads on its own
        # shock only, so nothing else is selected.
        assert abs(cocktail.loc[1, "y2"]) < 0.1
        assert np.all(np.abs(cocktail.loc[2:].values) < 0.1)
        assert result.summary()["baseline_probability"] == pytest.approx(0.5, abs=0.02)

    def test_direction_above_flips_the_cocktail_sign(self):
        fitted = _deterministic_fitted()
        identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        b1 = float(fitted.forecast(4, include_shock_uncertainty=False).median()["y1"].iloc[0])
        result = identified.reverse_stress(variable="y1", threshold=b1, steps=4, horizon=1, direction="above", seed=99)
        assert result.shock_cocktail().loc[1, "y1"] == pytest.approx(np.sqrt(2.0 / np.pi), abs=0.05)


class TestGuards:
    def test_empty_support_raises_with_the_draw_counts(self, identified_2v):
        with pytest.raises(ValueError, match=r"0 of 100 draws satisfy"):
            identified_2v.reverse_stress(variable="y1", threshold=-1e6, steps=4, seed=1)

    def test_horizon_beyond_steps_raises(self, identified_2v):
        with pytest.raises(ValueError, match=r"horizon must lie in 1\.\.steps"):
            identified_2v.reverse_stress(variable="y1", threshold=0.0, steps=4, horizon=9, seed=1)

    def test_zero_horizon_raises(self, identified_2v):
        with pytest.raises(ValueError, match="1-based"):
            identified_2v.reverse_stress(variable="y1", threshold=0.0, steps=4, horizon=0, seed=1)

    def test_unknown_variable_raises(self, identified_2v):
        with pytest.raises(ValueError, match="Unknown variable"):
            identified_2v.reverse_stress(variable="nope", threshold=0.0, steps=4, seed=1)

    def test_probability_outside_the_unit_interval_raises(self, identified_2v):
        with pytest.raises(ValueError, match="0 < p <= 1"):
            identified_2v.reverse_stress(variable="y1", threshold=0.0, steps=4, probability=1.5, seed=1)

    def test_exog_future_required_when_the_model_carries_exogenous_data(self, var_data_2v, synthetic_idata_2v):
        data = VARData(
            endog=var_data_2v.endog,
            endog_names=["y1", "y2"],
            index=var_data_2v.index,
            exog=np.ones((len(var_data_2v.index), 1)),
            exog_names=["const"],
        )
        posterior = synthetic_idata_2v.posterior.copy()
        posterior["B_exog"] = xr.DataArray(
            np.zeros((2, 50, 2, 1)),
            dims=["chain", "draw", "var", "exog"],
        )
        fitted = FittedVAR(
            idata=az.InferenceData(posterior=posterior),
            n_lags=1,
            data=data,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        with pytest.raises(ValueError, match="exog_future is required"):
            identified.reverse_stress(variable="y1", threshold=0.0, steps=4, seed=1)

    def test_ess_warning_propagates(self, identified_2v):
        paths, _ = structural_forecast_draws(identified_2v, 4, seed=21)
        threshold = float(np.quantile(paths[:, :, 3, 0], 0.03))
        with pytest.warns(UserWarning, match="tilt is concentrated"):
            result = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=4, seed=21)
        assert result.summary()["ess_fraction"] < 0.1


class TestResultSurface:
    def test_summary_keys_are_pinned(self, identified_2v):
        threshold, _ = _median_threshold(identified_2v, 4, 4, seed=31)
        result = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=4, seed=31)
        assert set(result.summary()) == {
            "baseline_probability",
            "requested_probability",
            "achieved_probability",
            "ess",
            "ess_fraction",
            "kl_divergence",
            "n_draws",
            "q",
            "q_cal",
        }

    def test_median_and_hdi_are_weighted(self, identified_2v):
        threshold, _ = _median_threshold(identified_2v, 4, 4, seed=32)
        result = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=4, seed=32)
        med = result.median()
        assert med.shape == (4, 2)
        assert list(med.columns) == ["y1", "y2"]
        # Conditioning on y1 being low pushes its conditioned median below
        # the untilted one at the conditioning horizon.
        assert med["y1"].iloc[3] < result.base_median()["y1"].iloc[3]
        hdi = result.hdi(prob=0.5)
        assert np.all(hdi.upper.values >= hdi.lower.values)
        assert result.to_dataframe().equals(med)

    def test_structural_shocks_are_carried_on_the_result(self, identified_2v):
        threshold, _ = _median_threshold(identified_2v, 4, 4, seed=33)
        result = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=4, seed=33)
        shocks = result.idata.posterior_predictive["structural_shocks"]
        assert shocks.dims == ("chain", "draw", "step", "shock")
        assert list(shocks.coords["shock"].values) == identified_2v.shock_names

    def test_plot_returns_a_figure(self, identified_2v):
        threshold, _ = _median_threshold(identified_2v, 4, 4, seed=34)
        result = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=4, seed=34)
        assert isinstance(result.plot(), Figure)

    def test_result_is_frozen(self, identified_2v):
        threshold, _ = _median_threshold(identified_2v, 4, 4, seed=35)
        result = identified_2v.reverse_stress(variable="y1", threshold=threshold, steps=4, seed=35)
        with pytest.raises(ValueError, match="frozen"):
            result.threshold = 0.0
