"""Invariant tests for IdentifiedVAR.structural_scenario (PR 4 of the scenario stack).

Core identities (design + ADR-0005): nesting to conditional_forecast under
adjusting=all with no prescriptions (exact per draw, Cholesky + matched
seed); the substitution edge (prescribing every shock equals deterministic
propagation); two-tier feasibility; the combined flavour; and the
prescribed-shock plausibility term.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from impulso._linalg import lag_matrices
from impulso._propagate import propagate
from impulso._scenario import _resolve_adjusting
from impulso.fitted import FittedVAR
from impulso.identification import Cholesky, SignRestriction
from impulso.identified import IdentifiedVAR
from impulso.results import ScenarioResult
from impulso.scenario import ShockPath, VariablePath
from impulso.volatility import Constant


@pytest.fixture
def fitted_2v(synthetic_idata_2v, var_data_2v):
    return FittedVAR(
        idata=synthetic_idata_2v,
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )


@pytest.fixture
def identified_2v(fitted_2v):
    return fitted_2v.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))


class TestConditionalForecastNesting:
    def test_all_adjusting_no_prescriptions_matches_conditional_forecast(self, identified_2v, fitted_2v):
        """Invariant 6: exact per draw under Cholesky identification + matched seed."""
        conditions = [VariablePath(variable="y1", values=0.4)]
        scn = identified_2v.structural_scenario(steps=5, conditions=conditions, seed=21)
        cf = fitted_2v.conditional_forecast(steps=5, conditions=conditions, seed=21)
        np.testing.assert_allclose(
            scn.idata.posterior_predictive["forecast"].values,
            cf.idata.posterior_predictive["forecast"].values,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            scn.idata.posterior_predictive["plausibility"].values,
            cf.idata.posterior_predictive["plausibility"].values,
            atol=1e-10,
        )

    def test_no_ingredients_matches_forecast_nesting(self, identified_2v, fitted_2v):
        scn = identified_2v.structural_scenario(steps=6, seed=7)
        fc = fitted_2v.forecast(steps=6, seed=7)
        np.testing.assert_allclose(
            scn.idata.posterior_predictive["forecast"].values,
            fc.idata.posterior_predictive["forecast"].values,
            atol=1e-8,
        )


class TestSubstitutionEdge:
    def test_prescribing_every_shock_is_deterministic_propagation(self, identified_2v, synthetic_idata_2v):
        """Invariant 8: full prescription with an empty adjusting set, no solve."""
        steps = 4
        v1 = 0.1 * np.arange(1, steps + 1)
        v2 = -0.05 * np.arange(1, steps + 1)
        kwargs = {
            "steps": steps,
            "shocks": [ShockPath(shock="y1", values=v1), ShockPath(shock="y2", values=v2)],
            "adjusting": [],
        }
        density = identified_2v.structural_scenario(**kwargs, seed=3)
        mean = identified_2v.structural_scenario(**kwargs, include_shock_uncertainty=False)
        # Deterministic given the posterior draw: density mode equals mean mode.
        np.testing.assert_allclose(
            density.idata.posterior_predictive["forecast"].values,
            mean.idata.posterior_predictive["forecast"].values,
            atol=1e-12,
        )

        # Hand propagation: y = b + propagate(P eps).
        posterior = synthetic_idata_2v.posterior
        B = posterior["B"].values
        intercept = posterior["intercept"].values
        P = identified_2v.shock_matrix(at=None).values
        n_chains, n_draws, n, _ = B.shape
        eps = np.stack([v1, v2], axis=-1)  # (steps, n)
        u = np.einsum("cdij,hj->cdhi", P, eps)
        A = lag_matrices(B, 1)
        forcing = np.broadcast_to(intercept[:, :, np.newaxis, :], (n_chains, n_draws, steps, n)).copy()
        b = propagate(A, forcing, identified_2v.data.endog[-1:])
        expected = b + propagate(A, u, np.zeros((1, n)))
        np.testing.assert_allclose(density.idata.posterior_predictive["forecast"].values, expected, atol=1e-8)


class TestCombinedFlavours:
    def test_pin_holds_while_other_shock_is_prescribed(self, identified_2v):
        scn = identified_2v.structural_scenario(
            steps=4,
            conditions=[VariablePath(variable="y1", values=0.5)],
            shocks=[ShockPath(shock="y2", values=1.0)],
            adjusting=["y1"],
            seed=2,
        )
        draws = scn.idata.posterior_predictive["forecast"].values
        np.testing.assert_allclose(draws[:, :, :, 0], 0.5, atol=1e-8)
        # Prescribed magnitude registers in the plausibility statistic.
        assert scn.idata.posterior_predictive["plausibility"].values.min() >= 4.0 - 1e-10

    def test_prescription_changes_the_path(self, identified_2v):
        base = identified_2v.structural_scenario(
            steps=4, shocks=[ShockPath(shock="y2", values=0.0)], adjusting=[], include_shock_uncertainty=False
        )
        pushed = identified_2v.structural_scenario(
            steps=4, shocks=[ShockPath(shock="y2", values=2.0)], adjusting=[], include_shock_uncertainty=False
        )
        diff = np.abs(
            base.idata.posterior_predictive["forecast"].values - pushed.idata.posterior_predictive["forecast"].values
        )
        assert diff.max() > 0.1


class TestPlausibility:
    def test_prescription_only_q_is_squared_magnitude(self, identified_2v):
        scn = identified_2v.structural_scenario(
            steps=3, shocks=[ShockPath(shock="y1", values=2.0)], adjusting=[], include_shock_uncertainty=False
        )
        pp = scn.idata.posterior_predictive
        np.testing.assert_allclose(pp["plausibility"].values, 3 * 4.0, atol=1e-12)
        np.testing.assert_array_equal(pp["plausibility_calibrated"].values, 1.0)

    def test_ray_monotone_in_expectation_under_proper_adjusting(self, identified_2v):
        """Invariant 9 (adjusting part): monotone in expectation over free draws."""

        def mean_q(value: float) -> float:
            scn = identified_2v.structural_scenario(
                steps=3,
                conditions=[VariablePath(variable="y1", values=value)],
                adjusting=["y1"],
                seed=5,
            )
            return float(scn.idata.posterior_predictive["plausibility"].values.mean())

        assert mean_q(3.0) > mean_q(1.0)


class TestFeasibility:
    def test_over_determination_errors_at_validation(self, identified_2v):
        with pytest.raises(ValueError, match="adjusting capacity"):
            identified_2v.structural_scenario(
                steps=4,
                conditions=[VariablePath(variable="y1", values=0.5)],
                shocks=[ShockPath(shock="y1", values=1.0)],
                adjusting=["y1"],
            )

    def test_leading_block_counting_errors(self, identified_2v):
        with pytest.raises(ValueError, match="through step 1"):
            identified_2v.structural_scenario(
                steps=4,
                conditions=[
                    VariablePath(variable="y1", values=np.array([0.5])),
                    VariablePath(variable="y2", values=np.array([0.5])),
                ],
                adjusting=["y1"],
            )

    def test_per_draw_rank_failure_errors(self, identified_2v):
        """Cholesky zero: variable 1 at step 1 loads on no adjusting shock 2 entry."""
        with pytest.raises(ValueError, match="rank-deficient"):
            identified_2v.structural_scenario(
                steps=2,
                conditions=[VariablePath(variable="y1", values=np.array([0.5]))],
                adjusting=["y2"],
            )


class TestGuards:
    def test_in_sample_window_on_prescription_errors(self, identified_2v, var_data_2v):
        with pytest.raises(ValueError, match="in-sample-only"):
            identified_2v.structural_scenario(
                steps=4,
                shocks=[ShockPath(shock="y1", values=0.0, start=var_data_2v.index[5])],
            )

    def test_unknown_adjusting_shock_errors(self, identified_2v):
        with pytest.raises(ValueError, match="Unknown adjusting shock"):
            identified_2v.structural_scenario(steps=4, adjusting=["oil"])

    def test_duplicate_prescriptions_error(self, identified_2v):
        with pytest.raises(ValueError, match="Duplicate prescription"):
            identified_2v.structural_scenario(
                steps=4,
                shocks=[
                    ShockPath(shock="y1", values=1.0),
                    ShockPath(shock="y1", values=np.array([0.5])),
                ],
            )

    def test_unidentified_adjusting_none_or_all_rule(self):
        names = ["target", "unidentified_0", "unidentified_1"]
        assert _resolve_adjusting(["target"], names) == [0]
        assert _resolve_adjusting(["unidentified_0", "unidentified_1"], names) == [1, 2]
        with pytest.raises(ValueError, match="none or all"):
            _resolve_adjusting(["unidentified_0"], names)

    def test_sign_restriction_under_time_varying_volatility_errors(self, synthetic_idata_2v, var_data_2v):
        class _TimeVaryingStub:
            name = "fake-sv"
            is_time_varying = True

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=_TimeVaryingStub(),
            scheme=SignRestriction.model_construct(restrictions={"y1": {"demand": "+", "supply": "+"}}),
        )
        with pytest.raises(ValueError, match="SignRestriction"):
            identified.structural_scenario(steps=3, conditions=[VariablePath(variable="y1", values=0.1)])


class TestPathUncertainty:
    def test_unconditional_mode_keeps_spread_while_mean_binds(self, identified_2v):
        conditions = [VariablePath(variable="y1", values=0.4)]
        hard = identified_2v.structural_scenario(steps=4, conditions=conditions, seed=9)
        soft = identified_2v.structural_scenario(
            steps=4, conditions=conditions, seed=9, path_uncertainty="unconditional"
        )
        hard_draws = hard.idata.posterior_predictive["forecast"].values
        soft_draws = soft.idata.posterior_predictive["forecast"].values
        np.testing.assert_allclose(hard_draws[:, :, :, 0], 0.4, atol=1e-8)
        assert soft_draws[:, :, 0, 0].std() > 10 * hard_draws[:, :, 0, 0].std()


class TestResultSurface:
    def test_scenario_result_fields_and_plot(self, identified_2v):
        scn = identified_2v.structural_scenario(
            steps=4,
            conditions=[VariablePath(variable="y1", values=0.2)],
            adjusting=["y1", "y2"],
            seed=1,
        )
        assert isinstance(scn, ScenarioResult)
        assert scn.adjusting == ["y1", "y2"]
        assert scn.median().shape == (4, 2)
        summary = scn.plausibility()
        assert summary["n_restrictions"] == 4
        fig = scn.plot()
        assert isinstance(fig, Figure)
        title = fig._suptitle.get_text()
        assert "Structural Scenario" in title
        assert "adjusting: y1, y2" in title
