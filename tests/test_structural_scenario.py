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
from impulso._scenario import _resolve_adjusting, structural_forecast_draws
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


class _RngConsumingVol:
    """Time-varying fake whose forecast Cholesky path consumes the generator."""

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


class TestTimeVaryingVolatility:
    @pytest.fixture
    def pair_sv(self, synthetic_idata_2v, var_data_2v):
        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=_RngConsumingVol(),
        )
        identified = IdentifiedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=_RngConsumingVol(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        return fitted, identified

    def test_per_slice_identify_happy_path_pins_hold(self, pair_sv):
        """The per-slice identify branch runs and pins hold per draw."""
        _, identified = pair_sv
        scn = identified.structural_scenario(steps=4, conditions=[VariablePath(variable="y1", values=0.4)], seed=3)
        draws = scn.idata.posterior_predictive["forecast"].values
        np.testing.assert_allclose(draws[:, :, :, 0], 0.4, atol=1e-8)

    def test_matched_seed_nesting_under_generator_consuming_volatility(self, pair_sv):
        """The rng stream order through _forecast_shock_matrices matches PR3's engine."""
        fitted, identified = pair_sv
        conditions = [VariablePath(variable="y1", values=0.4)]
        scn = identified.structural_scenario(steps=5, conditions=conditions, seed=17)
        cf = fitted.conditional_forecast(steps=5, conditions=conditions, seed=17)
        np.testing.assert_allclose(
            scn.idata.posterior_predictive["forecast"].values,
            cf.idata.posterior_predictive["forecast"].values,
            atol=1e-8,
        )

    def test_structural_forecast_draws_nest_under_generator_consuming_volatility(self, pair_sv):
        """structural_forecast_draws holds its RNG contract when the volatility path draws too.

        `_forecast_shock_matrices` consumes the generator only under
        time-varying volatility, so this is the branch where a mis-ordered
        stream would silently desynchronise the two code paths.
        """
        _, identified = pair_sv
        paths, eps = structural_forecast_draws(identified, 5, seed=23)
        scn = identified.structural_scenario(steps=5, seed=23)
        np.testing.assert_allclose(paths, scn.idata.posterior_predictive["forecast"].values, atol=1e-12)
        assert eps.shape == paths.shape


def _single_draw_identified_exog():
    """Single-draw exog posterior, Cholesky-identified."""
    import arviz as az
    import pandas as pd
    import xarray as xr

    from impulso.data import VARData

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
    fitted = FittedVAR(
        idata=az.InferenceData(posterior=posterior),
        n_lags=1,
        data=data,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )
    return fitted, fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))


class TestExog:
    def test_matched_seed_nesting_with_exog(self):
        fitted, identified = _single_draw_identified_exog()
        exog_future = np.ones((5, 1))
        conditions = [VariablePath(variable="y1", values=0.3)]
        scn = identified.structural_scenario(steps=5, conditions=conditions, seed=9, exog_future=exog_future)
        cf = fitted.conditional_forecast(steps=5, conditions=conditions, seed=9, exog_future=exog_future)
        np.testing.assert_allclose(
            scn.idata.posterior_predictive["forecast"].values,
            cf.idata.posterior_predictive["forecast"].values,
            atol=1e-8,
        )

    def test_exog_shifts_unpinned_path_while_pin_holds(self):
        _, identified = _single_draw_identified_exog()
        pin = [VariablePath(variable="y1", values=0.3)]
        low = identified.structural_scenario(
            steps=4, conditions=pin, include_shock_uncertainty=False, exog_future=np.zeros((4, 1))
        )
        high = identified.structural_scenario(
            steps=4, conditions=pin, include_shock_uncertainty=False, exog_future=np.ones((4, 1))
        )
        low_draws = low.idata.posterior_predictive["forecast"].values
        high_draws = high.idata.posterior_predictive["forecast"].values
        np.testing.assert_allclose(low_draws[:, :, :, 0], 0.3, atol=1e-8)
        np.testing.assert_allclose(high_draws[:, :, :, 0], 0.3, atol=1e-8)
        assert np.abs(low_draws[:, :, :, 1] - high_draws[:, :, :, 1]).max() > 1e-3

    def test_exog_guard_errors(self, identified_2v):
        _, identified_exog = _single_draw_identified_exog()
        with pytest.raises(ValueError, match="required"):
            identified_exog.structural_scenario(steps=4)
        with pytest.raises(ValueError, match="shape"):
            identified_exog.structural_scenario(steps=4, exog_future=np.zeros((3, 1)))
        with pytest.raises(ValueError, match="no B_exog"):
            identified_2v.structural_scenario(steps=4, exog_future=np.zeros((4, 1)))


class _PartialScheme:
    """Fake partial identification: L as-is, first column labelled identified."""

    def identify(self, L, var_names, posterior=None, data=None, n_lags=None):
        del var_names, posterior, data, n_lags
        return L

    def shock_coords(self, n_vars: int) -> list[str]:
        return ["target"] + [f"unidentified_{i}" for i in range(n_vars - 1)]


class TestPartialIdentification:
    @pytest.fixture
    def identified_partial(self, synthetic_idata_2v, var_data_2v):
        return IdentifiedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=_PartialScheme(),
        )

    def test_prescribing_unidentified_errors(self, identified_partial):
        with pytest.raises(ValueError, match="Cannot prescribe"):
            identified_partial.structural_scenario(steps=3, shocks=[ShockPath(shock="unidentified_0", values=1.0)])

    def test_end_to_end_prescribe_identified_column(self, identified_partial):
        """Prescribe the identified column; the unidentified block absorbs a pin."""
        scn = identified_partial.structural_scenario(
            steps=3,
            conditions=[VariablePath(variable="y2", values=np.array([0.4]))],
            shocks=[ShockPath(shock="target", values=1.0)],
            seed=4,
        )
        draws = scn.idata.posterior_predictive["forecast"].values
        np.testing.assert_allclose(draws[:, :, 0, 1], 0.4, atol=1e-8)
        assert scn.idata.posterior_predictive["plausibility"].values.min() >= 3.0 - 1e-10


class TestNumericalGuards:
    def test_near_collinear_adjusting_block_warns(self):
        """cond(C_A C_A') check fires where the full-Gram check cannot."""
        import arviz as az
        import pandas as pd
        import xarray as xr

        from impulso.data import VARData

        rng = np.random.default_rng(0)
        B = np.array([[[[0.5, 1e-6], [1e-6, 0.5]]]])
        posterior = xr.Dataset({
            "B": (("chain", "draw", "var", "coeff"), B),
            "intercept": (("chain", "draw", "var"), np.zeros((1, 1, 2))),
            "L": (("chain", "draw", "var1", "var2"), np.eye(2)[np.newaxis, np.newaxis]),
        })
        data = VARData(
            endog=rng.standard_normal((10, 2)),
            endog_names=["y1", "y2"],
            index=pd.date_range("2000-01-01", periods=10, freq="QS"),
        )
        fitted = FittedVAR(
            idata=az.InferenceData(posterior=posterior),
            n_lags=1,
            data=data,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        with pytest.warns(UserWarning, match="adjusting-block"):
            identified.structural_scenario(
                steps=2,
                conditions=[
                    VariablePath(variable="y1", values=np.array([np.nan, 0.5])),
                    VariablePath(variable="y2", values=np.array([np.nan, 0.5])),
                ],
                adjusting=["y1"],
                include_shock_uncertainty=False,
            )

    def test_scale_normalisation_warns_on_prescriptions(self, synthetic_idata_2v, var_data_2v):
        class _ScaledScheme:
            scale = 10.0

            def identify(self, L, var_names, posterior=None, data=None, n_lags=None):
                return L

            def shock_coords(self, n_vars):
                return ["y1", "y2"]

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=_ScaledScheme(),
        )
        with pytest.warns(UserWarning, match="one-standard-deviation"):
            identified.structural_scenario(
                steps=3, shocks=[ShockPath(shock="y1", values=2.0)], adjusting=[], include_shock_uncertainty=False
            )

    def test_chi2_tail_uses_condition_only_q(self, identified_2v):
        """The chi^2_r reference excludes the prescribed |v_S|^2 term."""
        from scipy.stats import chi2

        scn = identified_2v.structural_scenario(
            steps=3,
            conditions=[VariablePath(variable="y1", values=np.array([0.4]))],
            shocks=[ShockPath(shock="y2", values=2.0)],
            adjusting=["y1"],
            include_shock_uncertainty=False,
        )
        pp = scn.idata.posterior_predictive
        q_cond = pp["plausibility"].values - 3 * 4.0  # subtract |v_S|^2
        expected = float(chi2.sf(float(np.median(q_cond)), df=1))
        assert pp.attrs["chi2_tail_of_median"] == pytest.approx(expected)


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


class TestHeavyTailedErrorsRejected:
    """structural_scenario is Gaussian-only (issue #152).

    The ADPRR three-way partition draws the adjusting block from its
    Gaussian conditional law, and the plausibility statistic's chi-squared
    reference assumes Gaussian shocks.
    """

    @pytest.fixture
    def identified_2v_t(self, synthetic_idata_2v_t, var_data_2v):
        from impulso.observation import StudentT

        fitted = FittedVAR(
            idata=synthetic_idata_2v_t,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            error_dist=StudentT(nu=5.0),
        )
        return fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))

    def test_error_dist_survives_identification(self, identified_2v_t):
        assert identified_2v_t.error_dist.is_heavy_tailed

    @pytest.mark.parametrize("include_shock_uncertainty", [True, False])
    @pytest.mark.parametrize("conditions", [None, [VariablePath(variable="y1", values=0.2)]])
    def test_raises_not_implemented(self, identified_2v_t, include_shock_uncertainty, conditions):
        with pytest.raises(NotImplementedError, match="Gaussian-only"):
            identified_2v_t.structural_scenario(
                steps=4,
                conditions=conditions,
                include_shock_uncertainty=include_shock_uncertainty,
                seed=1,
            )

    def test_raises_with_shock_prescriptions_too(self, identified_2v_t):
        with pytest.raises(NotImplementedError, match="Gaussian-only"):
            identified_2v_t.structural_scenario(steps=4, shocks=[ShockPath(shock="y1", values=1.0)], seed=1)

    def test_error_names_the_alternatives(self, identified_2v_t):
        with pytest.raises(NotImplementedError, match=r"counterfactual\(\)"):
            identified_2v_t.structural_scenario(steps=4)
        with pytest.raises(NotImplementedError, match="error_dist='gaussian'"):
            identified_2v_t.structural_scenario(steps=4)

    def test_gaussian_sibling_still_works(self, identified_2v):
        result = identified_2v.structural_scenario(steps=4, seed=1)
        assert result.median().shape == (4, 2)
