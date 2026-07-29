"""Tests for Bayesian Granger causality and the Toda-Yamamoto mode.

The load-bearing group is `TestIndexing`. Everything else in the feature —
directionality, calibration, the augmentation contract — is downstream of
reading the right columns out of the stacked coefficient matrix `B`, and a
lag-major/variable-major slip there is silent: it still returns plausible
numbers, for the wrong pair. Those tests therefore run against a hand-built
posterior whose every entry is distinct, so any slip changes the answer.
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from impulso import GrangerCausalityResult, VARData, toda_yamamoto
from impulso._granger import _coefficient_indices
from impulso._linalg import lag_matrices
from impulso.conjugate import ConjugateVAR
from impulso.fitted import FittedVAR
from impulso.priors import NIWPrior
from impulso.results import IntegrationOrderResult
from impulso.volatility import Constant

# --------------- helpers ---------------


def _var_data(endog: np.ndarray, names: list[str]) -> VARData:
    """Wrap a raw array in VARData with a monthly index."""
    index = pd.date_range("2000-01-31", periods=endog.shape[0], freq="ME")
    return VARData(endog=endog, endog_names=names, index=index)


def _hand_built_fitted(B_matrix: np.ndarray, *, seed: int = 3, T: int = 60) -> FittedVAR:
    """FittedVAR whose posterior is one constant, hand-chosen `B`.

    No MCMC, no estimation: every draw carries exactly `B_matrix`, so the
    extracted coefficients can be asserted against literal values. The
    endogenous data is random (only its per-column standard deviations
    matter, for the standardisation tests).
    """
    n_vars = B_matrix.shape[0]
    n_lags = B_matrix.shape[1] // n_vars
    B = np.broadcast_to(B_matrix, (2, 10, *B_matrix.shape)).copy()
    posterior = xr.Dataset({
        "B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"]),
        "intercept": xr.DataArray(np.zeros((2, 10, n_vars)), dims=["chain", "draw", "var"]),
    })
    rng = np.random.default_rng(seed)
    # Distinct per-column scales so a swapped standardisation ratio shows up.
    endog = rng.standard_normal((T, n_vars)) * np.arange(1, n_vars + 1)
    data = _var_data(endog, [f"y{i + 1}" for i in range(n_vars)])
    return FittedVAR(
        idata=az.InferenceData(posterior=posterior),
        n_lags=n_lags,
        data=data,
        var_names=list(data.endog_names),
        volatility=Constant(),
    )


# Lag-major layout: columns are L1.y1, L1.y2, L2.y1, L2.y2. Every entry is
# distinct, so picking the wrong row, lag, or variable cannot go unnoticed.
B_2V_2L = np.array([
    [0.11, 0.12, 0.13, 0.14],
    [0.21, 0.22, 0.23, 0.24],
])

# 3 variables, 2 lags: L1.y1, L1.y2, L1.y3, L2.y1, L2.y2, L2.y3.
B_3V_2L = np.array([
    [0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
    [0.21, 0.22, 0.23, 0.24, 0.25, 0.26],
    [0.31, 0.32, 0.33, 0.34, 0.35, 0.36],
])


@pytest.fixture
def hand_built():
    """2-variable, 2-lag hand-built posterior."""
    return _hand_built_fitted(B_2V_2L)


# --------------- data-generating processes ---------------


def _unidirectional_data(seed: int = 0, T: int = 300) -> VARData:
    """y2 Granger-causes y1; y1 never feeds back into y2.

    y1_t = 0.4 y1_{t-1} + 0.4 y2_{t-1} + 0.1 e1_t
    y2_t =                0.5 y2_{t-1} + 0.1 e2_t
    """
    rng = np.random.default_rng(seed)
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t, 0] = 0.4 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + 0.1 * rng.standard_normal()
        y[t, 1] = 0.5 * y[t - 1, 1] + 0.1 * rng.standard_normal()
    return _var_data(y, ["y1", "y2"])


def _null_data(seed: int, T: int = 200) -> VARData:
    """Two independent AR(1)s with rho = 0.5 — no causality either way."""
    rng = np.random.default_rng(seed)
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + 0.1 * rng.standard_normal(2)
    return _var_data(y, ["y1", "y2"])


def _i1_unidirectional(seed: int = 7, T: int = 400) -> VARData:
    """Both series I(1); x's increments drive y's, never the reverse.

    x is a driftless random walk; y accumulates 0.5 times x's increment
    plus its own small noise. Seed chosen so that `integration_order`
    settles both series at d = 1 with nothing left inconclusive.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(T))
    y = np.zeros(T)
    for t in range(2, T):
        y[t] = y[t - 1] + 0.5 * (x[t - 1] - x[t - 2]) + 0.05 * rng.standard_normal()
    return _var_data(np.column_stack([x, y]), ["x", "y"])


# --------------- 1. indexing (load-bearing) ---------------


class TestIndexing:
    def test_helper_strides_by_n_vars_from_the_cause_column(self):
        # Lag-major: lag k's block starts at column k * n_vars.
        assert _coefficient_indices(2, 1, 2) == [1, 3]
        assert _coefficient_indices(3, 2, 2) == [2, 5]
        assert _coefficient_indices(3, 0, 3) == [0, 3, 6]

    def test_helper_matches_the_coeff_coordinate_labels(self):
        # The posterior's `coeff` coord is built lag-major in spec.py as
        # [f"L{lag}.{name}" for lag in 1..p for name in var_names]; the
        # indices must select exactly the cause's labels, in lag order.
        names = ["y1", "y2", "y3"]
        coeff = [f"L{lag}.{name}" for lag in (1, 2) for name in names]
        indices = _coefficient_indices(len(names), names.index("y2"), 2)
        assert [coeff[i] for i in indices] == ["L1.y2", "L2.y2"]

    def test_extracts_the_cause_columns_of_the_effect_equation(self, hand_built):
        result = hand_built.granger_causality("y2", "y1", standardize=False)
        assert result.coef_draws.shape == (2, 10, 2)
        # Row y1 (effect), columns L1.y2 and L2.y2.
        np.testing.assert_allclose(result.coef_draws, np.broadcast_to([0.12, 0.14], (2, 10, 2)))

    def test_reverse_direction_reads_the_other_equation(self, hand_built):
        result = hand_built.granger_causality("y1", "y2", standardize=False)
        # Row y2 (effect), columns L1.y1 and L2.y1.
        np.testing.assert_allclose(result.coef_draws, np.broadcast_to([0.21, 0.23], (2, 10, 2)))

    def test_test_lags_keeps_the_leading_lags_only(self, hand_built):
        result = hand_built.granger_causality("y2", "y1", standardize=False, test_lags=1)
        np.testing.assert_allclose(result.coef_draws, np.broadcast_to([0.12], (2, 10, 1)))
        assert result.n_lags_tested == 1
        assert result.n_lags_fitted == 2
        assert result.augmentation == 1
        assert result.augmentation_source == "user"

    def test_agrees_with_the_lag_matrices_split(self, hand_built):
        # Drift-proofing: `lag_matrices` is the other consumer of the same
        # layout, so the two must never disagree about which entry is
        # A_k[effect, cause].
        B = hand_built.idata.posterior["B"].values
        expected = np.stack([A[..., 0, 1] for A in lag_matrices(B, 2)], axis=-1)
        result = hand_built.granger_causality("y2", "y1", standardize=False)
        np.testing.assert_allclose(result.coef_draws, expected)

    def test_transposed_posterior_is_realigned_by_name(self, hand_built):
        # Hand-built posteriors may order their dims arbitrarily; the
        # canonical labels are enough to put them back.
        transposed = hand_built.idata.posterior["B"].transpose("var", "coeff", "chain", "draw")
        hand_built.idata.posterior["B"] = transposed
        result = hand_built.granger_causality("y2", "y1", standardize=False)
        np.testing.assert_allclose(result.coef_draws, np.broadcast_to([0.12, 0.14], (2, 10, 2)))

    def test_unlabelled_posterior_falls_back_to_the_positional_convention(self):
        # No canonical dim names at all — trust (chain, draw, var, coeff),
        # the same contract as `dynamic_multiplier`.
        fitted = _hand_built_fitted(B_2V_2L)
        values = fitted.idata.posterior["B"].values
        fitted.idata.posterior["B"] = xr.DataArray(values, dims=["a", "b", "c", "d"])
        result = fitted.granger_causality("y2", "y1", standardize=False)
        np.testing.assert_allclose(result.coef_draws, np.broadcast_to([0.12, 0.14], (2, 10, 2)))

    def test_three_variable_system_picks_the_right_pair(self):
        fitted = _hand_built_fitted(B_3V_2L)
        result = fitted.granger_causality("y3", "y2", standardize=False)
        # Row y2, columns L1.y3 (index 2) and L2.y3 (index 5).
        np.testing.assert_allclose(result.coef_draws, np.broadcast_to([0.23, 0.26], (2, 10, 2)))

        expected = np.stack([A[..., 1, 2] for A in lag_matrices(fitted.idata.posterior["B"].values, 2)], axis=-1)
        np.testing.assert_allclose(result.coef_draws, expected)


# --------------- 2. validation ---------------


class TestValidation:
    def test_unknown_cause_lists_the_model_variables(self, hand_built):
        with pytest.raises(ValueError, match=r"unknown variable\(s\) \['gdp'\]"):
            hand_built.granger_causality("gdp", "y1")

    def test_unknown_effect_is_reported_too(self, hand_built):
        with pytest.raises(ValueError, match="y1', 'y2'"):
            hand_built.granger_causality("y1", "inflation")

    def test_cause_equal_to_effect_is_refused(self, hand_built):
        with pytest.raises(ValueError, match="must be different variables"):
            hand_built.granger_causality("y1", "y1")

    @pytest.mark.parametrize("test_lags", [0, -1, 3])
    def test_test_lags_outside_the_fitted_order_is_refused(self, hand_built, test_lags):
        with pytest.raises(ValueError, match=r"test_lags must lie in \[1, 2\]"):
            hand_built.granger_causality("y2", "y1", test_lags=test_lags)

    @pytest.mark.parametrize("rope", [0.0, -0.1])
    def test_non_positive_rope_is_refused(self, hand_built, rope):
        with pytest.raises(ValueError, match="rope must be positive"):
            hand_built.granger_causality("y2", "y1", rope=rope)


# --------------- 3. standardisation ---------------


class TestStandardisation:
    def test_scale_is_sd_cause_over_sd_effect(self, hand_built):
        endog = np.asarray(hand_built.data.endog)
        expected = endog[:, 1].std(ddof=1) / endog[:, 0].std(ddof=1)
        result = hand_built.granger_causality("y2", "y1")
        assert result.scale == pytest.approx(expected)
        np.testing.assert_allclose(result.coef_draws, np.broadcast_to([0.12, 0.14], (2, 10, 2)) * expected)

    def test_reverse_direction_inverts_the_ratio(self, hand_built):
        forward = hand_built.granger_causality("y2", "y1").scale
        reverse = hand_built.granger_causality("y1", "y2").scale
        assert forward * reverse == pytest.approx(1.0)

    def test_disabling_standardisation_leaves_the_raw_draws(self, hand_built):
        result = hand_built.granger_causality("y2", "y1", standardize=False)
        assert result.standardize is False
        assert result.scale == 1.0
        np.testing.assert_allclose(result.coef_draws, np.broadcast_to([0.12, 0.14], (2, 10, 2)))


# --------------- 4. result surface ---------------


class TestResultSurface:
    def test_summary_index_is_the_lags_then_the_norm(self, hand_built):
        summary = hand_built.granger_causality("y2", "y1", standardize=False).summary()
        assert list(summary.index) == ["L1", "L2", "norm"]
        assert list(summary.columns) == ["median", "hdi_lower", "hdi_upper"]
        assert summary.loc["L1", "median"] == pytest.approx(0.12)
        assert summary.loc["norm", "median"] == pytest.approx(np.hypot(0.12, 0.14))

    def test_hdi_brackets_the_median(self):
        fitted = ConjugateVAR(lags=1, prior=NIWPrior(), draws=200, seed=0).fit(_unidirectional_data())
        result = fitted.granger_causality("y2", "y1")
        lower, upper = result.hdi()
        assert lower <= result.median() <= upper
        summary = result.summary()
        assert (summary["hdi_lower"] <= summary["median"]).all()
        assert (summary["median"] <= summary["hdi_upper"]).all()

    def test_norm_draws_are_the_per_draw_euclidean_norm(self, hand_built):
        result = hand_built.granger_causality("y2", "y1", standardize=False)
        assert result.norm_draws.shape == (2, 10)
        np.testing.assert_allclose(result.norm_draws, np.hypot(0.12, 0.14))

    def test_p_rope_is_none_without_a_rope(self, hand_built):
        result = hand_built.granger_causality("y2", "y1", standardize=False)
        assert result.rope is None
        assert result.p_rope is None
        assert "p_rope" not in result.summary().columns

    def test_p_rope_lands_on_the_norm_row_only(self, hand_built):
        result = hand_built.granger_causality("y2", "y1", standardize=False, rope=0.5)
        assert result.p_rope == pytest.approx(1.0)  # ||b|| = 0.185 < 0.5 in every draw
        summary = result.summary()
        assert summary.loc["norm", "p_rope"] == pytest.approx(1.0)
        assert np.isnan(summary.loc[["L1", "L2"], "p_rope"]).all()

    def test_p_rope_is_a_probability(self):
        fitted = ConjugateVAR(lags=1, prior=NIWPrior(), draws=200, seed=0).fit(_unidirectional_data())
        p_rope = fitted.granger_causality("y2", "y1", rope=0.2).p_rope
        assert p_rope is not None
        assert 0.0 <= p_rope <= 1.0

    def test_result_is_frozen(self, hand_built):
        result = hand_built.granger_causality("y2", "y1")
        with pytest.raises(ValueError, match="frozen"):
            result.cause = "y1"

    def test_metadata_defaults_to_no_augmentation(self, hand_built):
        result = hand_built.granger_causality("y2", "y1")
        assert result.n_lags_tested == result.n_lags_fitted == 2
        assert result.augmentation == 0
        assert result.augmentation_source == "none"
        assert result.integration_order_result is None


# --------------- 5. directionality ---------------


class TestDirectionality:
    """One decisive unidirectional system, both directions queried.

    `ConjugateVAR` draws in closed form, so 400 draws on 300 observations
    is a fraction of a second and needs no MCMC diagnostics.
    """

    @pytest.fixture(scope="class")
    def fitted(self):
        return ConjugateVAR(lags=1, prior=NIWPrior(), draws=400, seed=0).fit(_unidirectional_data())

    def test_true_edge_is_large_and_outside_the_rope(self, fitted):
        result = fitted.granger_causality("y2", "y1", rope=0.1)
        assert result.median() > 0.15
        assert result.p_rope < 0.05

    def test_absent_edge_is_small_and_inside_the_rope(self, fitted):
        result = fitted.granger_causality("y1", "y2", rope=0.1)
        assert result.p_rope > 0.5

    def test_true_edge_dominates_the_absent_one(self, fitted):
        assert fitted.granger_causality("y2", "y1").median() > fitted.granger_causality("y1", "y2").median()


# --------------- 6. null calibration ---------------


def test_null_system_is_mostly_inside_the_rope():
    """Sanity check, not an exact calibration claim.

    Twenty independent null systems, both directions each. Under the
    conjugate Minnesota prior the coefficients are shrunk toward a random
    walk, so cross-variable draws are pulled toward zero and `p_rope` at a
    rope of 0.1 should sit high nearly everywhere; the thresholds are loose
    on purpose, because the prior — not a sampling distribution — is what
    sets the exact level.
    """
    p_ropes = []
    for seed in range(20):
        fitted = ConjugateVAR(lags=1, prior=NIWPrior(), draws=200, seed=seed).fit(_null_data(seed))
        for cause, effect in (("y2", "y1"), ("y1", "y2")):
            p_ropes.append(fitted.granger_causality(cause, effect, rope=0.1).p_rope)

    assert len(p_ropes) == 40
    assert float(np.median(p_ropes)) > 0.5
    assert min(p_ropes) > 0.1


# --------------- 7. Toda-Yamamoto contract ---------------


def _integration_order_result(order: dict[str, int], inconclusive: list[str]) -> IntegrationOrderResult:
    """Hand-built diagnostics, so the refusal contract needs no statsmodels."""
    return IntegrationOrderResult(
        order=order,
        alpha=0.05,
        max_order=2,
        regression="c",
        inconclusive=inconclusive,
        table=pd.DataFrame(
            {"joint_status": ["inconclusive"] * len(order)},
            index=pd.MultiIndex.from_tuples([(name, 0) for name in order], names=["variable", "d"]),
        ),
    )


class TestTodaYamamoto:
    def test_inconclusive_diagnostics_are_refused_by_name(self):
        diagnostics = _integration_order_result({"y1": 1, "y2": 2}, ["y2"])
        with pytest.raises(ValueError, match="y2") as excinfo:
            toda_yamamoto(
                _null_data(0),
                "y2",
                "y1",
                lags=1,
                integration_order_result=diagnostics,
            )
        message = str(excinfo.value)
        assert "under-augment" in message
        assert "summary()" in message
        assert "d=" in message

    def test_explicit_d_splits_tested_from_fitted_lags(self):
        result = toda_yamamoto(_null_data(0), "y2", "y1", lags=1, d=1)
        assert result.n_lags_tested == 1
        assert result.n_lags_fitted == 2
        assert result.augmentation == 1
        assert result.augmentation_source == "user"
        assert result.integration_order_result is None
        # The augmented lag is fitted but never reported.
        assert list(result.summary().index) == ["L1", "norm"]
        assert result.coef_draws.shape[-1] == 1

    def test_injected_clean_diagnostics_are_consulted_and_attached(self):
        diagnostics = _integration_order_result({"y1": 1, "y2": 1}, [])
        result = toda_yamamoto(
            _null_data(0),
            "y2",
            "y1",
            lags=1,
            integration_order_result=diagnostics,
        )
        assert result.augmentation == 1
        assert result.augmentation_source == "integration_order"
        assert result.integration_order_result is diagnostics

    def test_d_max_of_zero_still_records_that_diagnostics_ran(self):
        diagnostics = _integration_order_result({"y1": 0, "y2": 0}, [])
        result = toda_yamamoto(
            _null_data(0),
            "y2",
            "y1",
            lags=2,
            integration_order_result=diagnostics,
        )
        assert result.augmentation == 0
        assert result.augmentation_source == "integration_order"
        assert result.n_lags_fitted == result.n_lags_tested == 2

    def test_explicit_d_needs_no_statsmodels(self, monkeypatch):
        """The `d=` route must not even try to import the optional extra."""
        import impulso._stationarity as stationarity

        def _absent(module, *, extra):
            raise ImportError(f"{module} is not installed")

        monkeypatch.setattr(stationarity, "require", _absent)
        result = toda_yamamoto(_null_data(0), "y2", "y1", lags=1, d=1)
        assert result.augmentation_source == "user"
        assert np.isfinite(result.median())

    def test_exogenous_data_points_at_the_manual_route(self):
        data = _null_data(0)
        with_exog = VARData(
            endog=np.asarray(data.endog),
            endog_names=list(data.endog_names),
            exog=np.arange(len(data.index), dtype=float).reshape(-1, 1),
            exog_names=["trend"],
            index=data.index,
        )
        with pytest.raises(ValueError, match="granger_causality") as excinfo:
            toda_yamamoto(with_exog, "y2", "y1", lags=1, d=0)
        assert "VAR(lags=p + d)" in str(excinfo.value)

    def test_unknown_variable_is_refused_before_fitting(self):
        with pytest.raises(ValueError, match=r"unknown variable\(s\)"):
            toda_yamamoto(_null_data(0), "gdp", "y1", lags=1, d=0)

    @pytest.mark.parametrize(("lags", "match"), [("nope", "lags must be an int"), (0, "lags must be >= 1")])
    def test_invalid_lag_specification_is_refused(self, lags, match):
        with pytest.raises(ValueError, match=match):
            toda_yamamoto(_null_data(0), "y2", "y1", lags=lags, d=0)

    def test_negative_d_is_refused(self):
        with pytest.raises(ValueError, match="d must be non-negative"):
            toda_yamamoto(_null_data(0), "y2", "y1", lags=1, d=-1)

    def test_criterion_string_selects_the_test_lag_order(self):
        result = toda_yamamoto(_null_data(0), "y2", "y1", lags="bic", max_lags=4, d=1)
        assert result.n_lags_tested >= 1
        assert result.n_lags_fitted == result.n_lags_tested + 1

    def test_returns_a_granger_causality_result(self):
        assert isinstance(toda_yamamoto(_null_data(0), "y2", "y1", lags=1, d=0), GrangerCausalityResult)


class TestTodaYamamotoEndToEnd:
    """Full path on an I(1) system, diagnostics included."""

    def test_augmented_fit_recovers_the_direction(self):
        pytest.importorskip("statsmodels")
        data = _i1_unidirectional()

        forward = toda_yamamoto(data, "x", "y", lags=2, rope=0.1, draws=400, seed=0)
        assert forward.augmentation_source == "integration_order"
        assert forward.augmentation >= 1
        assert forward.n_lags_fitted == forward.n_lags_tested + forward.augmentation
        assert forward.integration_order_result is not None
        assert forward.integration_order_result.d_max == forward.augmentation

        # Reuse the diagnostics rather than re-running them for the reverse
        # direction; the injected path is asserted separately above.
        reverse = toda_yamamoto(
            data,
            "y",
            "x",
            lags=2,
            rope=0.1,
            draws=400,
            seed=0,
            integration_order_result=forward.integration_order_result,
        )
        assert forward.p_rope is not None
        assert reverse.p_rope is not None
        assert forward.p_rope < reverse.p_rope


# --------------- 8. NUTS smoke ---------------


@pytest.mark.slow
def test_granger_causality_on_a_nuts_fit(var_data_2v):
    """The reduced-form read must work on a real PyMC posterior too."""
    from impulso import VAR
    from impulso.samplers import NUTSSampler

    fitted = VAR(lags=1).fit(var_data_2v, NUTSSampler(draws=100, tune=100, chains=2, cores=1))

    # Pin the coefficient layout against the real posterior's labels.
    coeff = [str(c) for c in fitted.idata.posterior["B"].coords["coeff"].values]
    assert coeff == ["L1.y1", "L1.y2"]

    for cause, effect in (("y2", "y1"), ("y1", "y2")):
        result = fitted.granger_causality(cause, effect, rope=0.1)
        summary = result.summary()[["median", "hdi_lower", "hdi_upper"]]
        assert np.isfinite(summary.to_numpy(dtype=float)).all()
        lower, upper = result.hdi()
        assert lower <= result.median() <= upper
        p_rope = result.p_rope
        assert p_rope is not None
        assert 0.0 <= p_rope <= 1.0

        index = list(var_data_2v.endog_names).index(cause)
        expected = fitted.idata.posterior["B"].values[..., list(var_data_2v.endog_names).index(effect), index]
        np.testing.assert_allclose(result.coef_draws[..., 0] / result.scale, expected)
