"""Tests for the stationarity, unit-root, and cointegration diagnostics.

Correctness is pinned two ways that do not rely on statsmodels being right:
known-answer data-generating processes (a stationary AR(1) must not look like
a random walk), and critical values copied from the original published
tables. A handful of round-trip tests against statsmodels are kept as change
detectors, and are labelled as such.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from impulso import (
    CointegrationTestResult,
    IntegrationOrderResult,
    StationarityTestResult,
    adf_test,
    integration_order,
    johansen_test,
    kpss_test,
)

pytest.importorskip("statsmodels")

T = 500
INDEX = pd.date_range("1980-01-01", periods=T, freq="ME")


# --------------- Known-answer data-generating processes ---------------


@pytest.fixture
def stationary_series():
    """AR(1) with rho = 0.5 — comfortably inside the unit circle."""
    rng = np.random.default_rng(11)
    e = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + e[t]
    return pd.Series(y, index=INDEX, name="ar1")


@pytest.fixture
def unit_root_series():
    """Driftless random walk — I(1) by construction."""
    rng = np.random.default_rng(12)
    return pd.Series(np.cumsum(rng.standard_normal(T)), index=INDEX, name="rw")


@pytest.fixture
def i2_series():
    """Doubly integrated series — I(2) by construction."""
    # Seed chosen for a decisive margin at both d = 0 and d = 1.
    rng = np.random.default_rng(31)
    return pd.Series(np.cumsum(np.cumsum(rng.standard_normal(T))), index=INDEX, name="i2")


@pytest.fixture
def trend_stationary_series():
    """Deterministic linear trend plus AR(1) noise — I(0) around a trend."""
    rng = np.random.default_rng(14)
    e = rng.standard_normal(T)
    noise = np.zeros(T)
    for t in range(1, T):
        noise[t] = 0.5 * noise[t - 1] + e[t]
    return pd.Series(0.05 * np.arange(T) + noise, index=INDEX, name="trend")


@pytest.fixture
def cointegrated_frame():
    """Two series sharing one stochastic trend — cointegration rank 1."""
    # Seed chosen for a decisive non-rejection of the r = 1 null.
    rng = np.random.default_rng(32)
    w = np.cumsum(rng.standard_normal(T))
    return pd.DataFrame(
        {
            "y1": w + rng.standard_normal(T) * 0.5,
            "y2": 2.0 * w + rng.standard_normal(T) * 0.5,
        },
        index=INDEX,
    )


@pytest.fixture
def independent_walks():
    """Two unrelated random walks — cointegration rank 0."""
    # Seed chosen so both statistics fall well short of their critical values;
    # seed 16 produced a borderline max-eigenvalue Type-I rejection.
    rng = np.random.default_rng(21)
    return pd.DataFrame(
        {
            "y1": np.cumsum(rng.standard_normal(T)),
            "y2": np.cumsum(rng.standard_normal(T)),
        },
        index=INDEX,
    )


@pytest.fixture
def mixed_frame(stationary_series, unit_root_series):
    """One I(0) and one I(1) series — d_max should be 1."""
    return pd.DataFrame({"ar1": stationary_series, "rw": unit_root_series})


@pytest.fixture
def broken_series():
    """Stationary AR(1) with a mid-sample level shift.

    A break is exactly the case where the two tests part company: ADF sees
    fast mean reversion within each regime and rejects the unit root, while
    KPSS sees the shift as a wandering mean and rejects stationarity.
    """
    rng = np.random.default_rng(0)
    e = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.4 * y[t - 1] + e[t]
    return pd.Series(y + np.where(np.arange(T) >= T // 2, 2.0, 0.0), index=INDEX, name="break")


# --------------- 1. DGP validation ---------------


def test_adf_rejects_unit_root_for_stationary_series(stationary_series):
    result = adf_test(stationary_series)
    assert result.conclusions["ar1"] == "stationary"
    assert result.pvalues["ar1"] < 0.01


def test_adf_does_not_reject_for_random_walk(unit_root_series):
    result = adf_test(unit_root_series)
    assert result.conclusions["rw"] == "non-stationary"
    assert result.pvalues["rw"] > 0.05


def test_adf_does_not_reject_for_i2_series(i2_series):
    result = adf_test(i2_series)
    assert result.conclusions["i2"] == "non-stationary"


def test_kpss_verdicts_are_reversed_relative_to_adf(stationary_series, i2_series):
    """KPSS's null is stationarity, so its conclusion column must flip."""
    stationary = kpss_test(stationary_series)
    assert stationary.conclusions["ar1"] == "stationary"
    assert not bool(stationary.table.loc["ar1", "reject"])

    integrated = kpss_test(i2_series)
    assert integrated.conclusions["i2"] == "non-stationary"
    assert bool(integrated.table.loc["i2", "reject"])


def test_null_hypotheses_are_opposite(stationary_series):
    adf = adf_test(stationary_series)
    kpss = kpss_test(stationary_series)
    assert adf.test == "adf"
    assert kpss.test == "kpss"
    assert "unit root" in adf.null_hypothesis
    assert "stationary" in kpss.null_hypothesis


def test_trend_stationary_series_needs_the_trend_regressor(trend_stationary_series):
    """Without a trend term ADF mistakes a deterministic trend for a unit root."""
    with_constant = adf_test(trend_stationary_series, regression="c")
    with_trend = adf_test(trend_stationary_series, regression="ct")
    assert with_constant.conclusions["trend"] == "non-stationary"
    assert with_trend.conclusions["trend"] == "stationary"


def test_kpss_trend_regressor_finds_trend_stationarity(trend_stationary_series):
    around_constant = kpss_test(trend_stationary_series, regression="c")
    around_trend = kpss_test(trend_stationary_series, regression="ct")
    assert around_constant.conclusions["trend"] == "non-stationary"
    assert around_trend.conclusions["trend"] == "stationary"


def test_johansen_finds_rank_one_for_cointegrated_pair(cointegrated_frame):
    result = johansen_test(cointegrated_frame)
    assert result.rank_trace == 1
    assert result.rank_max_eigen == 1
    assert result.rank == result.rank_trace


def test_johansen_finds_rank_zero_for_independent_walks(independent_walks):
    result = johansen_test(independent_walks)
    assert result.rank_trace == 0
    assert result.rank_max_eigen == 0


def test_johansen_finds_full_rank_for_stationary_var(var_data_2v):
    """A stationary VAR in levels is trivially "cointegrated" at full rank."""
    result = johansen_test(var_data_2v)
    assert result.rank_trace == 2
    assert result.rank_max_eigen == 2


def test_johansen_reports_effective_sample_and_eigenvalues(cointegrated_frame):
    result = johansen_test(cointegrated_frame, k_ar_diff=2)
    # One observation lost to differencing, k_ar_diff more to the lags.
    assert result.n_obs == T - 2 - 1
    assert result.eigenvalues.shape == (2,)
    assert np.all(np.diff(result.eigenvalues) <= 0)


def test_integration_order_recovers_zero_one_two(stationary_series, unit_root_series, i2_series):
    frame = pd.DataFrame({
        "ar1": stationary_series,
        "rw": unit_root_series,
        "i2": i2_series,
    })
    result = integration_order(frame)
    assert result.order == {"ar1": 0, "rw": 1, "i2": 2}
    assert result.d_max == 2


def test_d_max_is_the_maximum_over_variables(mixed_frame):
    result = integration_order(mixed_frame)
    assert result.order == {"ar1": 0, "rw": 1}
    assert result.d_max == 1


def test_integration_order_flags_series_still_integrated_at_max_order(i2_series):
    result = integration_order(i2_series, max_order=1)
    assert result.order == {"i2": 1}
    assert result.inconclusive == ["i2"]


def test_integration_order_flags_conflict_between_the_two_tests(broken_series):
    """A stopping level where both tests reject is not a clean I(0) verdict."""
    result = integration_order(broken_series)
    assert result.order == {"break": 0}
    assert result.table.loc[("break", 0), "joint_status"] == "conflicting"
    assert result.inconclusive == ["break"]


@pytest.mark.parametrize(
    ("adf_reject", "kpss_reject", "expected"),
    [
        (True, False, "stationary"),
        (False, True, "unit_root"),
        (True, True, "conflicting"),
        (False, False, "inconclusive"),
    ],
)
def test_joint_status_encodes_the_two_by_two_contract(adf_reject, kpss_reject, expected):
    from impulso._stationarity import _joint_status

    assert _joint_status(adf_reject, kpss_reject) == expected


def test_integration_order_table_is_indexed_by_variable_and_d(mixed_frame):
    result = integration_order(mixed_frame)
    assert result.table.index.names == ["variable", "d"]
    assert list(result.table.loc["ar1"].index) == [0]
    assert list(result.table.loc["rw"].index) == [0, 1]
    assert set(result.table["joint_status"]) <= {
        "stationary",
        "unit_root",
        "conflicting",
        "inconclusive",
    }


def test_integration_order_stops_at_first_adf_rejection(unit_root_series):
    result = integration_order(unit_root_series, max_order=2)
    # The d = 2 row must not exist: the search stops once ADF rejects.
    assert list(result.table.loc["rw"].index) == [0, 1]
    assert bool(result.table.loc[("rw", 1), "adf_reject"])


def test_integration_order_zero_max_order_reports_levels_only(unit_root_series):
    result = integration_order(unit_root_series, max_order=0)
    assert result.order == {"rw": 0}
    assert result.inconclusive == ["rw"]
    assert list(result.table.loc["rw"].index) == [0]


def test_integration_order_level_regression_does_not_leak_into_differences(trend_stationary_series):
    """`ct` applies to the level test only; differences are tested with `c`."""
    result = integration_order(trend_stationary_series, regression="ct")
    assert result.regression == "ct"
    assert result.order == {"trend": 0}


# --------------- 2. Published critical-value pins ---------------


def test_kpss_level_critical_values_match_kwiatkowski_1992_table_1(stationary_series):
    """KPSS (1992), Table 1, level-stationary row: 0.347 / 0.463 / 0.739."""
    row = kpss_test(stationary_series, regression="c").table.loc["ar1"]
    assert row["crit_10pct"] == pytest.approx(0.347)
    assert row["crit_5pct"] == pytest.approx(0.463)
    assert row["crit_1pct"] == pytest.approx(0.739)


def test_kpss_trend_critical_values_match_kwiatkowski_1992_table_1(stationary_series):
    """KPSS (1992), Table 1, trend-stationary row: 0.119 / 0.146 / 0.216."""
    row = kpss_test(stationary_series, regression="ct").table.loc["ar1"]
    assert row["crit_10pct"] == pytest.approx(0.119)
    assert row["crit_5pct"] == pytest.approx(0.146)
    assert row["crit_1pct"] == pytest.approx(0.216)


def test_adf_constant_critical_values_match_mackinnon(stationary_series):
    """MacKinnon (2010) tau_c, 5% asymptote is about -2.86 for large T."""
    row = adf_test(stationary_series, regression="c").table.loc["ar1"]
    assert row["crit_5pct"] == pytest.approx(-2.86, abs=0.03)
    assert row["crit_1pct"] == pytest.approx(-3.44, abs=0.03)
    assert row["crit_10pct"] == pytest.approx(-2.57, abs=0.03)
    # Ordering is a structural property of the tables, not a numeric accident.
    assert row["crit_1pct"] < row["crit_5pct"] < row["crit_10pct"]


def test_johansen_trace_critical_value_matches_osterwald_lenum(cointegrated_frame):
    """Osterwald-Lenum (1992), constant case: 95% trace CV at n - r = 2 is 15.49."""
    result = johansen_test(cointegrated_frame, det_order=0, alpha=0.05)
    assert result.table.loc[0, "trace_crit"] == pytest.approx(15.49, abs=0.1)
    assert result.table.loc[1, "trace_crit"] == pytest.approx(3.84, abs=0.1)


def test_johansen_alpha_picks_the_matching_critical_value_column(cointegrated_frame):
    """Tighter alpha means a larger critical value, hence a harder rejection."""
    ten = johansen_test(cointegrated_frame, alpha=0.10).table["trace_crit"]
    five = johansen_test(cointegrated_frame, alpha=0.05).table["trace_crit"]
    one = johansen_test(cointegrated_frame, alpha=0.01).table["trace_crit"]
    assert (ten < five).all()
    assert (five < one).all()


# --------------- 3. Round-trip against statsmodels (change detectors) ---------------


def test_adf_statistic_matches_statsmodels_directly(unit_root_series):
    from statsmodels.tsa.stattools import adfuller

    expected = adfuller(unit_root_series.to_numpy(), regression="c", autolag="aic")
    row = adf_test(unit_root_series).table.loc["rw"]
    assert row["statistic"] == pytest.approx(expected[0])
    assert row["pvalue"] == pytest.approx(expected[1])
    assert row["lags"] == expected[2]


def test_kpss_statistic_matches_statsmodels_directly(unit_root_series):
    from statsmodels.tsa.stattools import kpss as sm_kpss

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = sm_kpss(unit_root_series.to_numpy(), regression="c", nlags="auto")
    row = kpss_test(unit_root_series).table.loc["rw"]
    assert row["statistic"] == pytest.approx(expected[0])
    assert row["pvalue"] == pytest.approx(expected[1])
    assert row["lags"] == expected[2]


def test_johansen_statistics_match_statsmodels_directly(cointegrated_frame):
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    expected = coint_johansen(cointegrated_frame.to_numpy(), 0, 1)
    table = johansen_test(cointegrated_frame).table
    np.testing.assert_allclose(table["trace_stat"].to_numpy(), expected.lr1)
    np.testing.assert_allclose(table["maxeig_stat"].to_numpy(), expected.lr2)


# --------------- 4. API hygiene ---------------


def test_all_input_types_give_identical_numbers(var_data_2v):
    frame = pd.DataFrame(var_data_2v.endog, columns=var_data_2v.endog_names, index=var_data_2v.index)
    from_data = adf_test(var_data_2v).table
    from_frame = adf_test(frame).table
    from_series = adf_test(frame["y1"]).table
    pd.testing.assert_frame_equal(from_data, from_frame)
    assert from_series.loc["y1", "statistic"] == pytest.approx(from_frame.loc["y1", "statistic"])


def test_vardata_input_tests_endogenous_block_only(var_data_2v, rng):
    """Exogenous regressors are not integration-order candidates."""
    from impulso.data import VARData

    with_exog = VARData(
        endog=var_data_2v.endog,
        endog_names=var_data_2v.endog_names,
        exog=rng.standard_normal((var_data_2v.endog.shape[0], 1)),
        exog_names=["x"],
        index=var_data_2v.index,
    )
    result = adf_test(with_exog)
    assert list(result.table.index) == ["y1", "y2"]


def test_unnamed_series_gets_a_placeholder_name(unit_root_series):
    result = adf_test(pd.Series(unit_root_series.to_numpy(), index=INDEX))
    assert list(result.table.index) == ["series"]


def test_variables_argument_subsets_and_orders(mixed_frame):
    result = adf_test(mixed_frame, variables=["rw"])
    assert list(result.table.index) == ["rw"]

    reordered = kpss_test(mixed_frame, variables=["rw", "ar1"])
    assert list(reordered.table.index) == ["rw", "ar1"]


def test_unknown_variable_raises(mixed_frame):
    with pytest.raises(ValueError, match="variables not found"):
        adf_test(mixed_frame, variables=["nope"])


def test_non_finite_input_raises_naming_the_column(mixed_frame):
    broken = mixed_frame.copy()
    broken.loc[broken.index[0], "rw"] = np.nan
    with pytest.raises(ValueError, match="rw"):
        adf_test(broken)


def test_unsupported_input_type_raises():
    with pytest.raises(TypeError, match="VARData, DataFrame, or Series"):
        adf_test([1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"regression": "ctt"}, "regression must be one of"),
        ({"lag_selection": "hq"}, "lag_selection must be one of"),
        ({"alpha": 1.5}, r"alpha must lie in \(0, 1\)"),
    ],
)
def test_adf_rejects_invalid_arguments(stationary_series, kwargs, match):
    with pytest.raises(ValueError, match=match):
        adf_test(stationary_series, **kwargs)


def test_kpss_rejects_unsupported_regression(stationary_series):
    with pytest.raises(ValueError, match="regression must be one of"):
        kpss_test(stationary_series, regression="n")


def test_johansen_rejects_untabulated_alpha(cointegrated_frame):
    with pytest.raises(ValueError, match="alpha must be one of"):
        johansen_test(cointegrated_frame, alpha=0.07)


def test_johansen_rejects_bad_det_order_and_lags(cointegrated_frame):
    with pytest.raises(ValueError, match="det_order must be"):
        johansen_test(cointegrated_frame, det_order=2)
    with pytest.raises(ValueError, match="k_ar_diff must be non-negative"):
        johansen_test(cointegrated_frame, k_ar_diff=-1)


def test_johansen_needs_at_least_two_series(unit_root_series):
    with pytest.raises(ValueError, match="at least two series"):
        johansen_test(unit_root_series.to_frame())


def test_integration_order_rejects_invalid_arguments(stationary_series):
    with pytest.raises(ValueError, match="max_order must be non-negative"):
        integration_order(stationary_series, max_order=-1)
    with pytest.raises(ValueError, match="regression must be one of"):
        integration_order(stationary_series, regression="n")
    with pytest.raises(ValueError, match=r"alpha must lie in \(0, 1\)"):
        integration_order(stationary_series, alpha=0.0)


def test_adf_honours_explicit_lag_length(unit_root_series):
    result = adf_test(unit_root_series, max_lags=4, lag_selection=None)
    assert result.table.loc["rw", "lags"] == 4


def test_kpss_flags_bounded_pvalue_without_leaking_a_warning(i2_series):
    """The p-value is clipped at 0.01; that fact becomes a column, not a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = kpss_test(i2_series)
    row = result.table.loc["i2"]
    assert bool(row["pvalue_bounded"])
    assert row["pvalue"] == pytest.approx(0.01)


def test_integration_order_does_not_leak_warnings(mixed_frame):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        integration_order(mixed_frame)


def test_summary_returns_the_table(stationary_series, cointegrated_frame, mixed_frame):
    stationarity = adf_test(stationary_series)
    assert list(stationarity.summary().columns) == [
        "statistic",
        "pvalue",
        "lags",
        "crit_1pct",
        "crit_5pct",
        "crit_10pct",
        "reject",
        "conclusion",
    ]
    assert stationarity.summary().index.name == "variable"

    kpss_summary = kpss_test(stationary_series).summary()
    assert "pvalue_bounded" in kpss_summary.columns

    cointegration = johansen_test(cointegrated_frame).summary()
    assert list(cointegration.columns) == [
        "trace_stat",
        "trace_crit",
        "trace_reject",
        "maxeig_stat",
        "maxeig_crit",
        "maxeig_reject",
    ]
    assert cointegration.index.name == "r"

    orders = integration_order(mixed_frame).summary()
    assert list(orders.columns) == [
        "adf_stat",
        "adf_pvalue",
        "adf_lags",
        "adf_reject",
        "kpss_stat",
        "kpss_pvalue",
        "kpss_lags",
        "kpss_reject",
        "kpss_pvalue_bounded",
        "joint_status",
    ]


def test_results_are_frozen(stationary_series, cointegrated_frame, mixed_frame):
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        adf_test(stationary_series).alpha = 0.1
    with pytest.raises(pydantic.ValidationError):
        johansen_test(cointegrated_frame).rank_trace = 2
    with pytest.raises(pydantic.ValidationError):
        integration_order(mixed_frame).order = {}


def test_result_types_are_exported(stationary_series, cointegrated_frame, mixed_frame):
    assert isinstance(adf_test(stationary_series), StationarityTestResult)
    assert isinstance(johansen_test(cointegrated_frame), CointegrationTestResult)
    assert isinstance(integration_order(mixed_frame), IntegrationOrderResult)


def test_alpha_is_recorded_and_changes_the_verdict(unit_root_series):
    """A p-value of about 0.4 rejects at no sensible level; a looser alpha still records."""
    strict = adf_test(unit_root_series, alpha=0.01)
    loose = adf_test(unit_root_series, alpha=0.5)
    assert strict.alpha == 0.01
    assert loose.alpha == 0.5
    assert strict.conclusions["rw"] == "non-stationary"
    assert loose.conclusions["rw"] == "stationary"
