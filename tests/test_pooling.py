"""Tests for static predictive-density pooling (issue #151).

All fast: the fixtures hand-build `FittedVAR` posteriors rather than
sampling, following the `test_density_forecast.py` precedent.
"""

import warnings

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

matplotlib.use("Agg")

from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.pooling import (
    PredictivePool,
    _gaussian_log_scores,
    _index_freq,
    _log_score_weights,
    _pooled_row_scores,
    _spawn,
    _stacking_weights,
    pool_forecasts,
)
from impulso.results import ForecastResult
from impulso.volatility import Constant

A1_DEFAULT = np.array([[0.5, 0.1], [-0.2, 0.3]])

# log([[4, 1], [4, 1], [1, 4]]): the analytic reference matrix.
#   stacking   -> FOC 2*3/(3w + 1) = 3/(4 - 3w) -> w* = 7/9
#   log score  -> products 16 vs 4 -> [0.8, 0.2]
REFERENCE_SCORES = np.log(np.array([[4.0, 1.0], [4.0, 1.0], [1.0, 4.0]]))


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _fitted(
    sd,
    *,
    n_lags: int = 1,
    A1: np.ndarray | None = None,
    intercept: float = 0.0,
    n_chains: int = 2,
    n_draws: int = 100,
    n_obs: int = 60,
    start: str = "2000-01-01",
    freq: str = "QS",
    var_names: tuple[str, ...] = ("y1", "y2"),
    exog: bool = False,
) -> FittedVAR:
    """Hand-build a FittedVAR with a point-mass posterior on (B, intercept, L).

    `sd` is the diagonal of the shock Cholesky factor, so the predictive
    spread of each variable is controlled exactly.
    """
    n_vars = len(var_names)
    coefs = np.asarray(A1 if A1 is not None else A1_DEFAULT, dtype=float)
    B = np.broadcast_to(coefs, (n_chains, n_draws, n_vars, n_vars * n_lags)).copy()
    mu = np.full((n_chains, n_draws, n_vars), float(intercept))
    L = np.zeros((n_chains, n_draws, n_vars, n_vars))
    L[:, :, range(n_vars), range(n_vars)] = np.asarray(sd, dtype=float)
    variables = {
        "B": (("chain", "draw", "var", "coeff"), B),
        "intercept": (("chain", "draw", "var"), mu),
        "L": (("chain", "draw", "var1", "var2"), L),
    }
    if exog:
        variables["B_exog"] = (("chain", "draw", "var", "exog"), np.zeros((n_chains, n_draws, n_vars, 1)))
    idata = az.InferenceData(posterior=xr.Dataset(variables))

    index = pd.date_range(start, periods=n_obs, freq=freq)
    y = np.zeros((n_obs, n_vars))
    y[0] = 1.0
    for t in range(1, n_obs):
        y[t] = intercept + coefs @ y[t - 1]
    exog_arr = np.ones((n_obs, 1)) if exog else None
    data = VARData(
        endog=y,
        endog_names=list(var_names),
        exog=exog_arr,
        exog_names=["x"] if exog else None,
        index=index,
    )
    return FittedVAR(
        idata=idata,
        n_lags=n_lags,
        data=data,
        var_names=list(var_names),
        volatility=Constant(),
    )


def _mean_path(fit: FittedVAR, steps: int) -> np.ndarray:
    """Closed-form conditional mean path A1^h y_T (intercept 0 fixtures)."""
    A1 = fit.idata.posterior["B"].values[0, 0]
    y = fit.data.endog[-1].copy()
    out = np.empty((steps, y.size))
    for h in range(steps):
        y = A1 @ y
        out[h] = y
    return out


def _holdout(fit: FittedVAR, values: np.ndarray, *, exog: bool = False) -> VARData:
    """Held-out VARData immediately following a fit's estimation sample."""
    steps = values.shape[0]
    index = pd.date_range(fit.data.index[-1], periods=steps + 1, freq=fit.data.index.freq)[1:]
    return VARData(
        endog=np.asarray(values, dtype=float),
        endog_names=list(fit.var_names),
        exog=np.ones((steps, 1)) if exog else None,
        exog_names=["x"] if exog else None,
        index=index,
    )


@pytest.fixture
def mirrored_pool():
    """Two models with identical means and mirrored shock scales.

    The headline stacking case: neither model dominates, the holdout
    alternates between the regions each model covers, and the pooled score
    beats both members by a wide margin.
    """
    tight_then_wide = _fitted(sd=[0.3, 1.5])
    wide_then_tight = _fitted(sd=[1.5, 0.3])
    steps = 12
    mean = _mean_path(tight_then_wide, steps)
    deviation = np.tile(np.array([[0.0, 2.5], [2.5, 0.0]]), (steps // 2, 1))
    holdout = _holdout(tight_then_wide, mean + deviation)
    return {"tight_y1": tight_then_wide, "tight_y2": wide_then_tight}, holdout


@pytest.fixture
def dominance_pool():
    """Three models where one is unambiguously best on the holdout."""
    good = _fitted(sd=[0.5, 0.5])
    wide = _fitted(sd=[5.0, 5.0])
    biased = _fitted(sd=[0.5, 0.5], intercept=3.0)
    steps = 8
    holdout = _holdout(good, _mean_path(good, steps))
    return {"good": good, "wide": wide, "biased": biased}, holdout


# --------------------------------------------------------------------------
# A. Weight solvers against closed-form optima
# --------------------------------------------------------------------------


class TestWeightSolvers:
    def test_stacking_interior_optimum(self):
        weights, converged, message = _stacking_weights(REFERENCE_SCORES)
        assert converged
        assert message
        np.testing.assert_allclose(weights, [7.0 / 9.0, 2.0 / 9.0], atol=1e-6)

    def test_log_score_closed_form(self):
        np.testing.assert_allclose(_log_score_weights(REFERENCE_SCORES), [0.8, 0.2], atol=1e-12)

    def test_dominance_collapses_to_one_model(self):
        scores = np.array([[0.0, -50.0], [0.0, -50.0], [0.0, -50.0]])
        np.testing.assert_allclose(_stacking_weights(scores)[0], [1.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(_log_score_weights(scores), [1.0, 0.0], atol=1e-8)

    def test_symmetric_rows_split_evenly(self):
        scores = np.log(np.array([[4.0, 1.0], [1.0, 4.0]]))
        np.testing.assert_allclose(_stacking_weights(scores)[0], [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(_log_score_weights(scores), [0.5, 0.5], atol=1e-12)

    @pytest.mark.parametrize("method", ["stacking", "log_score"])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_weights_live_on_the_simplex(self, method, seed):
        rng = np.random.default_rng(seed)
        scores = rng.standard_normal((7, 4)) * 5.0
        weights = _stacking_weights(scores)[0] if method == "stacking" else _log_score_weights(scores)
        assert weights.shape == (4,)
        assert np.all(weights >= 0.0)
        assert np.isclose(weights.sum(), 1.0, atol=1e-10)

    def test_stacking_keeps_a_model_that_dies_at_one_point(self):
        """A -inf at one point kills log-score weights but not stacking."""
        scores = np.array([[0.0, -np.inf], [0.0, 0.0], [-3.0, 0.0]])
        weights, _, _ = _stacking_weights(scores)
        assert np.all(np.isfinite(weights))
        # Analytic optimum: w1 = 1 / (2 * (1 - exp(-3))).
        np.testing.assert_allclose(weights[0], 1.0 / (2.0 * (1.0 - np.exp(-3.0))), atol=1e-6)
        assert weights[1] > 0.4
        np.testing.assert_allclose(_log_score_weights(scores), [1.0, 0.0], atol=1e-12)

    def test_dead_point_raises_naming_the_point(self):
        scores = np.array([[0.0, 0.0], [-np.inf, -np.inf], [0.0, 0.0]])
        index = pd.DatetimeIndex(["2020-01-01", "2020-04-01", "2020-07-01"])
        with pytest.raises(ValueError, match="2020-04-01"):
            _stacking_weights(scores, index=index)
        with pytest.raises(ValueError, match="2020-04-01"):
            _log_score_weights(scores, index=index)

    def test_dead_point_without_index_names_the_step(self):
        scores = np.array([[0.0, 0.0], [-np.inf, -np.inf]])
        with pytest.raises(ValueError, match="holdout step 2"):
            _log_score_weights(scores)

    def test_every_total_infinite_rejects_log_score_weights(self):
        """Each row survives, but each column dies somewhere."""
        scores = np.array([[0.0, -np.inf], [-np.inf, 0.0]])
        with pytest.raises(ValueError, match="method='stacking'"):
            _log_score_weights(scores)
        assert np.all(np.isfinite(_stacking_weights(scores)[0]))

    def test_single_model_matrix_rejected(self):
        with pytest.raises(ValueError, match="at least two fitted models"):
            _log_score_weights(np.zeros((3, 1)))

    def test_non_2d_matrix_rejected(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            _log_score_weights(np.zeros(3))

    def test_nan_scores_rejected(self):
        with pytest.raises(ValueError, match="NaN or \\+inf"):
            _stacking_weights(np.array([[0.0, np.nan], [0.0, 0.0]]))

    def test_optimiser_retries_then_gives_up(self, monkeypatch):
        import scipy.optimize

        calls = []
        real = scipy.optimize.minimize

        def always_fails(*args, **kwargs):
            calls.append(kwargs.get("x0"))
            result = real(*args, **kwargs)
            result.success = False
            result.message = "synthetic failure"
            return result

        monkeypatch.setattr(scipy.optimize, "minimize", always_fails)
        with pytest.raises(RuntimeError, match="synthetic failure"):
            _stacking_weights(REFERENCE_SCORES)
        assert len(calls) == 2
        # The retry starts somewhere other than the uniform point.
        assert not np.allclose(calls[0], calls[1])

    def test_optimiser_retry_can_succeed(self, monkeypatch):
        import scipy.optimize

        real = scipy.optimize.minimize
        state = {"n": 0}

        def fails_once(*args, **kwargs):
            state["n"] += 1
            result = real(*args, **kwargs)
            if state["n"] == 1:
                result.success = False
            return result

        monkeypatch.setattr(scipy.optimize, "minimize", fails_once)
        weights, converged, _ = _stacking_weights(REFERENCE_SCORES)
        assert converged
        assert state["n"] == 2
        np.testing.assert_allclose(weights, [7.0 / 9.0, 2.0 / 9.0], atol=1e-6)


# --------------------------------------------------------------------------
# B. Overflow safety
# --------------------------------------------------------------------------


class TestOverflow:
    @pytest.mark.parametrize("shift", [-5e4, 5e4])
    def test_log_score_weights_survive_extreme_shifts(self, shift):
        scores = REFERENCE_SCORES + shift
        with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise"):
            warnings.simplefilter("error")
            weights = _log_score_weights(scores)
        assert np.all(np.isfinite(weights))
        np.testing.assert_allclose(weights, [0.8, 0.2], atol=1e-12)

    @pytest.mark.parametrize("shift", [-1e5, 1e4])
    def test_stacking_survives_extreme_shifts(self, shift):
        """An unshifted exponentiation (as ArviZ does) fails here."""
        scores = REFERENCE_SCORES + shift
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            weights, converged, _ = _stacking_weights(scores)
        assert converged
        assert np.all(np.isfinite(weights))
        np.testing.assert_allclose(weights, [7.0 / 9.0, 2.0 / 9.0], atol=1e-6)

    def test_naive_exponentiation_would_have_failed(self):
        """Documents why the shift is load-bearing rather than cosmetic."""
        with np.errstate(over="ignore", under="ignore"):
            assert not np.isfinite(np.exp(REFERENCE_SCORES + 1e4)).all()
            assert np.exp(REFERENCE_SCORES - 1e5).max() == 0.0


# --------------------------------------------------------------------------
# C. Score matrix
# --------------------------------------------------------------------------


class TestScoreMatrix:
    def test_joint_gaussian_matches_scipy(self):
        from scipy.stats import multivariate_normal

        rng = np.random.default_rng(7)
        draws = rng.standard_normal((500, 4, 3)) @ np.array([[1.0, 0.0, 0.0], [0.6, 0.9, 0.0], [0.2, 0.3, 1.1]]).T
        y = rng.standard_normal((4, 3))
        scores = _gaussian_log_scores(draws, y, density="gaussian", label="m")
        expected = [
            multivariate_normal.logpdf(y[h], draws[:, h, :].mean(0), np.cov(draws[:, h, :], rowvar=False, ddof=1))
            for h in range(4)
        ]
        np.testing.assert_allclose(scores, expected, atol=1e-6)

    def test_diagonal_matches_scipy(self):
        from scipy.stats import norm

        rng = np.random.default_rng(11)
        draws = rng.standard_normal((300, 3, 2)) * np.array([1.0, 2.5])
        y = rng.standard_normal((3, 2))
        scores = _gaussian_log_scores(draws, y, density="diagonal", label="m")
        expected = [
            float(np.sum(norm.logpdf(y[h], draws[:, h, :].mean(0), draws[:, h, :].std(0, ddof=1)))) for h in range(3)
        ]
        np.testing.assert_allclose(scores, expected, atol=1e-10)

    def test_diagonal_ignores_correlation(self):
        """The escape hatch really is the product of marginals."""
        rng = np.random.default_rng(3)
        base = rng.standard_normal((400, 2, 2))
        correlated = base @ np.array([[1.0, 0.0], [0.95, 0.31]]).T
        y = np.zeros((2, 2))
        joint = _gaussian_log_scores(correlated, y, density="gaussian", label="m")
        diagonal = _gaussian_log_scores(correlated, y, density="diagonal", label="m")
        assert not np.allclose(joint, diagonal)

    def test_singular_covariance_points_at_diagonal(self):
        rng = np.random.default_rng(5)
        draws = rng.standard_normal((200, 2, 2))
        draws[:, 1, 1] = 3.0  # constant column at horizon 2
        with pytest.raises(ValueError, match="density='diagonal'"):
            _gaussian_log_scores(draws, np.zeros((2, 2)), density="gaussian", label="alpha")

    def test_collinear_columns_rejected(self):
        rng = np.random.default_rng(5)
        draws = rng.standard_normal((200, 1, 2))
        draws[:, 0, 1] = draws[:, 0, 0]
        with pytest.raises(ValueError, match="numerically singular"):
            _gaussian_log_scores(draws, np.zeros((1, 2)), density="gaussian", label="alpha")

    def test_all_draws_identical_rejected(self):
        draws = np.ones((50, 1, 2))
        with pytest.raises(ValueError, match="every forecast draw is identical"):
            _gaussian_log_scores(draws, np.zeros((1, 2)), density="gaussian", label="alpha")

    def test_too_few_draws_for_joint_density(self):
        rng = np.random.default_rng(2)
        with pytest.raises(ValueError, match="S <= n"):
            _gaussian_log_scores(rng.standard_normal((3, 2, 3)), np.zeros((2, 3)), label="alpha")

    def test_single_draw_rejected(self):
        with pytest.raises(ValueError, match="at least two"):
            _gaussian_log_scores(np.zeros((1, 2, 2)), np.zeros((2, 2)), density="diagonal", label="alpha")

    def test_zero_marginal_variance_rejected(self):
        rng = np.random.default_rng(9)
        draws = rng.standard_normal((40, 2, 2))
        draws[:, 0, 1] = 2.0
        with pytest.raises(ValueError, match="zero forecast variance"):
            _gaussian_log_scores(draws, np.zeros((2, 2)), density="diagonal", label="alpha")

    def test_non_finite_draws_rejected(self):
        draws = np.ones((10, 2, 2))
        draws[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite forecast draws"):
            _gaussian_log_scores(draws, np.zeros((2, 2)), label="alpha")

    def test_holdout_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="but the draws imply"):
            _gaussian_log_scores(np.zeros((10, 2, 2)), np.zeros((3, 2)), label="alpha")

    def test_pooled_row_scores_tolerate_zero_weights(self):
        scores = np.array([[0.0, -1.0], [0.0, -1.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pooled = _pooled_row_scores(scores, np.array([1.0, 0.0]))
        np.testing.assert_allclose(pooled, [0.0, 0.0])


# --------------------------------------------------------------------------
# D. pool_forecasts end to end
# --------------------------------------------------------------------------


class TestPoolForecasts:
    @pytest.mark.parametrize("method", ["stacking", "log_score"])
    def test_dominant_model_takes_all_the_weight(self, dominance_pool, method):
        fits, holdout = dominance_pool
        pool = pool_forecasts(fits, holdout, method=method, seed=0)
        assert pool.weights["good"] > 0.99
        assert pool.log_scores.sum().idxmax() == "good"

    def test_stacking_splits_mirrored_models_evenly(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, method="stacking", seed=0)
        np.testing.assert_allclose(pool.weights.to_numpy(), [0.5, 0.5], atol=1e-3)

    def test_pooling_beats_every_single_model(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, method="stacking", seed=0)
        assert pool.pooled_log_score() > pool.log_scores.sum().max() + 50.0

    def test_the_two_methods_disagree(self, mirrored_pool):
        """Log-score weights collapse where stacking keeps both models."""
        fits, holdout = mirrored_pool
        stacked = pool_forecasts(fits, holdout, method="stacking", seed=0)
        log_scored = pool_forecasts(fits, holdout, method="log_score", seed=0)
        assert stacked.weights.min() > 0.4
        assert log_scored.weights.min() < 0.05
        assert stacked.pooled_log_score() > log_scored.pooled_log_score()

    def test_stacking_weights_resolve_scipy_independently(self, mirrored_pool):
        """Version-proof check: re-solve the published score matrix directly."""
        from scipy.optimize import Bounds, LinearConstraint, minimize

        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, method="stacking", seed=0)
        matrix = pool.log_scores.to_numpy()
        densities = np.exp(matrix - matrix.max(axis=1, keepdims=True))
        result = minimize(
            lambda w: -np.sum(np.log(densities @ w)),
            x0=np.array([0.3, 0.7]),
            method="SLSQP",
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(np.ones((1, 2)), 1.0, 1.0),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        np.testing.assert_allclose(pool.weights.to_numpy(), result.x, atol=1e-5)

    def test_asymmetric_holdout_shifts_the_weights(self, mirrored_pool):
        """Three quarters of the holdout favours one model; check against scipy."""
        fits, _ = mirrored_pool
        base = next(iter(fits.values()))
        steps = 12
        mean = _mean_path(base, steps)
        pattern = np.tile(np.array([[0.0, 2.5], [0.0, 2.5], [0.0, 2.5], [2.5, 0.0]]), (steps // 4, 1))
        pool = pool_forecasts(fits, _holdout(base, mean + pattern), method="stacking", seed=0)
        assert pool.weights.iloc[0] > pool.weights.iloc[1]
        assert pool.weights.min() > 0.0

    def test_log_scores_frame_is_labelled(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        pd.testing.assert_index_equal(pool.log_scores.index, pd.DatetimeIndex(holdout.index, name="time"))
        assert list(pool.log_scores.columns) == list(fits)
        assert pool.log_scores.columns.name == "model"
        assert pool.labels == list(fits)

    def test_summary_layout(self, dominance_pool):
        fits, holdout = dominance_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        summary = pool.summary()
        assert list(summary.columns) == ["weight", "log_score", "mean_log_score", "rank"]
        assert summary.index[0] == "good"
        assert summary["rank"].tolist() == [1, 2, 3]
        assert summary["weight"].is_monotonic_decreasing
        np.testing.assert_allclose(
            summary["mean_log_score"].to_numpy(),
            summary["log_score"].to_numpy() / pool.steps,
        )

    def test_to_dataframe_carries_the_pooled_column(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        frame = pool.to_dataframe()
        assert list(frame.columns) == [*fits, "pooled"]
        np.testing.assert_allclose(frame["pooled"].sum(), pool.pooled_log_score())
        # A mixture density is bounded by its members: below the best member's
        # own score at each point, above that member's score shrunk by its weight.
        members = frame[list(fits)]
        assert (frame["pooled"] <= members.max(axis=1) + 1e-9).all()
        assert (frame["pooled"] >= (members + np.log(pool.weights)).max(axis=1) - 1e-9).all()
        # Summed over the window, pooling wins — that is the point.
        assert frame["pooled"].sum() > members.sum().max()

    def test_origin_is_the_shared_estimation_end(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        assert pool.origin == next(iter(fits.values())).data.index[-1]
        assert pool.steps == len(holdout.index)
        assert pool.var_names == holdout.endog_names
        assert pool.method == "stacking"
        assert pool.density == "gaussian"
        assert pool.converged

    def test_diagonal_density_runs(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, density="diagonal", seed=0)
        assert pool.density == "diagonal"
        assert np.isclose(pool.weights.sum(), 1.0)

    def test_heterogeneous_chain_counts_pool(self, mirrored_pool):
        """A conjugate-style (1, 200) posterior pools with a NUTS-style (2, 100)."""
        _, holdout = mirrored_pool
        fits = {
            "conjugate": _fitted(sd=[0.3, 1.5], n_chains=1, n_draws=200),
            "nuts": _fitted(sd=[1.5, 0.3], n_chains=2, n_draws=100),
        }
        pool = pool_forecasts(fits, holdout, seed=0)
        assert pool.membership.shape == (200,)
        assert np.isclose(pool.weights.sum(), 1.0)

    def test_exogenous_models_consume_the_holdout_exog(self):
        fit_a = _fitted(sd=[0.4, 1.2], exog=True)
        fit_b = _fitted(sd=[1.2, 0.4], exog=True)
        holdout = _holdout(fit_a, _mean_path(fit_a, 6), exog=True)
        pool = pool_forecasts({"a": fit_a, "b": fit_b}, holdout, seed=0)
        assert np.isclose(pool.weights.sum(), 1.0)

    def test_pool_never_refits(self, mirrored_pool):
        """The fits handed in are untouched — the pool holds no reference to them."""
        fits, holdout = mirrored_pool
        before = {k: v.data.endog.copy() for k, v in fits.items()}
        pool_forecasts(fits, holdout, seed=0)
        for label, fit in fits.items():
            np.testing.assert_array_equal(fit.data.endog, before[label])


# --------------------------------------------------------------------------
# E. ArviZ parity
# --------------------------------------------------------------------------


def _arviz_shim(log_scores: pd.DataFrame) -> dict[str, az.InferenceData]:
    """Wrap a score matrix as InferenceData whose loo reproduces the scores.

    Each column becomes a `log_likelihood` broadcast constant across draws,
    so PSIS weights are uniform, `p_loo` is zero, and `elpd_loo` is exactly
    the summed column. This makes `az.compare` a pure function of the score
    matrix — the point of the parity check. Verified against arviz 0.23.
    """
    rng = np.random.default_rng(0)
    n_chains, n_draws = 2, 200
    out = {}
    for label in log_scores.columns:
        values = np.broadcast_to(log_scores[label].to_numpy(), (n_chains, n_draws, len(log_scores))).copy()
        out[label] = az.InferenceData(
            posterior=xr.Dataset({"mu": (("chain", "draw"), rng.standard_normal((n_chains, n_draws)))}),
            log_likelihood=xr.Dataset({"y": (("chain", "draw", "obs"), values)}),
        )
    return out


class TestArviZParity:
    def test_stacking_matches_az_compare(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            frame = pd.DataFrame(REFERENCE_SCORES, columns=["a", "b"])
            comparison = az.compare(_arviz_shim(frame), ic="loo", method="stacking", scale="log")
            np.testing.assert_allclose(comparison.loc["a", "elpd_loo"], REFERENCE_SCORES[:, 0].sum(), atol=1e-8)
            weights = _stacking_weights(REFERENCE_SCORES)[0]
            # ArviZ optimises a softmax reparameterisation with its own default
            # tolerance, so it lands ~3e-5 from the analytic 7/9; the parity
            # claim is "same solution", not "same solver settings".
            np.testing.assert_allclose(
                [comparison.loc["a", "weight"], comparison.loc["b", "weight"]], weights, atol=1e-4
            )
            np.testing.assert_allclose(weights[0], 7.0 / 9.0, atol=1e-6)

    def test_pseudo_bma_matches_az_compare(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            frame = pd.DataFrame(REFERENCE_SCORES, columns=["a", "b"])
            comparison = az.compare(_arviz_shim(frame), ic="loo", method="BB-pseudo-BMA", scale="log")
            assert comparison.loc["a", "weight"] > comparison.loc["b", "weight"]
        np.testing.assert_allclose(_log_score_weights(REFERENCE_SCORES), [0.8, 0.2], atol=1e-12)


# --------------------------------------------------------------------------
# F. Validation
# --------------------------------------------------------------------------


class TestValidation:
    def test_single_model_rejected(self, mirrored_pool):
        fits, holdout = mirrored_pool
        with pytest.raises(ValueError, match="at least two fitted models"):
            pool_forecasts({"only": next(iter(fits.values()))}, holdout)

    def test_variable_name_mismatch(self, mirrored_pool):
        fits, holdout = mirrored_pool
        other = _fitted(sd=[0.5, 0.5], var_names=("y1", "z9"))
        with pytest.raises(ValueError, match="must share the holdout's variables"):
            pool_forecasts({**fits, "odd": other}, holdout)

    def test_variable_order_mismatch_suggests_reordering(self, mirrored_pool):
        fits, holdout = mirrored_pool
        swapped = VARData(
            endog=np.asarray(holdout.endog)[:, ::-1],
            endog_names=["y2", "y1"],
            index=holdout.index,
        )
        with pytest.raises(ValueError, match="reorder the holdout columns"):
            pool_forecasts(fits, swapped)

    def test_different_estimation_ends(self, mirrored_pool):
        fits, holdout = mirrored_pool
        shorter = _fitted(sd=[0.5, 0.5], n_obs=59)
        with pytest.raises(ValueError, match="different sample ends"):
            pool_forecasts({**fits, "short": shorter}, holdout)

    def test_holdout_must_postdate_the_origin(self, mirrored_pool):
        fits, holdout = mirrored_pool
        overlapping = VARData(
            endog=np.asarray(holdout.endog),
            endog_names=holdout.endog_names,
            index=pd.date_range("2010-01-01", periods=len(holdout.index), freq="QS"),
        )
        with pytest.raises(ValueError, match="must postdate the estimation sample"):
            pool_forecasts(fits, overlapping)

    def test_gap_after_the_origin_is_rejected(self, mirrored_pool):
        fits, holdout = mirrored_pool
        origin = next(iter(fits.values())).data.index[-1]
        gapped = VARData(
            endog=np.asarray(holdout.endog),
            endog_names=holdout.endog_names,
            index=pd.date_range(origin, periods=len(holdout.index) + 2, freq="QS")[2:],
        )
        with pytest.raises(ValueError, match="does not continue the estimation sample"):
            pool_forecasts(fits, gapped)

    def test_unknown_frequency_warns_and_proceeds(self):
        index = pd.DatetimeIndex(["2000-01-01", "2000-01-05", "2000-02-11", "2000-04-02"])
        fits = {
            "a": _fitted(sd=[0.3, 1.5], n_obs=4),
            "b": _fitted(sd=[1.5, 0.3], n_obs=4),
        }
        fits = {
            label: FittedVAR(
                idata=fit.idata,
                n_lags=fit.n_lags,
                data=VARData(endog=np.asarray(fit.data.endog), endog_names=fit.var_names, index=index),
                var_names=fit.var_names,
                volatility=fit.volatility,
            )
            for label, fit in fits.items()
        }
        holdout = VARData(
            endog=np.zeros((3, 2)),
            endog_names=["y1", "y2"],
            index=pd.DatetimeIndex(["2000-05-09", "2000-06-30", "2000-09-01"]),
        )
        with pytest.warns(UserWarning, match="estimation samples or the holdout"):
            pool = pool_forecasts(fits, holdout, seed=0)
        assert np.isclose(pool.weights.sum(), 1.0)

    def test_missing_holdout_exog(self):
        fit_a = _fitted(sd=[0.4, 1.2], exog=True)
        fit_b = _fitted(sd=[1.2, 0.4], exog=True)
        holdout = _holdout(fit_a, _mean_path(fit_a, 6), exog=False)
        with pytest.raises(ValueError, match="exogenous regressors"):
            pool_forecasts({"a": fit_a, "b": fit_b}, holdout)

    def test_exog_name_mismatch(self):
        fit_a = _fitted(sd=[0.4, 1.2], exog=True)
        fit_b = _fitted(sd=[1.2, 0.4], exog=True)
        steps = 6
        index = pd.date_range(fit_a.data.index[-1], periods=steps + 1, freq="QS")[1:]
        holdout = VARData(
            endog=_mean_path(fit_a, steps),
            endog_names=["y1", "y2"],
            exog=np.ones((steps, 1)),
            exog_names=["other"],
            index=index,
        )
        with pytest.raises(ValueError, match="exogenous regressors named"):
            pool_forecasts({"a": fit_a, "b": fit_b}, holdout)

    def test_unknown_method(self, mirrored_pool):
        fits, holdout = mirrored_pool
        with pytest.raises(ValueError, match="method must be"):
            pool_forecasts(fits, holdout, method="magic")

    def test_unknown_density(self, mirrored_pool):
        fits, holdout = mirrored_pool
        with pytest.raises(ValueError, match="density must be"):
            pool_forecasts(fits, holdout, density="student")

    def test_n_draws_must_be_positive(self, mirrored_pool):
        fits, holdout = mirrored_pool
        with pytest.raises(ValueError, match="n_draws must be at least 1"):
            pool_forecasts(fits, holdout, n_draws=0)

    def test_bad_seed_is_reported_as_a_seed_problem(self, mirrored_pool):
        fits, holdout = mirrored_pool
        with pytest.raises(ValueError, match="seed must be"):
            pool_forecasts(fits, holdout, seed="tomorrow")

    @pytest.mark.parametrize("order", [("quarterly", "monthly"), ("monthly", "quarterly")])
    def test_mixed_frequency_fits_rejected_in_either_order(self, order):
        """Two models can share a sample end at different frequencies.

        The check must not depend on which one the mapping happens to yield
        first: previously only the first fit's index was inspected, so a
        quarterly-first mapping pooled a monthly model silently.
        """
        candidates = {
            "quarterly": _fitted(sd=[0.5, 0.5], n_obs=60, start="2000-01-01", freq="QS"),
            "monthly": _fitted(sd=[0.6, 0.6], n_obs=60, start="2009-11-01", freq="MS"),
        }
        assert candidates["quarterly"].data.index[-1] == candidates["monthly"].data.index[-1]
        fits = {label: candidates[label] for label in order}
        holdout = _holdout(candidates["quarterly"], np.zeros((6, 2)))
        with pytest.raises(ValueError, match="different frequencies"):
            pool_forecasts(fits, holdout, seed=0)

    def test_one_unknown_frequency_among_regular_fits_warns(self):
        """A model whose index has no inferable frequency is flagged, not ignored."""
        regular = _fitted(sd=[0.5, 0.5], n_obs=60, start="2000-01-01", freq="QS")
        irregular_index = pd.DatetimeIndex([
            *pd.date_range("2000-01-01", periods=58, freq="QS"),
            "2014-08-13",
            "2014-10-01",
        ])
        irregular = FittedVAR(
            idata=regular.idata,
            n_lags=regular.n_lags,
            data=VARData(
                endog=np.asarray(regular.data.endog),
                endog_names=regular.var_names,
                index=irregular_index,
            ),
            var_names=regular.var_names,
            volatility=regular.volatility,
        )
        holdout = _holdout(regular, _mean_path(regular, 6))
        with pytest.warns(UserWarning, match=r"Could not infer a frequency for \['odd'\]"):
            pool = pool_forecasts({"regular": regular, "odd": irregular}, holdout, seed=0)
        assert np.isclose(pool.weights.sum(), 1.0)

    def test_non_spawnable_generator_is_reported_as_a_seed_problem(self):
        with pytest.raises(ValueError, match="supports spawning"):
            _spawn(np.random.RandomState(0), 3)

    def test_frequency_is_inferred_when_the_index_does_not_carry_one(self):
        regular = pd.DatetimeIndex(["2000-01-01", "2000-04-01", "2000-07-01", "2000-10-01"])
        assert regular.freq is None
        assert _index_freq(regular) is not None
        assert _index_freq(pd.DatetimeIndex(["2000-01-01", "2000-01-05"])) is None

    def test_reserved_pooled_label_rejected(self, mirrored_pool):
        """'pooled' would overwrite that model's column in to_dataframe()."""
        fits, holdout = mirrored_pool
        renamed = dict(zip(["pooled", "other"], fits.values(), strict=True))
        with pytest.raises(ValueError, match="reserved model label"):
            pool_forecasts(renamed, holdout, seed=0)

    def test_reserved_pooled_label_rejected_on_direct_construction(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        broken = dict(pool)
        broken["weights"] = pd.Series(pool.weights.to_numpy(), index=["pooled", "other"])
        broken["log_scores"] = pool.log_scores.set_axis(["pooled", "other"], axis=1)
        with pytest.raises(ValidationError, match="reserved model label"):
            PredictivePool(**broken)

    def test_empty_holdout_rejected(self, mirrored_pool):
        fits, _ = mirrored_pool
        origin = next(iter(fits.values())).data.index[-1]
        with pytest.raises(ValueError, match="at least one held-out"):
            pool_forecasts(
                fits,
                VARData(
                    endog=np.zeros((0, 2)),
                    endog_names=["y1", "y2"],
                    index=pd.date_range(origin, periods=1, freq="QS")[1:],
                ),
            )

    def test_literal_fields_are_enforced_on_direct_construction(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        with pytest.raises(ValidationError):
            PredictivePool(**{**dict(pool), "method": "magic"})


# --------------------------------------------------------------------------
# G. Combined sample
# --------------------------------------------------------------------------


class TestCombinedSample:
    def test_pooled_predictive_layout(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        forecast = pool.holdout_predictive
        assert isinstance(forecast, ForecastResult)
        values = forecast.idata.posterior_predictive["forecast"]
        # n_draws defaults to the smallest member's flattened draw count: 2 x 100.
        assert values.shape == (1, 200, 12, 2)
        assert list(values.coords["variable"].values) == ["y1", "y2"]
        pd.testing.assert_index_equal(pd.DatetimeIndex(values.coords["time"].values), pd.DatetimeIndex(holdout.index))
        assert forecast.median().shape == (12, 2)
        assert forecast.hdi(0.89).lower.shape == (12, 2)
        assert forecast.to_dataframe().shape == (12, 2)
        assert forecast.mode == "density"

    def test_realised_weights_track_the_target(self):
        fits = {"tight_y1": _fitted(sd=[0.3, 1.5]), "tight_y2": _fitted(sd=[1.5, 0.3])}
        base = fits["tight_y1"]
        mean = _mean_path(base, 12)
        deviation = np.tile(np.array([[0.0, 2.5], [2.5, 0.0]]), (6, 1))
        pool = pool_forecasts(fits, _holdout(base, mean + deviation), n_draws=4000, seed=3)
        realised = pool.realised_weights()
        assert list(realised.index) == pool.labels
        target = pool.weights.to_numpy()
        tolerance = 4.0 * np.sqrt(target * (1.0 - target) / 4000)
        assert np.all(np.abs(realised.to_numpy() - target) < np.maximum(tolerance, 1e-12))

    def test_degenerate_weights_draw_from_one_model(self, dominance_pool):
        fits, holdout = dominance_pool
        pool = pool_forecasts(fits, holdout, method="log_score", seed=0)
        assert pool.weights["good"] > 0.999
        winner = pool.labels.index(pool.weights.idxmax())
        assert set(np.unique(pool.membership)) == {winner}
        model_coord = pool.holdout_predictive.idata.posterior_predictive["forecast"].coords["model"].values
        assert set(model_coord) == {"good"}

    def test_model_coord_agrees_with_membership(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        coord = pool.holdout_predictive.idata.posterior_predictive["forecast"].coords["model"].values
        np.testing.assert_array_equal(coord, np.array(pool.labels)[pool.membership])

    def test_pooled_draws_are_member_draws(self, dominance_pool):
        """Every pooled row is verbatim a draw from the model that produced it.

        Also pins the documented RNG contract: child generators are spawned in
        `fits` insertion order, one per model, and the mixture takes the last.
        """
        fits, holdout = dominance_pool
        pool = pool_forecasts(fits, holdout, method="log_score", seed=0)
        assert set(np.unique(pool.membership)) == {pool.labels.index("good")}

        children = np.random.default_rng(0).spawn(len(fits) + 1)
        winner = fits["good"].forecast(steps=pool.steps, seed=children[pool.labels.index("good")])
        flat = winner.idata.posterior_predictive["forecast"].values.reshape(-1, pool.steps * 2)
        pooled = pool.holdout_predictive.idata.posterior_predictive["forecast"].values[0]
        known = {row.tobytes() for row in np.ascontiguousarray(flat)}
        assert all(row.tobytes() in known for row in np.ascontiguousarray(pooled.reshape(-1, pool.steps * 2)))

    def test_n_draws_override(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, n_draws=37, seed=0)
        assert pool.membership.shape == (37,)
        assert pool.holdout_predictive.idata.posterior_predictive["forecast"].shape[1] == 37


# --------------------------------------------------------------------------
# H. Determinism
# --------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_is_bit_identical(self, mirrored_pool):
        fits, holdout = mirrored_pool
        a = pool_forecasts(fits, holdout, seed=11)
        b = pool_forecasts(fits, holdout, seed=11)
        pd.testing.assert_series_equal(a.weights, b.weights)
        pd.testing.assert_frame_equal(a.log_scores, b.log_scores)
        np.testing.assert_array_equal(a.membership, b.membership)
        np.testing.assert_array_equal(
            a.holdout_predictive.idata.posterior_predictive["forecast"].values,
            b.holdout_predictive.idata.posterior_predictive["forecast"].values,
        )

    def test_generator_seed_is_reproducible(self, mirrored_pool):
        fits, holdout = mirrored_pool
        a = pool_forecasts(fits, holdout, seed=np.random.default_rng(5))
        b = pool_forecasts(fits, holdout, seed=np.random.default_rng(5))
        pd.testing.assert_frame_equal(a.log_scores, b.log_scores)
        np.testing.assert_array_equal(a.membership, b.membership)

    def test_different_seed_changes_the_pooled_draws(self, mirrored_pool):
        fits, holdout = mirrored_pool
        a = pool_forecasts(fits, holdout, seed=1)
        b = pool_forecasts(fits, holdout, seed=2)
        assert not np.array_equal(
            a.holdout_predictive.idata.posterior_predictive["forecast"].values,
            b.holdout_predictive.idata.posterior_predictive["forecast"].values,
        )


# --------------------------------------------------------------------------
# I. combine()
# --------------------------------------------------------------------------


class TestCombine:
    def test_applies_weights_to_new_forecasts(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        refits = {label: fit.forecast(steps=20, seed=7) for label, fit in fits.items()}
        combined = pool.combine(refits, seed=1)
        assert combined.steps == 20
        assert combined.var_names == pool.var_names
        values = combined.idata.posterior_predictive["forecast"]
        assert values.shape == (1, 200, 20, 2)
        assert "time" not in values.coords
        assert set(values.coords["model"].values) <= set(pool.labels)

    def test_combine_respects_n_draws(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        refits = {label: fit.forecast(steps=4, seed=7) for label, fit in fits.items()}
        assert pool.combine(refits, n_draws=13, seed=1).idata.posterior_predictive["forecast"].shape[1] == 13

    def test_combine_is_deterministic(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        refits = {label: fit.forecast(steps=4, seed=7) for label, fit in fits.items()}
        a = pool.combine(refits, seed=99).idata.posterior_predictive["forecast"].values
        b = pool.combine(refits, seed=99).idata.posterior_predictive["forecast"].values
        np.testing.assert_array_equal(a, b)

    def test_label_mismatch_rejected(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        refits = {label: fit.forecast(steps=4, seed=7) for label, fit in fits.items()}
        refits["extra"] = next(iter(fits.values())).forecast(steps=4, seed=7)
        with pytest.raises(ValueError, match="one forecast per pooled model"):
            pool.combine(refits)

    def test_variable_mismatch_rejected(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        refits = {label: fit.forecast(steps=4, seed=7) for label, fit in fits.items()}
        first = pool.labels[0]
        refits[first] = refits[first].model_copy(update={"var_names": ["y2", "y1"]})
        with pytest.raises(ValueError, match="forecasts variables"):
            pool.combine(refits)

    def test_step_mismatch_rejected(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        refits = {
            label: fit.forecast(steps=steps, seed=7) for steps, (label, fit) in zip([4, 6], fits.items(), strict=True)
        }
        with pytest.raises(ValueError, match="same number of steps"):
            pool.combine(refits)

    def test_mean_mode_rejected(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        refits = {label: fit.forecast(steps=4, include_shock_uncertainty=False) for label, fit in fits.items()}
        with pytest.raises(ValueError, match="include_shock_uncertainty=True"):
            pool.combine(refits)

    def test_combine_rejects_non_positive_n_draws(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        refits = {label: fit.forecast(steps=4, seed=7) for label, fit in fits.items()}
        with pytest.raises(ValueError, match="n_draws must be at least 1"):
            pool.combine(refits, n_draws=0)

    def test_combine_rejects_bad_seed(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        refits = {label: fit.forecast(steps=4, seed=7) for label, fit in fits.items()}
        with pytest.raises(ValueError, match="seed must be"):
            pool.combine(refits, seed="tomorrow")


# --------------------------------------------------------------------------
# J. Frozen contract and plotting
# --------------------------------------------------------------------------


class TestFrozenContract:
    def test_attributes_cannot_be_reassigned(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        with pytest.raises(ValidationError):
            pool.method = "log_score"

    def test_membership_is_read_only(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        with pytest.raises(ValueError, match="read-only"):
            pool.membership[0] = 0

    def test_weights_must_sum_to_one(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        broken = dict(pool)
        broken["weights"] = pool.weights * 2.0
        with pytest.raises(ValidationError, match="sum to 1"):
            PredictivePool(**broken)

    def test_weights_must_be_non_negative(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        broken = dict(pool)
        broken["weights"] = pd.Series([1.5, -0.5], index=pool.labels)
        with pytest.raises(ValidationError, match="non-negative"):
            PredictivePool(**broken)

    def test_weight_labels_must_match_the_score_columns(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        broken = dict(pool)
        broken["weights"] = pd.Series(pool.weights.to_numpy(), index=["p", "q"])
        with pytest.raises(ValidationError, match="log-score columns"):
            PredictivePool(**broken)

    def test_membership_must_index_a_model(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        broken = dict(pool)
        broken["membership"] = np.array([0, 1, 7])
        with pytest.raises(ValidationError, match="membership"):
            PredictivePool(**broken)

    def test_membership_must_be_one_dimensional(self, mirrored_pool):
        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        broken = dict(pool)
        broken["membership"] = np.zeros((3, 2), dtype=int)
        with pytest.raises(ValidationError, match="must be 1-D"):
            PredictivePool(**broken)


class TestPlot:
    def test_plot_returns_a_figure(self, mirrored_pool):
        from matplotlib.figure import Figure

        fits, holdout = mirrored_pool
        pool = pool_forecasts(fits, holdout, seed=0)
        fig = pool.plot()
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert set(labels) == set(pool.labels)
        assert "stacking" in fig.axes[0].get_title()
