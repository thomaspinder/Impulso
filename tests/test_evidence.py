"""Tests for conjugate model evidence and Bayes-factor comparison.

Three things are pinned:

1. The value `ConjugateVAR.fit` attaches to `FittedVAR.evidence` is the conjugate log
   marginal likelihood *at the selected hyperparameters*, verified against independent
   sequential-predictive-t and matrix-t evaluations (and, for a volatility break, against
   the rescaled reference plus an explicitly added Jacobian).
2. The metadata that decides comparability — effective sample, window, response digest.
3. The comparison arithmetic: Bayes factors, posterior model probabilities, saturation
   instead of overflow, and every rejection path.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.special import multigammaln  # ty: ignore[unresolved-import]
from scipy.stats import multivariate_t as mvt

from impulso import ConjugateVAR, NIWPrior, PandemicBreak, VARData, compare_evidence
from impulso._conjugate import ar1_residual_sd
from impulso.evidence import EvidenceComparison, ModelEvidence, _response_digest
from impulso.fitted import FittedVAR

# --------------------------------------------------------------------------- helpers


def _series(seed: int, n: int, t_full: int) -> np.ndarray:
    """Small persistent synthetic series of shape (t_full, n)."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((t_full, n)), axis=0) + 5.0


def _var_data(seed: int = 3, n: int = 2, t_full: int = 24) -> VARData:
    """VARData over a monthly index, small enough for fast fits."""
    y = _series(seed, n, t_full)
    index = pd.date_range("2018-01-01", periods=t_full, freq="MS")
    names = [f"y{i + 1}" for i in range(n)]
    return VARData(endog=y, endog_names=names, index=index)


def _design(y: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """(Y, X) with a leading constant column, matching the engine's lag order."""
    t_full, _ = y.shape
    t_obs = t_full - n_lags
    cols = [np.ones((t_obs, 1))]
    for ell in range(1, n_lags + 1):
        cols.append(y[n_lags - ell : t_full - ell])
    return y[n_lags:], np.hstack(cols)


def _sequential_logml(Y, X, Yd, Xd) -> float:
    """Independent log ML as a product of one-step conjugate predictive multivariate-t
    densities (recursive Bayesian updating; well-conditioned, no batch determinants)."""
    n = Y.shape[1]
    omega = np.linalg.inv(Xd.T @ Xd)
    b = omega @ Xd.T @ Yd
    r0 = Yd - Xd @ b
    psi = r0.T @ r0
    nu = float(n + 2)
    total = 0.0
    for t in range(Y.shape[0]):
        x, y = X[t], Y[t]
        pred_mean = b.T @ x
        s = 1.0 + x @ omega @ x
        df = nu - n + 1.0
        total += mvt.logpdf(y, loc=pred_mean, shape=(s / df) * psi, df=df)
        omega_x = omega @ x
        omega = omega - np.outer(omega_x, omega_x) / s
        resid = y - pred_mean
        b = b + np.outer(omega_x, resid) / s
        psi = psi + np.outer(resid, resid) / s
        nu += 1.0
    return float(total)


def _matrix_t_logml(Y, X, Yd, Xd) -> float:
    """Independent log ML via the matrix-t density (T x T determinant form)."""
    n, t_obs = Y.shape[1], Y.shape[0]
    d0, d_t = n + 2, n + 2 + t_obs
    omega0 = np.linalg.inv(Xd.T @ Xd)
    b0 = omega0 @ Xd.T @ Yd
    psi0 = (Yd - Xd @ b0).T @ (Yd - Xd @ b0)
    a_mat = np.eye(t_obs) + X @ omega0 @ X.T
    err = Y - X @ b0
    quad = psi0 + err.T @ np.linalg.solve(a_mat, err)
    ld = lambda m: np.linalg.slogdet(m)[1]
    return float(
        -0.5 * n * t_obs * np.log(np.pi)
        + multigammaln(d_t / 2, n)
        - multigammaln(d0 / 2, n)
        - 0.5 * n * ld(a_mat)
        + 0.5 * d0 * ld(psi0)
        - 0.5 * d_t * ld(quad)
    )


def _reference_logml(
    data: VARData,
    n_lags: int,
    prior: NIWPrior,
    tightness: float,
    log_scales: np.ndarray | None = None,
) -> float:
    """Rebuild the conjugate log ML independently: rescale, then sequential predictive-t.

    The change-of-variables Jacobian ``-n * sum(log_scales)`` is added here explicitly,
    so a passing test pins that the library adds it exactly once.
    """
    y = data.endog
    response, regressors = _design(y, n_lags)
    dummies_y, dummies_x = prior.build_dummies(y, n_lags, ar1_residual_sd(y), tightness=tightness)
    jacobian = 0.0
    if log_scales is not None:
        weights = np.exp(-np.asarray(log_scales, dtype=float).ravel())[:, None]
        response = response * weights
        regressors = regressors * weights
        jacobian = -y.shape[1] * float(np.asarray(log_scales).sum())
    return _sequential_logml(response, regressors, dummies_y, dummies_x) + jacobian


def _evidence(name_suffix: str = "", **overrides) -> ModelEvidence:
    """A hand-built ModelEvidence; every field overridable for the mismatch tests."""
    fields = {
        "log_marginal_likelihood": -100.0,
        "n_obs": 20,
        "n_vars": 2,
        "var_names": ["y1", "y2"],
        "n_lags": 1,
        "volatility": None,
        "hyperparameters": {},
        "sample_start": pd.Timestamp("2018-02-01"),
        "sample_end": pd.Timestamp("2019-09-01"),
        "sample_digest": f"digest{name_suffix}",
    }
    fields.update(overrides)
    return ModelEvidence(**fields)


# ------------------------------------------------------------- A: the fitted-side value


class TestEvidenceValue:
    """The number on `FittedVAR.evidence` is the conjugate log ML the model implies."""

    def test_fixed_prior_matches_sequential_reference(self):
        data = _var_data()
        prior = NIWPrior(tightness=0.25)
        fitted = ConjugateVAR(lags=1, prior=prior, draws=4, tune=0, seed=0).fit(data)

        expected = _reference_logml(data, 1, prior, prior.tightness)
        assert fitted.evidence is not None
        assert fitted.evidence.log_marginal_likelihood == pytest.approx(expected, abs=1e-8)

    def test_fixed_prior_matches_matrix_t_reference(self):
        data = _var_data(seed=5)
        prior = NIWPrior(tightness=0.3, sum_of_coefficients=1.0)
        fitted = ConjugateVAR(lags=2, prior=prior, draws=4, tune=0, seed=1).fit(data)

        response, regressors = _design(data.endog, 2)
        dummies = prior.build_dummies(data.endog, 2, ar1_residual_sd(data.endog), tightness=prior.tightness)
        expected = _matrix_t_logml(response, regressors, *dummies)
        assert fitted.evidence.log_marginal_likelihood == pytest.approx(expected, rel=1e-8)

    def test_selected_prior_evidence_is_evaluated_at_the_selected_lambda(self):
        data = _var_data(seed=7)
        prior = NIWPrior(select=True)
        fitted = ConjugateVAR(lags=1, prior=prior, draws=4, tune=0, seed=2).fit(data)

        lam = fitted.evidence.hyperparameters["lambda_"]
        expected = _reference_logml(data, 1, prior, lam)
        assert fitted.evidence.log_marginal_likelihood == pytest.approx(expected, abs=1e-8)

        # A wrong lambda gives a materially different value, so the test above has teeth.
        off = _reference_logml(data, 1, prior, lam + 0.1)
        assert abs(off - expected) > 1e-4

    def test_volatility_break_evidence_includes_the_jacobian(self):
        data = _var_data(seed=11, t_full=26)
        prior = NIWPrior(tightness=0.2)
        break_at = 8
        volatility = PandemicBreak(start=break_at)
        fitted = ConjugateVAR(lags=1, prior=prior, volatility=volatility, draws=4, tune=0, seed=3).fit(data)

        theta = fitted.evidence.hyperparameters
        log_scales = volatility.log_scales(theta, data.endog.shape[0] - 1)
        expected = _reference_logml(data, 1, prior, prior.tightness, log_scales=log_scales)
        assert fitted.evidence.log_marginal_likelihood == pytest.approx(expected, abs=1e-8)
        # The Jacobian is a real, nonzero contribution on this path.
        assert abs(float(np.sum(log_scales))) > 0.0

    def test_evidence_is_seed_independent(self):
        data = _var_data(seed=13)
        prior = NIWPrior(select=True)
        a = ConjugateVAR(lags=1, prior=prior, draws=4, tune=0, seed=0).fit(data)
        b = ConjugateVAR(lags=1, prior=prior, draws=4, tune=0, seed=7).fit(data)

        assert a.evidence.log_marginal_likelihood == b.evidence.log_marginal_likelihood
        assert a.evidence.hyperparameters == b.evidence.hyperparameters

    def test_hand_built_fitted_var_has_no_evidence(self, synthetic_idata_2v, var_data_2v):
        from impulso.volatility import Constant

        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        assert fitted.evidence is None


# ------------------------------------------------------------------------ B: metadata


class TestEvidenceMetadata:
    """The comparability metadata describes the sample the model actually saw."""

    def test_metadata_for_a_fixed_homoscedastic_fit(self):
        data = _var_data(t_full=24)
        fitted = ConjugateVAR(lags=2, prior=NIWPrior(), draws=4, tune=0, seed=0).fit(data)
        ev = fitted.evidence

        assert ev.n_obs == 24 - 2
        assert ev.n_vars == 2
        assert ev.var_names == ["y1", "y2"]
        assert ev.n_lags == 2
        assert ev.volatility is None
        assert ev.hyperparameters == {}
        assert ev.sample_start == data.index[2]
        assert ev.sample_end == data.index[-1]

    def test_metadata_for_a_selected_break_fit(self):
        data = _var_data(seed=17, t_full=26)
        fitted = ConjugateVAR(
            lags=1,
            prior=NIWPrior(select=True),
            volatility=PandemicBreak(start=8),
            draws=4,
            tune=0,
            seed=0,
        ).fit(data)
        ev = fitted.evidence

        assert ev.volatility == "pandemic_break"
        assert set(ev.hyperparameters) == {"lambda_", "s_march", "s_april", "s_may", "rho"}

    def test_digest_is_stable_and_sensitive(self):
        data = _var_data()
        names = data.endog_names
        base = _response_digest(data.endog, names, 1)

        assert _response_digest(data.endog.copy(), names, 1) == base
        assert _response_digest(data.endog * 1.001, names, 1) != base

    def test_digest_is_invariant_to_column_permutation(self):
        data = _var_data(n=3)
        names = data.endog_names
        order = [2, 0, 1]
        permuted = data.endog[:, order]
        permuted_names = [names[i] for i in order]

        assert _response_digest(permuted, permuted_names, 1) == _response_digest(data.endog, names, 1)

    def test_aligned_lag_orders_share_sample_metadata(self):
        data = _var_data(t_full=24)
        trimmed = VARData(endog=data.endog[1:], endog_names=data.endog_names, index=data.index[1:])

        fit_p2 = ConjugateVAR(lags=2, prior=NIWPrior(), draws=4, tune=0, seed=0).fit(data)
        fit_p1 = ConjugateVAR(lags=1, prior=NIWPrior(), draws=4, tune=0, seed=0).fit(trimmed)

        assert fit_p1.evidence.n_obs == fit_p2.evidence.n_obs
        assert fit_p1.evidence.sample_start == fit_p2.evidence.sample_start
        assert fit_p1.evidence.sample_end == fit_p2.evidence.sample_end
        assert fit_p1.evidence.sample_digest == fit_p2.evidence.sample_digest


# ---------------------------------------------------------------------- C: comparison


class TestComparison:
    """Bayes-factor arithmetic on hand-built evidences — no fitting involved."""

    def test_bayes_factor_from_a_known_log_difference(self):
        a = _evidence(log_marginal_likelihood=-100.0)
        b = _evidence(log_marginal_likelihood=-100.0 + np.log(10.0))
        comparison = compare_evidence(base=a, alt=b)

        assert comparison.bayes_factor("alt") == pytest.approx(10.0, rel=1e-12)
        assert comparison.log10_bayes_factor("alt") == pytest.approx(1.0, rel=1e-12)
        assert comparison.log_bayes_factor("alt") == pytest.approx(np.log(10.0), rel=1e-12)

    def test_reference_defaults_to_the_first_fit_and_inverts(self):
        a = _evidence(log_marginal_likelihood=-100.0)
        b = _evidence(log_marginal_likelihood=-97.0)
        comparison = compare_evidence(base=a, alt=b)

        assert comparison.reference == "base"
        assert comparison.log_bayes_factor("base") == 0.0
        assert comparison.bayes_factor("alt", against="base") == pytest.approx(
            1.0 / comparison.bayes_factor("base", against="alt"), rel=1e-12
        )
        assert comparison.best == "alt"

    def test_to_dataframe_shape_order_and_reference_row(self):
        a = _evidence(log_marginal_likelihood=-100.0)
        b = _evidence(log_marginal_likelihood=-97.0, n_lags=2)
        frame = compare_evidence(base=a, alt=b).to_dataframe()

        assert frame.index.name == "model"
        assert list(frame.index) == ["base", "alt"]
        assert list(frame.columns) == [
            "log_marginal_likelihood",
            "log_bayes_factor",
            "log10_bayes_factor",
            "bayes_factor",
            "posterior_probability",
            "n_obs",
            "n_vars",
            "n_lags",
            "volatility",
        ]
        assert frame.loc["base", "log_bayes_factor"] == 0.0
        assert frame.loc["base", "bayes_factor"] == 1.0
        assert frame["posterior_probability"].sum() == pytest.approx(1.0, rel=1e-12)
        assert list(frame["n_lags"]) == [1, 2]

    def test_to_dataframe_accepts_an_explicit_reference(self):
        a = _evidence(log_marginal_likelihood=-100.0)
        b = _evidence(log_marginal_likelihood=-97.0)
        frame = compare_evidence(base=a, alt=b).to_dataframe(reference="alt")

        assert frame.loc["alt", "log_bayes_factor"] == 0.0
        assert frame.loc["base", "log_bayes_factor"] == pytest.approx(-3.0, rel=1e-12)

    def test_extreme_log_differences_saturate_instead_of_overflowing(self):
        a = _evidence(log_marginal_likelihood=-1e4)
        b = _evidence(log_marginal_likelihood=1e4)
        comparison = compare_evidence(base=a, alt=b)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert comparison.bayes_factor("alt") == np.inf
            assert comparison.bayes_factor("base", against="alt") == 0.0
            assert comparison.log_bayes_factor("alt") == pytest.approx(2e4, rel=1e-12)
            assert comparison.log10_bayes_factor("alt") == pytest.approx(2e4 / np.log(10.0), rel=1e-12)
            probs = comparison.posterior_probabilities()
        assert probs == {"base": 0.0, "alt": 1.0}

    def test_unknown_model_name_raises_key_error(self):
        comparison = compare_evidence(base=_evidence(), alt=_evidence())
        with pytest.raises(KeyError, match="missing"):
            comparison.log_bayes_factor("missing")

    def test_single_fit_is_rejected(self):
        with pytest.raises(ValueError, match="at least two"):
            compare_evidence(only=_evidence())

    def test_mismatched_variables_are_rejected(self):
        a = _evidence()
        b = _evidence(var_names=["y1", "y3"])
        with pytest.raises(ValueError, match="variables"):
            compare_evidence(base=a, alt=b)

    def test_permuted_variable_names_are_accepted(self):
        a = _evidence()
        b = _evidence(var_names=["y2", "y1"])
        assert compare_evidence(base=a, alt=b).reference == "base"

    def test_mismatched_sample_size_is_rejected(self):
        a = _evidence()
        b = _evidence(n_obs=19)
        with pytest.raises(ValueError, match="observations"):
            compare_evidence(base=a, alt=b)

    def test_mismatched_window_is_rejected(self):
        a = _evidence()
        b = _evidence(sample_start=pd.Timestamp("2018-03-01"), sample_end=pd.Timestamp("2019-10-01"))
        with pytest.raises(ValueError, match="window"):
            compare_evidence(base=a, alt=b)

    def test_mismatched_digest_is_rejected(self):
        a = _evidence()
        b = _evidence("-other")
        with pytest.raises(ValueError, match="digest"):
            compare_evidence(base=a, alt=b)

    def test_fit_without_evidence_is_rejected(self, synthetic_idata_2v, var_data_2v):
        from impulso.volatility import Constant

        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        with pytest.raises(ValueError, match="ConjugateVAR"):
            compare_evidence(base=_evidence(), nuts=fitted)

    def test_non_fit_argument_is_rejected(self):
        with pytest.raises(TypeError, match="ModelEvidence"):
            compare_evidence(base=_evidence(), oops="not a fit")

    def test_non_finite_evidence_is_rejected_at_construction(self):
        with pytest.raises(ValidationError, match="marginal likelihood evaluated to"):
            _evidence(log_marginal_likelihood=-np.inf)

    def test_comparison_requires_a_reference_it_holds(self):
        with pytest.raises(ValidationError, match="not one of the compared models"):
            EvidenceComparison(evidences={"a": _evidence(), "b": _evidence()}, reference="c")


# ------------------------------------------------------------------- D: end-to-end fast


class TestEndToEnd:
    """Real fits, small enough to stay off the slow marker."""

    def test_lag_order_comparison_on_an_aligned_window(self):
        data = _var_data(seed=21, t_full=28)
        trimmed = VARData(endog=data.endog[1:], endog_names=data.endog_names, index=data.index[1:])

        fit_p1 = ConjugateVAR(lags=1, prior=NIWPrior(select=True), draws=2, tune=0, seed=0).fit(trimmed)
        fit_p2 = ConjugateVAR(lags=2, prior=NIWPrior(select=True), draws=2, tune=0, seed=0).fit(data)

        frame = compare_evidence(one_lag=fit_p1, two_lags=fit_p2).to_dataframe()
        assert list(frame["n_lags"]) == [1, 2]
        assert frame.loc["one_lag", "log_bayes_factor"] == 0.0
        assert frame["posterior_probability"].sum() == pytest.approx(1.0, rel=1e-12)

    def test_unaligned_lag_orders_are_rejected_with_the_trimming_hint(self):
        data = _var_data(seed=21, t_full=28)
        fit_p1 = ConjugateVAR(lags=1, prior=NIWPrior(), draws=2, tune=0, seed=0).fit(data)
        fit_p2 = ConjugateVAR(lags=2, prior=NIWPrior(), draws=2, tune=0, seed=0).fit(data)

        with pytest.raises(ValueError, match="trimmed by the largest lag order"):
            compare_evidence(one_lag=fit_p1, two_lags=fit_p2)

    def test_break_versus_no_break_on_the_same_sample(self):
        data = _var_data(seed=23, t_full=26)
        homoscedastic = ConjugateVAR(lags=1, prior=NIWPrior(), draws=2, tune=0, seed=0).fit(data)
        with_break = ConjugateVAR(
            lags=1,
            prior=NIWPrior(),
            volatility=PandemicBreak(start=8),
            draws=2,
            tune=0,
            seed=0,
        ).fit(data)

        frame = compare_evidence(homoscedastic=homoscedastic, with_break=with_break).to_dataframe()
        assert list(frame["volatility"]) == [None, "pandemic_break"]
        assert np.isfinite(frame["log_bayes_factor"]).all()
