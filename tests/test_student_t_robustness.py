"""Robustness evidence for Student-t observation errors (issue #152).

The fast test is the primary evidence and needs no MCMC: it asserts the
*mechanism* directly on the compiled PyMC gradient. The t score with
respect to the location is

    d log p / d mu = w * Omega^-1 (y - mu),   w = (nu + n) / (nu + q),

with `q` the squared Mahalanobis distance. As an observation moves into the
tail, `q` grows like ||y - mu||^2 while the linear factor grows like
||y - mu||, so the score is bounded and eventually *redescending*. The
Gaussian score is `Omega^-1 (y - mu)`: unbounded and linear. That
difference is the whole robustness story, and it is deterministic.

The slow tests check the issue's literal acceptance criterion end to end:
on contaminated data, a t-errors VAR recovers the true coefficients better
than a Gaussian one, and an inferred `nu` lands in the heavy-tailed region.
"""

import numpy as np
import pytest

from impulso.data import VARData
from impulso.observation import Gaussian, StudentT


def _location_score(error_dist, y, m_true, scale):
    """Gradient of the log-likelihood wrt a free location, at m = m_true.

    Builds a minimal PyMC model — free location `m`, fixed lower-triangular
    scale factor — and compiles d logp / d m. No sampling involved.
    """
    import pymc as pm
    import pytensor.tensor as pt

    n = y.shape[-1]
    with pm.Model() as model:
        m = pm.Flat("m", shape=n)
        chol = pt.as_tensor(np.linalg.cholesky(scale))
        mu = pt.tile(m, (y.shape[0], 1))
        error_dist.build_likelihood("obs", mu=mu, chol=chol, observed=y)
    grad = model.compile_dlogp([model["m"]])({"m": np.asarray(m_true, dtype=float)})
    return float(np.linalg.norm(grad))


class TestScoreIsBoundedUnderAnOutlier:
    """The mechanism, asserted deterministically (no MCMC, sub-second)."""

    SCALE = np.array([[1.0, 0.2], [0.2, 1.0]])
    M_TRUE = np.array([0.0, 0.0])

    def _contaminated(self, k):
        """One observation displaced `k` scale units along the first axis."""
        return (self.M_TRUE + np.array([k, 0.0]))[None, :]

    @pytest.mark.parametrize("error_dist", [Gaussian(), StudentT(nu=4.0)])
    def test_scores_are_finite(self, error_dist):
        for k in (1.0, 5.0, 50.0):
            assert np.isfinite(_location_score(error_dist, self._contaminated(k), self.M_TRUE, self.SCALE))

    def test_gaussian_score_grows_without_bound(self):
        scores = [_location_score(Gaussian(), self._contaminated(k), self.M_TRUE, self.SCALE) for k in (1.0, 5.0, 50.0)]
        assert scores[0] < scores[1] < scores[2]
        # Linear in the displacement: a 10x further outlier pulls 10x harder.
        assert scores[2] / scores[1] == pytest.approx(10.0, rel=0.05)

    def test_student_t_score_redescends(self):
        scores = [
            _location_score(StudentT(nu=4.0), self._contaminated(k), self.M_TRUE, self.SCALE) for k in (1.0, 5.0, 50.0)
        ]
        # Bounded *and* redescending: the far outlier pulls less than the
        # near one, which is exactly the automatic downweighting.
        assert scores[2] < scores[1]

    def test_student_t_influence_is_an_order_of_magnitude_smaller(self):
        gauss = _location_score(Gaussian(), self._contaminated(50.0), self.M_TRUE, self.SCALE)
        student = _location_score(StudentT(nu=4.0), self._contaminated(50.0), self.M_TRUE, self.SCALE)
        assert student < gauss / 10

    def test_the_two_agree_on_a_typical_observation(self):
        """Robustness is about the tail; the bulk should look similar."""
        gauss = _location_score(Gaussian(), self._contaminated(0.5), self.M_TRUE, self.SCALE)
        student = _location_score(StudentT(nu=4.0), self._contaminated(0.5), self.M_TRUE, self.SCALE)
        assert 0.5 < student / gauss < 1.5


def _contaminated_var_data():
    """VAR(1), n=2, T=400, with three large additive outliers in y1.

    Contamination is deliberately gross — 15 innovation standard deviations,
    alternating in sign so it cannot be soaked up by the intercept — and
    placed at three interior dates so a Gaussian fit has to bend the
    coefficients to accommodate it.
    """
    rng = np.random.default_rng(0)
    T, n = 400, 2
    A1 = np.array([[0.5, 0.1], [-0.2, 0.3]])
    innovation_sd = 0.5
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = A1 @ y[t - 1] + rng.standard_normal(n) * innovation_sd
    for i, t in enumerate((100, 200, 300)):
        y[t, 0] += 15.0 * innovation_sd * (1.0 if i % 2 == 0 else -1.0)

    import pandas as pd

    index = pd.date_range("1970-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2"], index=index), A1


@pytest.fixture(scope="module")
def contaminated_fits():
    """Fit the same contaminated sample under Gaussian and Student-t errors.

    Module-scoped: two NUTS runs are the expensive part of this file, and
    both slow tests read from them.

    The prior is loosened to `tightness=0.5`. At the default 0.1 the
    Minnesota prior's own shrinkage bias dominates the comparison — it pulls
    the estimate toward a random walk hard enough that the *clean*-data
    recovery error is ~0.30, larger than anything the contamination does —
    and the test would be measuring the prior rather than the error law. At
    0.5 the clean-data error is ~0.085, so what is left is attributable to
    the outliers.
    """
    from impulso.priors import MinnesotaPrior
    from impulso.samplers import NUTSSampler
    from impulso.spec import VAR

    data, A1 = _contaminated_var_data()

    def _fit(error_dist):
        sampler = NUTSSampler(draws=400, tune=400, chains=2, cores=1, random_seed=7, progressbar=False)
        spec = VAR(lags=1, prior=MinnesotaPrior(tightness=0.5), error_dist=error_dist)
        return spec.fit(data, sampler=sampler)

    return {
        "A1": A1,
        "data": data,
        "gaussian": _fit("gaussian"),
        "student_t": _fit(StudentT()),
    }


def _coefficient_error(fitted, A1):
    """Frobenius distance of the posterior-median B from the true A1."""
    B_median = np.median(fitted.idata.posterior["B"].values, axis=(0, 1))
    return float(np.linalg.norm(B_median - A1, ord="fro"))


@pytest.mark.slow
class TestRecoveryUnderContamination:
    def test_student_t_recovers_coefficients_better_than_gaussian(self, contaminated_fits):
        A1 = contaminated_fits["A1"]
        err_gaussian = _coefficient_error(contaminated_fits["gaussian"], A1)
        err_student = _coefficient_error(contaminated_fits["student_t"], A1)
        assert err_student < err_gaussian, (err_student, err_gaussian)
        # Absolute anchor, so "both are equally terrible" cannot pass. The
        # clean-sample floor for this DGP and prior is ~0.085.
        assert err_student < 0.15, err_student
        # Measured separation is a factor of ~3; assert a conservative half.
        assert err_student < 0.5 * err_gaussian, (err_student, err_gaussian)

    def test_student_t_scale_is_not_inflated_by_the_outliers(self, contaminated_fits):
        """Companion to the coefficient test: the t absorbs outliers in the
        tail rather than in the scale matrix."""
        sigma_gaussian = np.median(contaminated_fits["gaussian"].sigma(), axis=(0, 1))
        sigma_student = np.median(contaminated_fits["student_t"].sigma(), axis=(0, 1))
        assert np.all(np.diag(sigma_student) < np.diag(sigma_gaussian))


@pytest.mark.slow
class TestInferredDegreesOfFreedom:
    def test_nu_is_learned_bounded_and_heavy_tailed(self, contaminated_fits):
        posterior = contaminated_fits["student_t"].idata.posterior
        assert "nu" in posterior
        nu = posterior["nu"].values
        assert nu.shape == posterior["intercept"].values.shape[:2]
        # The shifted Gamma prior guarantees the support; check it holds.
        assert nu.min() > 2.0
        # Learned from the data, not echoing the prior (prior mean = 22).
        assert np.median(nu) < 12.0, float(np.median(nu))

    def test_innovation_covariance_exceeds_the_scale_matrix(self, contaminated_fits):
        fitted = contaminated_fits["student_t"]
        scale = np.median(fitted.sigma(), axis=(0, 1))
        covariance = np.median(fitted.innovation_covariance(), axis=(0, 1))
        assert np.all(np.diag(covariance) > np.diag(scale))
