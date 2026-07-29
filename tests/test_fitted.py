"""Tests for FittedVAR."""

import numpy as np
import pandas as pd
import pytest

from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.samplers import NUTSSampler
from impulso.spec import VAR

# var_data_2v comes from conftest.py


@pytest.fixture
def var_data(var_data_2v):
    return var_data_2v


class TestFittedVAR:
    @pytest.mark.slow
    def test_fit_returns_fitted_var(self, var_data):
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        result = spec.fit(var_data, sampler=sampler)
        assert isinstance(result, FittedVAR)

    @pytest.mark.slow
    def test_fitted_properties(self, var_data):
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        result = spec.fit(var_data, sampler=sampler)
        assert result.n_lags == 1
        assert result.has_exog is False
        assert result.idata is not None

    @pytest.mark.slow
    def test_fit_with_auto_lags(self, var_data):
        spec = VAR(lags="bic", max_lags=4, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        result = spec.fit(var_data, sampler=sampler)
        assert result.n_lags >= 1

    @pytest.mark.slow
    def test_repr_is_compact(self, var_data):
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        result = spec.fit(var_data, sampler=sampler)
        r = repr(result)
        assert "FittedVAR" in r
        assert "n_lags=1" in r


class TestForecasting:
    @pytest.mark.slow
    def test_forecast_returns_forecast_result(self, var_data):
        from impulso.results import ForecastResult

        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        fitted = spec.fit(var_data, sampler=sampler)
        fcast = fitted.forecast(steps=4)
        assert isinstance(fcast, ForecastResult)
        assert fcast.steps == 4

    @pytest.mark.slow
    def test_forecast_median_shape(self, var_data):
        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        fitted = spec.fit(var_data, sampler=sampler)
        fcast = fitted.forecast(steps=8)
        med = fcast.median()
        assert med.shape == (8, 2)

    @pytest.mark.slow
    def test_forecast_hdi_returns_hdi_result(self, var_data):
        from impulso.results import HDIResult

        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        fitted = spec.fit(var_data, sampler=sampler)
        fcast = fitted.forecast(steps=4)
        hdi = fcast.hdi(prob=0.89)
        assert isinstance(hdi, HDIResult)

    @pytest.mark.slow
    def test_forecast_exog_required_error(self):
        """If model has exog, forecast without exog_future raises."""
        rng = np.random.default_rng(42)
        T, n = 200, 2
        y = np.zeros((T, n))
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
        exog = rng.standard_normal((T, 1))
        index = pd.date_range("2000-01-01", periods=T, freq="QS")
        data = VARData(endog=y, endog_names=["y1", "y2"], exog=exog, exog_names=["x1"], index=index)

        spec = VAR(lags=1, prior="minnesota")
        sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
        fitted = spec.fit(data, sampler=sampler)
        with pytest.raises(ValueError, match="exog_future"):
            fitted.forecast(steps=4)


class TestFittedVARFast:
    """Fast tests using synthetic InferenceData (no MCMC)."""

    def test_properties_from_synthetic(self, synthetic_idata_2v, var_data_2v):
        from impulso.volatility import Constant

        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        assert fitted.n_lags == 1
        assert fitted.has_exog is False
        assert fitted.coefficients.shape == (2, 50, 2, 2)
        assert fitted.intercepts.shape == (2, 50, 2)
        assert fitted.sigma().shape == (2, 50, 2, 2)

    def test_repr_from_synthetic(self, synthetic_idata_2v, var_data_2v):
        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )
        r = repr(fitted)
        assert "FittedVAR" in r
        assert "n_lags=1" in r

    def test_forecast_shape(self, synthetic_idata_2v, var_data_2v):
        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )
        result = fitted.forecast(steps=4, include_shock_uncertainty=False)
        med = result.median()
        assert med.shape == (4, 2)

    def test_forecast_hdi(self, synthetic_idata_2v, var_data_2v):
        from impulso.results import HDIResult

        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
        )
        result = fitted.forecast(steps=4, include_shock_uncertainty=False)
        hdi = result.hdi(prob=0.89)
        assert isinstance(hdi, HDIResult)
        assert hdi.lower.shape == (4, 2)
        assert hdi.upper.shape == (4, 2)


class TestFittedVARVolatility:
    def test_fitted_var_carries_volatility(self, var_data_2v):
        """FittedVAR must expose the volatility process used at fit time."""
        from impulso.protocols import VolatilityProcess
        from impulso.samplers import NUTSSampler
        from impulso.spec import VAR
        from impulso.volatility import Constant

        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)

        assert isinstance(fitted.volatility, VolatilityProcess)
        assert isinstance(fitted.volatility, Constant)
        assert fitted.volatility.name == "constant"

    def test_fitted_var_volatility_round_trips_explicit(self, var_data_2v):
        """A custom Constant() instance passed to VAR is preserved on FittedVAR."""
        from impulso.samplers import NUTSSampler
        from impulso.spec import VAR
        from impulso.volatility import Constant

        custom = Constant(sigma_sd_beta=3.0)
        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1, volatility=custom).fit(var_data_2v, sampler=sampler)

        assert fitted.volatility is custom
        assert fitted.volatility.sigma_sd_beta == 3.0


class TestSetIdentificationStrategyRoutesThroughSeam:
    @pytest.mark.slow
    def test_shock_matrix_triggers_cholesky_at(self, var_data_2v):
        """shock_matrix() queries volatility.cholesky_at lazily.
        set_identification_strategy no longer calls it eagerly."""
        from unittest.mock import MagicMock

        from impulso.identification import Cholesky
        from impulso.samplers import NUTSSampler
        from impulso.spec import VAR

        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        spy = MagicMock(wraps=identified.volatility)
        object.__setattr__(identified, "volatility", spy)

        sm = identified.shock_matrix()
        spy.cholesky_at.assert_called_once()

        # structural_shock_matrix no longer stored in the posterior.
        assert "structural_shock_matrix" not in identified.idata.posterior
        assert sm.dims == ("chain", "draw", "response", "shock")
        assert list(sm.coords["response"].values) == fitted.var_names
        assert list(sm.coords["shock"].values) == fitted.var_names


class TestFittedVarSigmaDispatch:
    """`FittedVAR.sigma()` dispatches by volatility adapter type."""

    @pytest.mark.slow
    def test_constant_sigma_unchanged(self, var_data_2v):
        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)
        sigma = fitted.sigma()
        # For Constant: (chains, draws, n_vars, n_vars).
        assert sigma.shape == (1, 20, 2, 2)

    @pytest.mark.slow
    def test_sv_sigma_returns_per_t(self, var_data_2v):
        from impulso.sv.spec import StochasticVolatility

        sampler = NUTSSampler(cores=1, chains=1, draws=20, tune=20, random_seed=0, nuts_sampler="pymc")
        fitted = VAR(lags=1, volatility=StochasticVolatility()).fit(var_data_2v, sampler=sampler)
        sigma_path = fitted.sigma()
        # For SV: (chains, draws, T, n_vars, n_vars).
        T = var_data_2v.endog.shape[0] - 1  # n_lags=1
        assert sigma_path.shape == (1, 20, T, 2, 2)


class TestExogCoefficientRecovery:
    """Regression for #192: B_exog must survive a small-scale regressor.

    The old prior was `Normal(0, 1)` in coefficient space regardless of the
    data. On this DGP the true coefficient is 50 (the regressor has sd 0.01,
    so its contribution has sd 0.5 against a shock sd of 0.1 — a strong,
    easily-identified signal). Under the old prior the posterior mean came
    back at 3.9 with a 94% HDI of [2.2, 5.7]: the truth excluded by an order
    of magnitude, silently. The scale-adaptive prior recovers it.

    The reference is the OLS estimate rather than the literal 50: with a
    near-flat prior the posterior should agree with the likelihood, and the
    likelihood's own answer on this finite sample is 49.04. Asserting "50 is
    inside the HDI" would get *harder* to satisfy as the sample grows, which
    is the wrong direction for a regression test.
    """

    @staticmethod
    def _tiny_regressor_dgp():
        T, n = 200, 2
        rng = np.random.default_rng(20250729)
        z = rng.normal(0.0, 0.01, T)
        A = np.array([[0.5, 0.1], [0.0, 0.5]])
        beta = np.array([50.0, 0.0])
        y = np.zeros((T, n))
        for t in range(1, T):
            y[t] = A @ y[t - 1] + beta * z[t] + rng.standard_normal(n) * 0.1
        index = pd.date_range("2000-01-01", periods=T, freq="QS")
        return VARData(
            endog=y,
            endog_names=["y1", "y2"],
            exog=z[:, None],
            exog_names=["z"],
            index=index,
        )

    @staticmethod
    def _ols_exog_coefficients(data):
        """Equation-by-equation OLS on [const, y_{t-1}, z_t] — the flat-prior answer."""
        y, z = data.endog, data.exog[:, 0]
        rhs = np.column_stack([np.ones(len(y) - 1), y[:-1], z[1:]])
        beta, *_ = np.linalg.lstsq(rhs, y[1:], rcond=None)
        return beta[-1]

    @pytest.mark.slow
    def test_recovers_a_large_coefficient_on_a_tiny_regressor(self):
        import arviz as az

        data = self._tiny_regressor_dgp()
        ols = self._ols_exog_coefficients(data)
        assert 45.0 < ols[0] < 55.0, f"fixture drifted: OLS says {ols[0]}, expected ~50"

        sampler = NUTSSampler(draws=200, tune=200, chains=1, cores=1, random_seed=1234, progressbar=False)
        fitted = VAR(lags=1).fit(data, sampler=sampler)

        b = fitted.idata.posterior["B_exog"].sel(var="y1", exog="z")
        mean = float(b.mean())
        hdi = az.hdi(b, hdi_prob=0.94)["B_exog"].values

        # The near-flat prior must let the likelihood speak.
        assert hdi[0] <= ols[0] <= hdi[1], f"OLS estimate {ols[0]} outside 94% HDI {hdi}"
        assert abs(mean - 50.0) < 5.0, f"posterior mean {mean} far from the true coefficient 50"
        # The old prior's entire posterior lived in [2.2, 5.7]; nothing near it survives.
        assert hdi[0] > 25.0, f"posterior still crushed towards zero: 94% HDI {hdi}"

        # The unaffected equation is still centred on zero.
        b2 = fitted.idata.posterior["B_exog"].sel(var="y2", exog="z")
        assert abs(float(b2.mean())) < 5.0


class TestErrorDistributionField:
    """The `error_dist` field on FittedVAR (issue #152)."""

    @pytest.fixture
    def gaussian_fitted(self, synthetic_idata_2v, var_data_2v):
        from impulso.volatility import Constant

        return FittedVAR(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )

    @pytest.fixture
    def student_t_fitted(self, synthetic_idata_2v_t, var_data_2v):
        from impulso.observation import StudentT
        from impulso.volatility import Constant

        return FittedVAR(
            idata=synthetic_idata_2v_t,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            error_dist=StudentT(nu=5.0),
        )

    def test_default_is_gaussian(self, gaussian_fitted):
        from impulso.observation import Gaussian

        assert isinstance(gaussian_fitted.error_dist, Gaussian)

    def test_default_is_gaussian_under_model_construct(self, synthetic_idata_2v, var_data_2v):
        from impulso.observation import Gaussian
        from impulso.volatility import Constant

        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        assert isinstance(fitted.error_dist, Gaussian)

    def test_identified_var_inherits_the_adapter(self, student_t_fitted):
        from impulso.identification import Cholesky

        identified = student_t_fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        assert identified.error_dist is student_t_fitted.error_dist


class TestInnovationCovariance:
    """sigma() returns the scale matrix; innovation_covariance() the covariance."""

    @pytest.fixture
    def gaussian_fitted(self, synthetic_idata_2v, var_data_2v):
        from impulso.volatility import Constant

        return FittedVAR(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )

    @pytest.fixture
    def student_t_fitted(self, synthetic_idata_2v_t, var_data_2v):
        from impulso.observation import StudentT
        from impulso.volatility import Constant

        return FittedVAR(
            idata=synthetic_idata_2v_t,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            error_dist=StudentT(nu=5.0),
        )

    def test_gaussian_covariance_equals_sigma(self, gaussian_fitted):
        np.testing.assert_array_equal(gaussian_fitted.innovation_covariance(), gaussian_fitted.sigma())

    def test_student_t_sigma_is_unchanged_by_the_error_law(self, gaussian_fitted, student_t_fitted):
        """The scale matrix is the volatility process's business, not the t's."""
        np.testing.assert_array_equal(student_t_fitted.sigma(), gaussian_fitted.sigma())

    def test_student_t_covariance_is_nu_over_nu_minus_two_times_sigma(self, student_t_fitted):
        expected = (5.0 / 3.0) * student_t_fitted.sigma()
        np.testing.assert_allclose(student_t_fitted.innovation_covariance(), expected)

    def test_shapes_match_sigma(self, student_t_fitted):
        assert student_t_fitted.innovation_covariance().shape == student_t_fitted.sigma().shape

    def test_per_draw_nu_broadcasts(self, synthetic_idata_2v, var_data_2v):
        import arviz as az
        import xarray as xr

        from impulso.observation import StudentT
        from impulso.volatility import Constant

        posterior = synthetic_idata_2v.posterior.copy()
        n_chains, n_draws = posterior.sizes["chain"], posterior.sizes["draw"]
        nu = np.linspace(3.0, 30.0, n_chains * n_draws).reshape(n_chains, n_draws)
        posterior["nu"] = xr.DataArray(nu, dims=["chain", "draw"])
        fitted = FittedVAR(
            idata=az.InferenceData(posterior=posterior),
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            error_dist=StudentT(),
        )
        expected = (nu / (nu - 2.0))[:, :, None, None] * fitted.sigma()
        np.testing.assert_allclose(fitted.innovation_covariance(), expected)
        # Heavier tails inflate more: the covariance ratio is decreasing in nu.
        ratio = fitted.innovation_covariance()[0, 0, 0, 0] / fitted.sigma()[0, 0, 0, 0]
        ratio_last = fitted.innovation_covariance()[-1, -1, 0, 0] / fitted.sigma()[-1, -1, 0, 0]
        assert ratio > ratio_last

    def test_time_varying_sigma_broadcasts_over_the_time_axis(self):
        """The (C, D, T, n, n) reshape branch, against hand-computed numbers.

        `VAR` rejects `volatility="sv"` with `error_dist="student_t"` at spec
        level (ADR-0007), so the 5-dim branch is only reachable from a
        hand-built `FittedVAR` — but `innovation_covariance` documents and
        handles the shape, so it is pinned here (issue #175).

        The posterior is rigged so the answer is readable by eye: `R_chol` is
        the identity and `h = log(v)`, which makes `L_t = diag(exp(h_t / 2))`
        and hence `Sigma_t = diag(v_t)` exactly. The two draws share the same
        volatility path and differ only in `nu`, so any mis-broadcast of the
        `(C, D)` inflation onto the `(C, D, T, n, n)` scale matrix shows up as
        a wrong number rather than a shape error.
        """
        import arviz as az
        import xarray as xr

        from impulso.observation import StudentT
        from impulso.sv.spec import StochasticVolatility

        n_chains, n_draws, T, n_vars = 1, 2, 3, 2
        variances = np.array([[1.0, 2.0], [4.0, 8.0], [9.0, 18.0]])  # (T, n_vars)
        h = np.broadcast_to(np.log(variances), (n_chains, n_draws, T, n_vars)).copy()
        R_chol = np.broadcast_to(np.eye(n_vars), (n_chains, n_draws, n_vars, n_vars)).copy()
        nu = np.array([[4.0, 6.0]])  # inflations nu/(nu-2) = 2.0 and 1.5

        posterior = xr.Dataset({
            "h": (("chain", "draw", "time", "variable"), h),
            "R_chol": (("chain", "draw", "i", "j"), R_chol),
            "nu": (("chain", "draw"), nu),
        })
        data = VARData(
            endog=np.zeros((T + 1, n_vars)),
            endog_names=["y1", "y2"],
            index=pd.date_range("2000-01-01", periods=T + 1, freq="MS"),
        )
        fitted = FittedVAR(
            idata=az.InferenceData(posterior=posterior),
            n_lags=1,  # T = endog rows - n_lags = 3, matching h's time axis
            data=data,
            var_names=["y1", "y2"],
            volatility=StochasticVolatility(),
            error_dist=StudentT(),  # nu comes from the posterior, not the field
        )

        sigma = fitted.sigma()
        assert sigma.shape == (n_chains, n_draws, T, n_vars, n_vars)
        np.testing.assert_allclose(
            np.diagonal(sigma, axis1=-2, axis2=-1),
            np.broadcast_to(variances, (n_chains, n_draws, T, n_vars)),
        )

        # nu = 4 doubles Sigma_t; nu = 6 multiplies it by 3/2. Same path, both draws.
        expected = np.array([
            [
                [[[2.0, 0.0], [0.0, 4.0]], [[8.0, 0.0], [0.0, 16.0]], [[18.0, 0.0], [0.0, 36.0]]],
                [[[1.5, 0.0], [0.0, 3.0]], [[6.0, 0.0], [0.0, 12.0]], [[13.5, 0.0], [0.0, 27.0]]],
            ]
        ])
        actual = fitted.innovation_covariance()
        assert actual.shape == sigma.shape
        np.testing.assert_allclose(actual, expected)

        # The inflation is per draw, so the ratio is flat along the T axis.
        ratios = np.diagonal(actual, axis1=-2, axis2=-1) / np.diagonal(sigma, axis1=-2, axis2=-1)
        np.testing.assert_allclose(ratios[0, 0], 2.0)
        np.testing.assert_allclose(ratios[0, 1], 1.5)


class TestForecastExogValidation:
    """`forecast` validates the future exogenous block before propagating it.

    Users generate this block programmatically now
    (`DeterministicDesign.exog_future`), so a mis-shaped one must fail with a
    named contract rather than an opaque einsum error deep in the loop.
    """

    @pytest.fixture
    def fitted_with_exog(self, synthetic_idata_2v, var_data_2v):
        import xarray as xr

        rng = np.random.default_rng(3)
        exog = rng.standard_normal((var_data_2v.endog.shape[0], 2))
        data = VARData(
            endog=var_data_2v.endog,
            endog_names=list(var_data_2v.endog_names),
            exog=exog,
            exog_names=["x1", "x2"],
            index=var_data_2v.index,
        )
        idata = synthetic_idata_2v.copy()
        idata.posterior["B_exog"] = xr.DataArray(
            rng.standard_normal((2, 50, 2, 2)),
            dims=["chain", "draw", "var", "exog"],
            coords={"exog": ["x1", "x2"]},
        )
        return FittedVAR.model_construct(
            idata=idata,
            n_lags=1,
            data=data,
            var_names=["y1", "y2"],
        )

    def test_correct_shape_forecasts(self, fitted_with_exog):
        result = fitted_with_exog.forecast(
            steps=4,
            exog_future=np.zeros((4, 2)),
            include_shock_uncertainty=False,
        )
        assert result.median().shape == (4, 2)

    def test_wrong_column_count_raises(self, fitted_with_exog):
        with pytest.raises(ValueError, match=r"exog_future must have shape \(4, 2\), got \(4, 1\)"):
            fitted_with_exog.forecast(steps=4, exog_future=np.zeros((4, 1)))

    def test_wrong_step_count_raises(self, fitted_with_exog):
        with pytest.raises(ValueError, match=r"exog_future must have shape \(4, 2\), got \(6, 2\)"):
            fitted_with_exog.forecast(steps=4, exog_future=np.zeros((6, 2)))

    def test_posterior_without_b_exog_raises(self, synthetic_idata_2v, var_data_2v):
        rng = np.random.default_rng(4)
        data = VARData(
            endog=var_data_2v.endog,
            endog_names=list(var_data_2v.endog_names),
            exog=rng.standard_normal((var_data_2v.endog.shape[0], 1)),
            exog_names=["x1"],
            index=var_data_2v.index,
        )
        fitted = FittedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=data,
            var_names=["y1", "y2"],
        )
        with pytest.raises(ValueError, match="never consumed"):
            fitted.forecast(steps=3, exog_future=np.zeros((3, 1)))
