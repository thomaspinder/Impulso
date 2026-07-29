"""Tests for IdentifiedVAR."""

import warnings

import arviz as az
import numpy as np
import pytest
import xarray as xr

from impulso.fitted import FittedVAR
from impulso.identification import Cholesky, SignRestriction
from impulso.identified import IdentifiedVAR
from impulso.results import FEVDResult, HistoricalDecompositionResult, IRFResult
from impulso.samplers import NUTSSampler
from impulso.spec import VAR


@pytest.fixture
def fitted_var(var_data_2v):
    """Fit a small VAR for testing."""
    spec = VAR(lags=1, prior="minnesota")
    sampler = NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42)
    return spec.fit(var_data_2v, sampler=sampler)


@pytest.fixture
def identified_cholesky(synthetic_idata_2v, var_data_2v):
    """IdentifiedVAR via public FittedVAR -> set_identification_strategy flow."""
    from impulso.volatility import Constant

    fitted = FittedVAR(
        idata=synthetic_idata_2v,
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )
    return fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))


class TestIdentifiedVAR:
    @pytest.mark.slow
    def test_set_identification_returns_identified(self, fitted_var):
        identified = fitted_var.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        assert isinstance(identified, IdentifiedVAR)

    @pytest.mark.slow
    def test_impulse_response(self, fitted_var):
        identified = fitted_var.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        irf = identified.impulse_response(horizon=10)
        assert isinstance(irf, IRFResult)
        assert irf.horizon == 10

    @pytest.mark.slow
    def test_fevd(self, fitted_var):
        identified = fitted_var.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        fevd = identified.fevd(horizon=10)
        assert isinstance(fevd, FEVDResult)

    @pytest.mark.slow
    def test_historical_decomposition(self, fitted_var):
        identified = fitted_var.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        hd = identified.historical_decomposition()
        assert isinstance(hd, HistoricalDecompositionResult)


class TestIdentifiedVARFast:
    """Fast tests using synthetic InferenceData (no MCMC)."""

    def test_impulse_response_shape(self, identified_cholesky):
        irf = identified_cholesky.impulse_response(horizon=10)
        assert isinstance(irf, IRFResult)
        assert irf.horizon == 10
        assert irf.median().shape == (11, 4)

    def test_fevd_shape(self, identified_cholesky):
        fevd = identified_cholesky.fevd(horizon=10)
        assert isinstance(fevd, FEVDResult)
        assert fevd.median().shape == (11, 4)

    def test_fevd_sums_to_one(self, identified_cholesky):
        """FEVD shares should sum to ~1 for each response at each horizon."""
        fevd = identified_cholesky.fevd(horizon=10)
        fevd_da = fevd.idata.posterior_predictive["fevd"]
        med = fevd_da.median(dim=("chain", "draw"))
        for resp in ["y1", "y2"]:
            sums = med.sel(response=resp).values.sum(axis=1)
            np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_historical_decomposition_shape(self, identified_cholesky):
        hd = identified_cholesky.historical_decomposition()
        assert isinstance(hd, HistoricalDecompositionResult)

    def test_irf_deterministic_values(self, identified_cholesky):
        """IRF at horizon 0 should equal the structural shock matrix."""
        irf = identified_cholesky.impulse_response(horizon=5)
        irf_draws = irf.idata.posterior_predictive["irf"].values
        P = identified_cholesky.shock_matrix().values
        np.testing.assert_allclose(irf_draws[:, :, 0, :, :], P, atol=1e-12)

    def test_irf_shock_coords_from_scheme(self, identified_cholesky):
        """IRF shock coordinates come from scheme.shock_coords()."""
        irf = identified_cholesky.impulse_response(horizon=5)
        irf_shocks = list(irf.idata.posterior_predictive["irf"].coords["shock"].values)
        assert irf_shocks == ["y1", "y2"]

    def test_repr(self, identified_cholesky):
        r = repr(identified_cholesky)
        assert "IdentifiedVAR" in r


class TestP2PosteriorEquivalence:
    """P2: shock_matrix output matches the mathematically expected value."""

    @pytest.mark.slow
    def test_cholesky_shock_matrix_matches_cholesky_of_sigma(self, var_data_2v):
        """Cholesky-identified shock_matrix equals np.linalg.cholesky(Sigma)."""
        sampler = NUTSSampler(cores=1, chains=2, draws=200, tune=200, random_seed=42, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        P = identified.shock_matrix().values
        assert P.shape == (2, 200, 2, 2)
        sigma = fitted.idata.posterior["Sigma"].values
        np.testing.assert_allclose(P, np.linalg.cholesky(sigma), rtol=1e-10)

    @pytest.mark.slow
    def test_sign_restriction_shock_matrix_runs(self, var_data_2v):
        """Smoke test: SignRestriction produces a valid shock_matrix."""
        sampler = NUTSSampler(cores=1, chains=2, draws=100, tune=100, random_seed=42, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)
        scheme = SignRestriction(
            restrictions={fitted.var_names[0]: {"shock_a": "+"}},
            n_rotations=50,
            restriction_horizon=2,
            random_seed=42,
        )
        identified = fitted.set_identification_strategy(scheme)

        P = identified.shock_matrix().values
        assert P.shape == (2, 100, 2, 2)
        assert not np.isnan(P).any()


def _fitted_from_synthetic(synthetic_idata_2v, var_data_2v):
    """Build a FittedVAR from synthetic InferenceData (no MCMC)."""
    from impulso.fitted import FittedVAR
    from impulso.volatility import Constant

    return FittedVAR.model_construct(
        idata=synthetic_idata_2v,
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )


class TestShockMatrixDiagnostics:
    """Scheme diagnostics surface on the shock-matrix attrs (the tutorials read them)."""

    def test_sign_restriction_acceptance_rate_attr(self, synthetic_idata_2v, var_data_2v):
        """Attr must be present even at 100% acceptance (regression: `rate < 1.0` guard dropped it)."""
        fitted = _fitted_from_synthetic(synthetic_idata_2v, var_data_2v)
        scheme = SignRestriction(
            restrictions={"y1": {"shock_a": "+"}},
            n_rotations=50,
            restriction_horizon=1,
            random_seed=7,
        )
        sm = fitted.set_identification_strategy(scheme).shock_matrix()
        rate = sm.attrs["sign_restriction_acceptance_rate"]
        assert isinstance(rate, float)
        assert 0.0 < rate <= 1.0


class TestIdentifiedVarCarriesVolatilityAndScheme:
    def test_carries_volatility_and_scheme(self, synthetic_idata_2v, var_data_2v):
        from impulso.protocols import IdentificationScheme, VolatilityProcess

        fitted = _fitted_from_synthetic(synthetic_idata_2v, var_data_2v)
        scheme = Cholesky(ordering=fitted.var_names)
        identified = fitted.set_identification_strategy(scheme)

        assert isinstance(identified.volatility, VolatilityProcess)
        assert isinstance(identified.scheme, IdentificationScheme)
        assert identified.scheme is scheme


class TestImpulseResponseAt:
    def test_at_ignored_for_constant(self, synthetic_idata_2v, var_data_2v):
        """For Constant volatility, at= must be a no-op."""
        fitted = _fitted_from_synthetic(synthetic_idata_2v, var_data_2v)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        irf_default = identified.impulse_response(horizon=5)
        irf_at_last = identified.impulse_response(horizon=5, at="last")
        irf_at_int = identified.impulse_response(horizon=5, at=10)
        irf_at_none = identified.impulse_response(horizon=5, at=None)

        np.testing.assert_array_equal(
            irf_default.idata.posterior_predictive["irf"].values,
            irf_at_last.idata.posterior_predictive["irf"].values,
        )
        np.testing.assert_array_equal(
            irf_default.idata.posterior_predictive["irf"].values,
            irf_at_int.idata.posterior_predictive["irf"].values,
        )
        np.testing.assert_array_equal(
            irf_default.idata.posterior_predictive["irf"].values,
            irf_at_none.idata.posterior_predictive["irf"].values,
        )

    def test_at_all_raises_for_constant_volatility(self, synthetic_idata_2v, var_data_2v):
        """``at='all'`` is meaningless under constant Σ — the per-t IRF would
        be identical at every t. Refuse with a ``ValueError`` pointing the
        caller at ``at=None`` / ``at='last'``.
        """
        fitted = _fitted_from_synthetic(synthetic_idata_2v, var_data_2v)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        with pytest.raises(ValueError, match=r"at=None.*at='last'"):
            identified.impulse_response(horizon=5, at="all")


class TestFEVDAt:
    def test_at_ignored_for_constant(self, synthetic_idata_2v, var_data_2v):
        fitted = _fitted_from_synthetic(synthetic_idata_2v, var_data_2v)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        fevd_default = identified.fevd(horizon=5)
        fevd_at_int = identified.fevd(horizon=5, at=3)
        np.testing.assert_array_equal(
            fevd_default.idata.posterior_predictive["fevd"].values,
            fevd_at_int.idata.posterior_predictive["fevd"].values,
        )

    def test_at_all_raises_for_constant_volatility(self, synthetic_idata_2v, var_data_2v):
        """``at='all'`` is meaningless under constant Σ — same reasoning as IRF."""
        fitted = _fitted_from_synthetic(synthetic_idata_2v, var_data_2v)
        identified = fitted.set_identification_strategy(Cholesky(ordering=fitted.var_names))

        with pytest.raises(ValueError, match=r"at=None.*at='last'"):
            identified.fevd(horizon=5, at="all")

    def test_median_raises_for_time_dim_fevd(self):
        """``FEVDResult.median()/hdi()/to_dataframe()`` must refuse FEVDs that
        carry a ``time`` dim (only reachable via time-varying volatility now
        that constant + ``at='all'`` raises at the call site)."""
        rng = np.random.default_rng(0)
        T_eff, horizon = 5, 4
        data = rng.uniform(size=(2, 10, T_eff, horizon + 1, 2, 2))
        fevd_da = xr.DataArray(
            data,
            dims=["chain", "draw", "time", "horizon", "response", "shock"],
            coords={
                "response": ["y1", "y2"],
                "shock": ["y1", "y2"],
                "horizon": np.arange(horizon + 1),
            },
            name="fevd",
        )
        idata = az.InferenceData(posterior_predictive=xr.Dataset({"fevd": fevd_da}))
        fevd_all = FEVDResult.model_construct(idata=idata, horizon=horizon, var_names=["y1", "y2"])

        with pytest.raises(NotImplementedError, match="time-varying FEVDs"):
            fevd_all.median()
        with pytest.raises(NotImplementedError, match="time-varying FEVDs"):
            fevd_all.hdi()
        with pytest.raises(NotImplementedError, match="time-varying FEVDs"):
            fevd_all.to_dataframe()


class TestHistoricalDecompositionAt:
    """Task 11: ``at=`` parameter on historical_decomposition.

    HD is intrinsically time-indexed, so ``at=None`` / ``at="all"`` both
    use the per-t identification path. For Constant volatility every
    L_t is identical, so all ``at=`` modes produce identical results.
    """

    def test_at_default_matches_at_all_for_constant(self, identified_cholesky):
        """For Constant volatility, ``at=None`` and ``at='all'`` are identical."""
        hd_default = identified_cholesky.historical_decomposition()
        hd_all = identified_cholesky.historical_decomposition(at="all")
        np.testing.assert_array_equal(
            hd_default.idata.posterior_predictive["hd"].values,
            hd_all.idata.posterior_predictive["hd"].values,
        )

    def test_at_int_matches_default_for_constant(self, identified_cholesky):
        """For Constant volatility, ``at=int`` and ``at='last'`` match the default."""
        hd_default = identified_cholesky.historical_decomposition()
        hd_at_int = identified_cholesky.historical_decomposition(at=5)
        hd_at_last = identified_cholesky.historical_decomposition(at="last")
        np.testing.assert_array_equal(
            hd_default.idata.posterior_predictive["hd"].values,
            hd_at_int.idata.posterior_predictive["hd"].values,
        )
        np.testing.assert_array_equal(
            hd_default.idata.posterior_predictive["hd"].values,
            hd_at_last.idata.posterior_predictive["hd"].values,
        )

    def test_at_default_matches_shock_matrix_reconstruction(self, identified_cholesky, synthetic_idata_2v, var_data_2v):
        """For Constant, HD equals the hand-propagated single-P decomposition."""
        hd = identified_cholesky.historical_decomposition()
        hd_vals = hd.idata.posterior_predictive["hd"].values

        # Recompute by hand: contemporaneous impact via shock_matrix, then
        # the lag-1 propagation recursion (A_1 = B for n_lags = 1).
        B = synthetic_idata_2v.posterior["B"].values
        P = identified_cholesky.shock_matrix().values
        intercept = synthetic_idata_2v.posterior["intercept"].values
        y = var_data_2v.endog
        n_lags = 1
        T = y.shape[0]
        x_lag = np.concatenate([y[n_lags - lag : T - lag] for lag in range(1, n_lags + 1)], axis=1)
        y_hat = intercept[:, :, np.newaxis, :] + np.einsum("cdij,tj->cdti", B, x_lag)
        resid = y[n_lags:][np.newaxis, np.newaxis, :, :] - y_hat
        P_inv = np.linalg.inv(P)
        s = np.einsum("cdij,cdtj->cdti", P_inv, resid)
        impact = P[:, :, np.newaxis, :, :] * s[:, :, :, np.newaxis, :]
        expected = np.zeros_like(impact)
        carry = np.zeros(impact.shape[:2] + impact.shape[3:])
        for t in range(impact.shape[2]):
            carry = impact[:, :, t] + np.einsum("cdij,cdjs->cdis", B, carry)
            expected[:, :, t] = carry
        np.testing.assert_allclose(hd_vals, expected, atol=1e-10)

    def test_at_preserves_time_dim(self, identified_cholesky, var_data_2v):
        """HD always carries a time dim; ``at=`` must not change its length."""
        hd = identified_cholesky.historical_decomposition(at="all")
        hd_da = hd.idata.posterior_predictive["hd"]
        assert "time" in hd_da.dims
        T_eff = var_data_2v.endog.shape[0] - 1
        assert hd_da.sizes["time"] == T_eff

    def test_at_int_warns_for_sv(self, var_data_2v, synthetic_idata_2v):
        """Under SV, ``at=int`` is a non-standard hypothetical and warns."""

        from impulso.volatility import Constant

        # Build a fake SV-flavoured IdentifiedVAR by overriding ``name`` via a
        # minimal stub that satisfies the VolatilityProcess Protocol and
        # delegates to Constant for the actual computation. This is the
        # cheapest way to exercise the warning branch without fitting an SV
        # model (sv/ adapters require their own MCMC fixtures).
        constant = Constant()

        class _FakeSV:
            name = "sv"
            is_time_varying = True

            def build_pymc_latent(self, n_vars, T):  # pragma: no cover
                raise NotImplementedError

            def cholesky_at(self, posterior, t):
                return constant.cholesky_at(posterior, t=t)

            def forecast_cholesky_path(self, posterior, steps, rng):  # pragma: no cover
                return constant.forecast_cholesky_path(posterior, steps=steps, rng=rng)

            def cholesky_path(self, posterior, T):
                return constant.cholesky_path(posterior, T=T)

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=_FakeSV(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        with pytest.warns(UserWarning, match="non-standard hypothetical"):
            identified.historical_decomposition(at=3)
        with pytest.warns(UserWarning, match="non-standard hypothetical"):
            identified.historical_decomposition(at="last")

    def test_at_default_does_not_warn_for_sv(self, var_data_2v, synthetic_idata_2v):
        """Default (per-t) HD is the standard SV decomposition — no warning."""
        import warnings as _warnings

        from impulso.volatility import Constant

        constant = Constant()

        class _FakeSV:
            name = "sv"
            is_time_varying = True

            def build_pymc_latent(self, n_vars, T):  # pragma: no cover
                raise NotImplementedError

            def cholesky_at(self, posterior, t):
                return constant.cholesky_at(posterior, t=t)

            def forecast_cholesky_path(self, posterior, steps, rng):  # pragma: no cover
                return constant.forecast_cholesky_path(posterior, steps=steps, rng=rng)

            def cholesky_path(self, posterior, T):
                return constant.cholesky_path(posterior, T=T)

        identified = IdentifiedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=_FakeSV(),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")  # any warning becomes an error
            identified.historical_decomposition()
            identified.historical_decomposition(at=None)
            identified.historical_decomposition(at="all")


class TestCholeskyOrderingLabels:
    """A non-identity Cholesky ordering must stay label-consistent end to end.

    `Cholesky.identify` returns rows in **data** order and columns in
    **ordering** order; `shock_matrix` labels them `var_names` / `shock_names`
    and `impulse_response` left-multiplies by MA coefficients built in data
    order. If `identify` returned rows in ordering order instead, the labels
    would be permuted and `Phi @ P` would mix coordinate systems. Regression
    for #184.
    """

    @staticmethod
    def _permuted_idata(synthetic_idata_2v, perm):
        """Relabel the 2-var VAR(1) posterior into `perm` variable order.

        Sigma -> Pi Sigma Pi', B -> Pi B Pi' (single lag), intercept -> Pi c.
        This is the *same* model written in a different variable order, so
        every label-indexed quantity must be invariant to it.
        """
        post = synthetic_idata_2v.posterior
        ix0, ix1 = np.ix_(perm, perm)

        sigma = post["Sigma"].values[:, :, ix0, ix1]
        B = post["B"].values[:, :, ix0, ix1]  # n_lags == 1, so B is (C, D, n, n)
        intercept = post["intercept"].values[:, :, perm]
        L = np.linalg.cholesky(sigma)

        names = [["y1", "y2"][i] for i in perm]
        return az.InferenceData(
            posterior=xr.Dataset({
                "B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"]),
                "intercept": xr.DataArray(intercept, dims=["chain", "draw", "var"]),
                "Sigma": xr.DataArray(
                    sigma,
                    dims=["chain", "draw", "var1", "var2"],
                    coords={"var1": names, "var2": names},
                ),
                "L": xr.DataArray(L, dims=["chain", "draw", "var1", "var2"]),
            })
        )

    @staticmethod
    def _identified(idata, var_data_2v, var_names, ordering):
        from impulso.volatility import Constant

        return IdentifiedVAR.model_construct(
            idata=idata,
            n_lags=1,
            data=var_data_2v,
            var_names=var_names,
            volatility=Constant(),
            scheme=Cholesky(ordering=ordering),
        )

    def test_shock_matrix_zero_lands_in_the_labelled_cell(self, synthetic_idata_2v, var_data_2v):
        """With ordering ["y2","y1"], y2 is most exogenous: y2 cannot respond to a y1 shock."""
        identified = self._identified(synthetic_idata_2v, var_data_2v, ["y1", "y2"], ["y2", "y1"])
        sm = identified.shock_matrix()

        assert list(sm.coords["response"].values) == ["y1", "y2"]
        assert list(sm.coords["shock"].values) == ["y2", "y1"]

        # The structural zero belongs to (response=y2, shock=y1) — exactly.
        assert np.abs(sm.sel(response="y2", shock="y1").values).max() == 0.0
        # ...and the transposed cell must be non-zero for every draw.
        assert np.abs(sm.sel(response="y1", shock="y2").values).min() > 0.0

    def test_irf_labels_invariant_to_relabelling(self, synthetic_idata_2v, var_data_2v):
        """IRFs indexed by (response, shock) labels are invariant to variable order."""
        perm = np.array([1, 0])
        sigma = synthetic_idata_2v.posterior["Sigma"].values
        # Guard: the two variables must be distinguishable, otherwise the
        # assertion below could pass through symmetry rather than correctness.
        assert np.abs(sigma[..., 0, 0] - sigma[..., 1, 1]).max() > 0.1

        # (A) Data already in ["y2","y1"] order, identity Cholesky ordering.
        a = self._identified(self._permuted_idata(synthetic_idata_2v, perm), var_data_2v, ["y2", "y1"], ["y2", "y1"])
        # (B) Data in ["y1","y2"] order, non-identity Cholesky ordering.
        b = self._identified(synthetic_idata_2v, var_data_2v, ["y1", "y2"], ["y2", "y1"])

        irf_a = a.impulse_response(horizon=6).idata.posterior_predictive["irf"]
        irf_b = b.impulse_response(horizon=6).idata.posterior_predictive["irf"]

        for response in ("y1", "y2"):
            for shock in ("y1", "y2"):
                np.testing.assert_allclose(
                    irf_a.sel(response=response, shock=shock).values,
                    irf_b.sel(response=response, shock=shock).values,
                    atol=1e-10,
                    err_msg=f"IRF(response={response}, shock={shock}) is order-dependent",
                )

    def test_fevd_shares_match_under_relabelling(self, synthetic_idata_2v, var_data_2v):
        """FEVD squares Theta, so it catches mislabelling that sign flips would hide."""
        perm = np.array([1, 0])
        a = self._identified(self._permuted_idata(synthetic_idata_2v, perm), var_data_2v, ["y2", "y1"], ["y2", "y1"])
        b = self._identified(synthetic_idata_2v, var_data_2v, ["y1", "y2"], ["y2", "y1"])

        fevd_a = a.fevd(horizon=6).idata.posterior_predictive["fevd"]
        fevd_b = b.fevd(horizon=6).idata.posterior_predictive["fevd"]

        for response in ("y1", "y2"):
            for shock in ("y1", "y2"):
                np.testing.assert_allclose(
                    fevd_a.sel(response=response, shock=shock).values,
                    fevd_b.sel(response=response, shock=shock).values,
                    atol=1e-10,
                    err_msg=f"FEVD(response={response}, shock={shock}) is order-dependent",
                )


class TestFEVDIsInvariantToTheErrorLaw:
    """FEVD is exactly invariant to the scale-vs-covariance convention (#152).

    Theta = Phi @ P, and P -> cP scales numerator and denominator by c^2.
    """

    @pytest.fixture
    def identified_cholesky_t(self, synthetic_idata_2v_t, var_data_2v):
        from impulso.observation import StudentT
        from impulso.volatility import Constant

        fitted = FittedVAR(
            idata=synthetic_idata_2v_t,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
            error_dist=StudentT(nu=5.0),
        )
        return fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))

    def test_fevd_identical_to_gaussian(self, identified_cholesky, identified_cholesky_t):
        gaussian = identified_cholesky.fevd(horizon=6)
        student = identified_cholesky_t.fevd(horizon=6)
        np.testing.assert_allclose(
            student.idata.posterior_predictive["fevd"].values,
            gaussian.idata.posterior_predictive["fevd"].values,
            rtol=0,
            atol=0,
        )

    def test_irf_identical_to_gaussian(self, identified_cholesky, identified_cholesky_t):
        """IRFs are in *scale* units under t, so the raw numbers coincide.

        The convention caveat is about interpretation (one scale unit is
        sqrt((nu-2)/nu) unconditional sd), not about a different computation.
        """
        gaussian = identified_cholesky.impulse_response(horizon=6)
        student = identified_cholesky_t.impulse_response(horizon=6)
        np.testing.assert_allclose(
            student.idata.posterior_predictive["irf"].values,
            gaussian.idata.posterior_predictive["irf"].values,
            rtol=0,
            atol=0,
        )

    def test_error_dist_defaults_to_gaussian(self, identified_cholesky):
        from impulso.observation import Gaussian

        assert isinstance(identified_cholesky.error_dist, Gaussian)


@pytest.fixture
def identified_long_run(permanent_transitory_2v, var_data_2v):
    """IdentifiedVAR over the exact-arithmetic long-run fixture."""
    from impulso.identification import LongRunRestriction
    from impulso.volatility import Constant

    fitted = FittedVAR(
        idata=permanent_transitory_2v["idata"],
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )
    scheme = LongRunRestriction(ordering=["y1", "y2"], shock_names=["permanent", "transitory"])
    return fitted.set_identification_strategy(scheme)


def _identified_with_singular_draws(permanent_transitory_2v, var_data_2v, n_bad: int = 5):
    """Same pipeline, but the first `n_bad` draws of chain 0 have M = 0."""
    from impulso.identification import LongRunRestriction
    from impulso.volatility import Constant

    idata = permanent_transitory_2v["idata"].copy()
    B = idata.posterior["B"].values.copy()
    B[0, :n_bad] = np.eye(2)
    idata.posterior["B"] = (("chain", "draw", "var", "coeff"), B)

    fitted = FittedVAR(
        idata=idata,
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )
    scheme = LongRunRestriction(ordering=["y1", "y2"], shock_names=["permanent", "transitory"])
    return fitted.set_identification_strategy(scheme)


class TestLongRunRestrictionPipeline:
    """LongRunRestriction through the FittedVAR -> IdentifiedVAR pipeline."""

    # --- 20. shock matrix labelling and diagnostics ------------------------

    def test_shock_matrix_labels_and_attrs(self, identified_long_run, permanent_transitory_2v):
        P = identified_long_run.shock_matrix()
        assert P.dims == ("chain", "draw", "response", "shock")
        assert list(P.coords["response"].values) == ["y1", "y2"]
        assert list(P.coords["shock"].values) == ["permanent", "transitory"]
        np.testing.assert_allclose(P.values, np.broadcast_to(permanent_transitory_2v["P_true"], P.shape), atol=1e-12)
        assert P.attrs["long_run_singular_draws"] == 0.0
        assert P.attrs["long_run_explosive_draws"] == 0.0
        assert "sign_restriction_acceptance_rate" not in P.attrs

    # --- 21. the headline: cumulative IRF converges to Theta(1) ------------

    def test_cumulative_irf_converges_to_the_imposed_long_run_matrix(
        self, identified_long_run, permanent_transitory_2v
    ):
        """Sum of IRFs over horizons is C(1) P — computed here by brute force."""
        irf = identified_long_run.impulse_response(horizon=200)
        cumulative = np.cumsum(irf.idata.posterior_predictive["irf"].values, axis=2)
        long_run = cumulative[:, :, -1, :, :]

        G = permanent_transitory_2v["G"]
        np.testing.assert_allclose(long_run, np.broadcast_to(G, long_run.shape), atol=1e-8)
        # The restriction: the transitory shock has no permanent effect on y1.
        assert np.abs(long_run[..., 0, 1]).max() < 1e-10
        np.testing.assert_allclose(long_run[..., 0, 0], 1.0, atol=1e-8)

    # --- 22. FEVD is well-behaved ------------------------------------------

    def test_fevd_shares_sum_to_one_without_warning(self, identified_long_run):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fevd = identified_long_run.fevd(horizon=10)
        shares = fevd.idata.posterior_predictive["fevd"].values
        np.testing.assert_allclose(shares.sum(axis=-1), 1.0, atol=1e-10)

    # --- 23. undefined draws stay undefined through FEVD -------------------

    def test_fevd_propagates_nan_draws(self, permanent_transitory_2v, var_data_2v):
        identified = _identified_with_singular_draws(permanent_transitory_2v, var_data_2v)
        with pytest.warns(UserWarning, match="long-run multiplier"):
            fevd = identified.fevd(horizon=5)
        shares = fevd.idata.posterior_predictive["fevd"].values

        assert np.isnan(shares[0, :5]).all()
        assert not np.isnan(shares[0, 5:]).any()
        good = np.concatenate([shares[0, 5:], shares[1]], axis=0)
        np.testing.assert_allclose(good.sum(axis=-1), 1.0, atol=1e-10)

    # --- 24. historical decomposition stays additive ------------------------

    def test_historical_decomposition_is_additive(self, identified_long_run, var_data_2v):
        hd = identified_long_run.historical_decomposition()
        contributions = hd.idata.posterior_predictive["hd"].values
        baseline = hd.idata.posterior_predictive["baseline"].values

        assert list(hd.idata.posterior_predictive["hd"].coords["shock"].values) == ["permanent", "transitory"]
        reconstructed = contributions.sum(axis=-1) + baseline
        expected = var_data_2v.endog[1:]
        np.testing.assert_allclose(reconstructed, np.broadcast_to(expected, reconstructed.shape), atol=1e-8)

    # --- 25. undefined draws do not crash the decomposition ----------------

    def test_historical_decomposition_survives_nan_draws(self, permanent_transitory_2v, var_data_2v):
        identified = _identified_with_singular_draws(permanent_transitory_2v, var_data_2v)
        with pytest.warns(UserWarning, match="long-run multiplier"):
            hd = identified.historical_decomposition()
        contributions = hd.idata.posterior_predictive["hd"].values

        assert np.isnan(contributions[0, :5]).all()
        assert not np.isnan(contributions[0, 5:]).any()


@pytest.fixture
def identified_max_share(single_driver_2v, var_data_2v):
    """IdentifiedVAR over the exact-arithmetic max-share fixture."""
    from impulso.identification import MaxShare
    from impulso.volatility import Constant

    fitted = FittedVAR(
        idata=single_driver_2v["idata"],
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )
    return fitted.set_identification_strategy(MaxShare(target="y2", band=(6, 32)))


class TestMaxSharePipeline:
    """MaxShare through the FittedVAR -> IdentifiedVAR pipeline."""

    def test_shock_matrix_labels_and_attrs(self, identified_max_share, single_driver_2v):
        P = identified_max_share.shock_matrix()

        assert P.dims == ("chain", "draw", "response", "shock")
        assert list(P.coords["response"].values) == ["y1", "y2"]
        assert list(P.coords["shock"].values) == ["max_share", "unidentified_1"]
        expected = np.broadcast_to(single_driver_2v["P_true"][:, 0], P.values[..., :, 0].shape)
        np.testing.assert_allclose(P.values[..., :, 0], expected, atol=1e-8)
        assert P.attrs["max_share_share_median"] >= 1 - 1e-10
        assert P.attrs["max_share_singular_draws"] == 0.0
        assert "sign_restriction_acceptance_rate" not in P.attrs

    def test_fevd_masks_the_unidentified_column(self, identified_max_share):
        with pytest.warns(UserWarning, match="unidentified"):
            fevd = identified_max_share.fevd(horizon=10)
        shares = fevd.idata.posterior_predictive["fevd"].values

        identified_col = shares[..., 0]
        assert np.isfinite(identified_col).all()
        assert ((identified_col >= 0.0) & (identified_col <= 1.0)).all()
        assert np.isnan(shares[..., 1]).all()

    def test_fevd_recovers_the_full_share_for_the_target(self, identified_max_share):
        """The fixture's shock 0 drives y2 entirely, so its FEVD share is 1."""
        with pytest.warns(UserWarning, match="unidentified"):
            fevd = identified_max_share.fevd(horizon=20)
        shares = fevd.idata.posterior_predictive["fevd"].values
        np.testing.assert_allclose(shares[..., 1, 0], 1.0, atol=1e-10)

    def test_fevd_keeps_nan_draws_nan(self, single_driver_2v, var_data_2v):
        """A blanked draw must not surface as a clean 0.0 share."""
        from impulso.identification import MaxShare
        from impulso.volatility import Constant

        idata = single_driver_2v["idata"].copy()
        B = idata.posterior["B"].values.copy()
        phi = 2.0 * np.pi / 12.0  # unit-circle root at a period inside the band
        B[0, :5] = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        idata.posterior["B"] = (("chain", "draw", "var", "coeff"), B)

        fitted = FittedVAR(
            idata=idata,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        scheme = MaxShare(target="y2", band=(6, 32), max_condition=100.0)
        identified = fitted.set_identification_strategy(scheme)
        with pytest.warns(UserWarning, match="numerically undefined"):
            fevd = identified.fevd(horizon=5)
        shares = fevd.idata.posterior_predictive["fevd"].values

        assert np.isnan(shares[0, :5, :, :, 0]).all()
        assert np.isfinite(shares[0, 5:, :, :, 0]).all()

    def test_historical_decomposition_collapses_the_remainder(self, identified_max_share, var_data_2v):
        hd = identified_max_share.historical_decomposition()
        contributions = hd.idata.posterior_predictive["hd"]

        assert list(contributions.coords["shock"].values) == ["max_share", "unidentified_remainder"]
        baseline = hd.idata.posterior_predictive["baseline"].values
        reconstructed = contributions.values.sum(axis=-1) + baseline
        expected = np.broadcast_to(var_data_2v.endog[1:], reconstructed.shape)
        np.testing.assert_allclose(reconstructed, expected, atol=1e-8)

    def test_impulse_response_shapes_and_labels(self, identified_max_share):
        irf = identified_max_share.impulse_response(horizon=5)
        da = irf.idata.posterior_predictive["irf"]

        assert da.shape == (2, 50, 6, 2, 2)
        assert list(da.coords["shock"].values) == ["max_share", "unidentified_1"]
        assert list(da.coords["response"].values) == ["y1", "y2"]
        impact = da.values[..., 0, :, 0]
        np.testing.assert_allclose(impact, np.broadcast_to([0.3, 0.7], impact.shape), atol=1e-8)
