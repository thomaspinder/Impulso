"""Tests for ProxySVAR external-instrument identification."""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from impulso.data import VARData
from impulso.identification import ProxySVAR


def _make_svar_world(relevance_noise: float = 0.1, T: int = 400, seed: int = 3):
    """Simulate a 2-var SVAR(1) with known impact matrix and instrument.

    Returns (data, idata, P_true, instrument) where the posterior is a
    small synthetic ensemble centred exactly on the truth so identification
    error comes only from the finite instrument sample.
    """
    rng = np.random.default_rng(seed)
    n = 2
    B_true = np.array([[0.5, 0.1], [0.0, 0.4]])
    c_true = np.array([0.0, 0.0])
    P_true = np.array([[1.0, 0.0], [0.6, 0.8]])  # impact matrix, col 0 = target shock

    eps = rng.standard_normal((T, n))  # structural shocks
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = c_true + B_true @ y[t - 1] + P_true @ eps[t]

    index = pd.date_range("1990-01-01", periods=T, freq="MS")
    data = VARData(endog=y, endog_names=["y1", "y2"], index=index)

    # Instrument: true target shock + noise, on the effective sample.
    z = eps[1:, 0] + relevance_noise * rng.standard_normal(T - 1)
    instrument = pd.Series(z, index=index[1:], name="z")

    # Synthetic posterior centred on the truth (tiny jitter for spread).
    n_chains, n_draws = 2, 40
    sigma_true = P_true @ P_true.T
    L_true = np.linalg.cholesky(sigma_true)
    B = np.broadcast_to(B_true, (n_chains, n_draws, n, n)).copy()
    B += 0.001 * rng.standard_normal(B.shape)
    intercept = np.broadcast_to(c_true, (n_chains, n_draws, n)).copy()
    L = np.broadcast_to(L_true, (n_chains, n_draws, n, n)).copy()

    posterior = xr.Dataset({
        "B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"]),
        "intercept": xr.DataArray(intercept, dims=["chain", "draw", "var"]),
        "L": xr.DataArray(L, dims=["chain", "draw", "var1", "var2"]),
    })
    idata = az.InferenceData(posterior=posterior)
    return data, idata, P_true, instrument, L


class TestProxySVARRecovery:
    def test_recovers_impact_column_up_to_normalisation(self):
        data, idata, _P_true, instrument, L = _make_svar_world(relevance_noise=0.05)
        scheme = ProxySVAR(instrument=instrument, policy_variable="y1")
        P = scheme.identify(L, ["y1", "y2"], posterior=idata.posterior, data=data, n_lags=1)

        # True relative impact: P_true[:, 0] / P_true[0, 0] = [1, 0.6].
        med_ratio = np.median(P[:, :, 1, 0] / P[:, :, 0, 0])
        assert abs(med_ratio - 0.6) < 0.1

    def test_error_shrinks_with_instrument_noise(self):
        errs = []
        for noise in (1.0, 0.01):
            data, idata, _P_true, instrument, L = _make_svar_world(relevance_noise=noise, T=2000)
            scheme = ProxySVAR(instrument=instrument, policy_variable="y1")
            P = scheme.identify(L, ["y1", "y2"], posterior=idata.posterior, data=data, n_lags=1)
            med_ratio = np.median(P[:, :, 1, 0] / P[:, :, 0, 0])
            errs.append(abs(med_ratio - 0.6))
        assert errs[1] < errs[0]

    def test_sigma_consistency_when_scale_none(self):
        data, idata, _P_true, instrument, L = _make_svar_world()
        scheme = ProxySVAR(instrument=instrument, policy_variable="y1")
        P = scheme.identify(L, ["y1", "y2"], posterior=idata.posterior, data=data, n_lags=1)
        sigma = L @ np.swapaxes(L, -1, -2)
        np.testing.assert_allclose(P @ np.swapaxes(P, -1, -2), sigma, atol=1e-10)

    def test_unit_effect_scale(self):
        data, idata, _P_true, instrument, L = _make_svar_world()
        scheme = ProxySVAR(instrument=instrument, policy_variable="y1", scale=10.0)
        P = scheme.identify(L, ["y1", "y2"], posterior=idata.posterior, data=data, n_lags=1)
        np.testing.assert_allclose(P[:, :, 0, 0], 10.0, atol=1e-10)

    def test_policy_impact_positive_by_construction(self):
        data, idata, _P_true, instrument, L = _make_svar_world()
        scheme = ProxySVAR(instrument=instrument, policy_variable="y1")
        P = scheme.identify(L, ["y1", "y2"], posterior=idata.posterior, data=data, n_lags=1)
        assert (P[:, :, 0, 0] > 0).all()


class TestProxySVARDiagnostics:
    def test_strong_instrument_diagnostics(self):
        data, idata, _P_true, instrument, L = _make_svar_world(relevance_noise=0.05)
        scheme = ProxySVAR(instrument=instrument, policy_variable="y1")
        scheme.identify(L, ["y1", "y2"], posterior=idata.posterior, data=data, n_lags=1)
        diag = scheme._last_diagnostics
        assert diag["proxy_first_stage_f_median"] > 10.0
        assert diag["proxy_first_stage_f_q05"] <= diag["proxy_first_stage_f_median"]

    def test_public_first_stage_matches_diagnostics(self):
        data, idata, _P_true, instrument, L = _make_svar_world(relevance_noise=0.05)
        scheme = ProxySVAR(instrument=instrument, policy_variable="y1")
        scheme.identify(L, ["y1", "y2"], posterior=idata.posterior, data=data, n_lags=1)
        f_draws = scheme.first_stage(idata.posterior, data, n_lags=1)
        assert f_draws.shape == (2, 40)
        assert np.isclose(
            float(np.median(f_draws)),
            scheme._last_diagnostics["proxy_first_stage_f_median"],
        )

    def test_pure_noise_instrument_warns_weak(self):
        data, idata, _P_true, _, L = _make_svar_world()
        rng = np.random.default_rng(99)
        noise_instrument = pd.Series(rng.standard_normal(len(data.index) - 1), index=data.index[1:], name="noise")
        scheme = ProxySVAR(instrument=noise_instrument, policy_variable="y1")
        with pytest.warns(UserWarning, match="Weak instrument"):
            scheme.identify(L, ["y1", "y2"], posterior=idata.posterior, data=data, n_lags=1)
        assert scheme._last_diagnostics["proxy_first_stage_f_median"] < 10.0


class TestProxySVARAlignment:
    def test_shorter_offset_instrument_aligns(self):
        data, idata, _P_true, instrument, L = _make_svar_world(relevance_noise=0.05, T=600)
        # Drop the first 100 months of the instrument: shorter overlap, same answer.
        short = instrument.iloc[100:]
        scheme = ProxySVAR(instrument=short, policy_variable="y1")
        P = scheme.identify(L, ["y1", "y2"], posterior=idata.posterior, data=data, n_lags=1)
        med_ratio = np.median(P[:, :, 1, 0] / P[:, :, 0, 0])
        assert abs(med_ratio - 0.6) < 0.1

    def test_disjoint_index_raises(self):
        data, idata, _P_true, instrument, L = _make_svar_world()
        disjoint = pd.Series(
            instrument.to_numpy(),
            index=pd.date_range("2050-01-01", periods=len(instrument), freq="MS"),
        )
        scheme = ProxySVAR(instrument=disjoint, policy_variable="y1")
        with pytest.raises(ValueError, match="does not overlap"):
            scheme.identify(L, ["y1", "y2"], posterior=idata.posterior, data=data, n_lags=1)

    def test_missing_context_raises(self):
        _data, _idata, _P_true, instrument, L = _make_svar_world()
        scheme = ProxySVAR(instrument=instrument, policy_variable="y1")
        with pytest.raises(ValueError, match="requires posterior, data, and n_lags"):
            scheme.identify(L, ["y1", "y2"])

    def test_unknown_policy_variable_raises(self):
        data, idata, _P_true, instrument, L = _make_svar_world()
        scheme = ProxySVAR(instrument=instrument, policy_variable="nope")
        with pytest.raises(ValueError, match="policy_variable"):
            scheme.identify(L, ["y1", "y2"], posterior=idata.posterior, data=data, n_lags=1)


class TestProxySVARShockCoords:
    def test_labels(self):
        instrument = pd.Series([0.0], index=pd.DatetimeIndex(["2000-01-01"]))
        scheme = ProxySVAR(instrument=instrument, policy_variable="x", shock_name="oil_news")
        assert scheme.shock_coords(3) == ["oil_news", "unidentified_1", "unidentified_2"]


class TestProxySVARPipeline:
    def test_shock_matrix_attrs_carry_diagnostics(self):
        """Through IdentifiedVAR.shock_matrix, diagnostics land in attrs."""
        from impulso.identified import IdentifiedVAR
        from impulso.volatility import Constant

        data, idata, _P_true, instrument, _L = _make_svar_world(relevance_noise=0.05)
        scheme = ProxySVAR(instrument=instrument, policy_variable="y1", shock_name="target")
        ivar = IdentifiedVAR(
            idata=idata,
            n_lags=1,
            data=data,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=scheme,
        )
        sm = ivar.shock_matrix()
        assert "proxy_first_stage_f_median" in sm.attrs
        assert list(sm.coords["shock"].values) == ["target", "unidentified_1"]

    def test_impulse_response_through_pipeline(self):
        from impulso.identified import IdentifiedVAR
        from impulso.volatility import Constant

        data, idata, _P_true, instrument, _L = _make_svar_world(relevance_noise=0.05)
        scheme = ProxySVAR(instrument=instrument, policy_variable="y1", shock_name="target", scale=1.0)
        ivar = IdentifiedVAR(
            idata=idata,
            n_lags=1,
            data=data,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=scheme,
        )
        irf = ivar.impulse_response(horizon=10)
        med = irf.median()
        # Impact response of the policy variable to the identified shock = scale.
        vals = irf.idata.posterior_predictive["irf"].sel(shock="target", response="y1").isel(horizon=0)
        np.testing.assert_allclose(vals.values, 1.0, atol=1e-10)
        assert med is not None


class TestProxySVARGuardRails:
    def _identified(self, scale=None):
        from impulso.identified import IdentifiedVAR
        from impulso.volatility import Constant

        data, idata, _P_true, instrument, _L = _make_svar_world(relevance_noise=0.05)
        scheme = ProxySVAR(instrument=instrument, policy_variable="y1", shock_name="target", scale=scale)
        return IdentifiedVAR(
            idata=idata,
            n_lags=1,
            data=data,
            var_names=["y1", "y2"],
            volatility=Constant(),
            scheme=scheme,
        )

    def test_fevd_masks_unidentified_columns(self):
        ivar = self._identified()
        with pytest.warns(UserWarning, match="rotation-arbitrary"):
            fevd = ivar.fevd(horizon=5)
        da = fevd.idata.posterior_predictive["fevd"]
        assert np.isnan(da.sel(shock="unidentified_1").values).all()
        target = da.sel(shock="target").values
        assert not np.isnan(target).any()
        assert ((target >= 0) & (target <= 1)).all()

    def test_fevd_warns_under_unit_effect_scale(self):
        ivar = self._identified(scale=10.0)
        with pytest.warns(UserWarning, match="unit-effect"):
            ivar.fevd(horizon=5)

    def test_hd_collapses_unidentified_to_remainder(self):
        ivar = self._identified()
        hd = ivar.historical_decomposition()
        da = hd.idata.posterior_predictive["hd"]
        assert list(da.coords["shock"].values) == ["target", "unidentified_remainder"]

    def test_hd_additivity_preserved(self):
        """Sum over shock columns reproduces the reduced-form residual, so
        collapsing the unidentified columns must not change the total."""
        from impulso._residuals import reduced_form_residuals

        ivar = self._identified()
        da = ivar.historical_decomposition().idata.posterior_predictive["hd"]
        total = da.sum("shock").values  # (C, D, T, n)
        resid = reduced_form_residuals(ivar.idata.posterior, ivar.data, n_lags=1)
        np.testing.assert_allclose(total, resid, atol=1e-10)

    def test_hd_identified_column_scale_invariant(self):
        """The identified shock's HD contribution must not depend on the
        unit-effect rescaling of the impact column."""
        hd_sd = self._identified(scale=None).historical_decomposition()
        hd_unit = self._identified(scale=10.0).historical_decomposition()
        a = hd_sd.idata.posterior_predictive["hd"].sel(shock="target").values
        b = hd_unit.idata.posterior_predictive["hd"].sel(shock="target").values
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_cholesky_fevd_unaffected(self, synthetic_idata_2v, var_data_2v):
        """Fully-identified schemes see no masking and no warning."""
        import warnings as _warnings

        from impulso.fitted import FittedVAR
        from impulso.identification import Cholesky
        from impulso.volatility import Constant

        fitted = FittedVAR(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        ivar = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            fevd = ivar.fevd(horizon=5)
        assert not np.isnan(fevd.idata.posterior_predictive["fevd"].values).any()


@pytest.mark.slow
class TestProxySVARPipelineSlow:
    def test_full_fit_identify_irf(self):
        """End-to-end: simulate SVAR, fit with NUTS, identify with the
        instrument, recover the relative impact within estimation error."""
        from impulso import VAR
        from impulso.samplers import NUTSSampler

        rng = np.random.default_rng(11)
        T, n = 300, 2
        B_true = np.array([[0.5, 0.1], [0.0, 0.4]])
        P_true = np.array([[1.0, 0.0], [0.6, 0.8]])
        eps = rng.standard_normal((T, n))
        y = np.zeros((T, n))
        for t in range(1, T):
            y[t] = B_true @ y[t - 1] + P_true @ eps[t]
        index = pd.date_range("1990-01-01", periods=T, freq="MS")
        data = VARData(endog=y, endog_names=["y1", "y2"], index=index)
        instrument = pd.Series(eps[1:, 0] + 0.1 * rng.standard_normal(T - 1), index=index[1:])

        sampler = NUTSSampler(draws=200, tune=300, chains=2, cores=1, random_seed=42)
        fitted = VAR(lags=1, prior="minnesota").fit(data, sampler=sampler)
        scheme = ProxySVAR(instrument=instrument, policy_variable="y1", shock_name="target")
        ivar = fitted.set_identification_strategy(scheme)

        irf = ivar.impulse_response(horizon=10)
        da = irf.idata.posterior_predictive["irf"].sel(shock="target")
        impact = da.isel(horizon=0)
        ratio = (impact.sel(response="y2") / impact.sel(response="y1")).median().item()
        assert abs(ratio - 0.6) < 0.25  # generous: finite sample + estimation error

        sm = ivar.shock_matrix()
        assert sm.attrs["proxy_first_stage_f_median"] > 10.0

    def test_sv_composition_per_t_identify(self, var_data_2v):
        """ProxySVAR composed with stochastic volatility: the per-t identify
        path must produce a full time-indexed structural matrix, and IRF/HD
        must flow through it. This is the headline differentiator."""
        from impulso import VAR
        from impulso.samplers import NUTSSampler

        # var_data_2v DGP: y_t = 0.5 y_{t-1} + 0.1 eps_t (seed 42). An
        # instrument for shock 1 from the known DGP innovations.
        y = var_data_2v.endog
        u_true = y[1:] - 0.5 * y[:-1]
        rng = np.random.default_rng(7)
        instrument = pd.Series(
            u_true[:, 0] + 0.02 * rng.standard_normal(len(u_true)),
            index=var_data_2v.index[1:],
        )

        sampler = NUTSSampler(draws=100, tune=200, chains=2, cores=1, random_seed=42)
        fitted = VAR(lags=1, volatility="sv").fit(var_data_2v, sampler=sampler)
        scheme = ProxySVAR(instrument=instrument, policy_variable="y1", shock_name="target")
        ivar = fitted.set_identification_strategy(scheme)

        sm = ivar.shock_matrix(at="all")
        assert "time" in sm.dims
        assert sm.shape[2] == len(var_data_2v.index) - 1
        assert list(sm.coords["shock"].values) == ["target", "unidentified_1"]
        assert not np.isnan(sm.values).any()

        irf = ivar.impulse_response(horizon=5, at="last")
        da = irf.idata.posterior_predictive["irf"].sel(shock="target")
        assert (da.isel(horizon=0).sel(response="y1") > 0).all()
