"""Invariant tests for the propagated historical decomposition.

PR 1 of the scenario-analysis stack (ADR-0005): historical_decomposition
propagates shock contributions through the lag dynamics and exposes the
deterministic baseline, restoring the textbook additivity identity
y_t = baseline_t + sum_j c_{j,t}.
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from impulso._linalg import lag_matrices
from impulso._ma import compute_ma_phi
from impulso._residuals import reduced_form_residuals
from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.identification import Cholesky
from impulso.identified import IdentifiedVAR
from impulso.results import HistoricalDecompositionResult
from impulso.volatility import Constant


@pytest.fixture
def identified_2v(synthetic_idata_2v, var_data_2v):
    """Cholesky-identified 2-var VAR(1) from the synthetic posterior."""
    fitted = FittedVAR(
        idata=synthetic_idata_2v,
        n_lags=1,
        data=var_data_2v,
        var_names=["y1", "y2"],
        volatility=Constant(),
    )
    return fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))


class TestAdditivity:
    def test_baseline_plus_contributions_reproduce_observations(self, identified_2v, var_data_2v):
        """y_t = baseline_t + sum_j c_{j,t} exactly, for every posterior draw."""
        pp = identified_2v.historical_decomposition().idata.posterior_predictive
        total = pp["hd"].sum("shock").values + pp["baseline"].values
        y = var_data_2v.endog[1:]
        np.testing.assert_allclose(total, np.broadcast_to(y, total.shape), atol=1e-8)

    def test_contributions_match_ma_form(self, identified_2v, synthetic_idata_2v, var_data_2v):
        """The recursion equals the moving-average form sum_s Phi_{t-s} P eps_s."""
        t_max = 30
        hd_vals = identified_2v.historical_decomposition().idata.posterior_predictive["hd"].values[:, :, :t_max]

        P = identified_2v.shock_matrix().values
        resid = reduced_form_residuals(synthetic_idata_2v.posterior, var_data_2v, n_lags=1)
        eps = np.einsum("cdij,cdtj->cdti", np.linalg.inv(P), resid)
        impact = P[:, :, np.newaxis, :, :] * eps[:, :, :, np.newaxis, :]
        Phi = compute_ma_phi(lag_matrices(synthetic_idata_2v.posterior["B"].values, 1), t_max - 1)

        expected = np.zeros_like(hd_vals)
        for t in range(t_max):
            for s in range(t + 1):
                expected[:, :, t] += np.einsum("cdij,cdjs->cdis", Phi[:, :, t - s], impact[:, :, s])
        np.testing.assert_allclose(hd_vals, expected, atol=1e-8)

    def test_zero_dynamics_reduces_to_impact(self, synthetic_idata_2v, var_data_2v):
        """With B = 0 the propagated HD equals the contemporaneous impact."""
        posterior = synthetic_idata_2v.posterior.copy(deep=True)
        posterior["B"].values[:] = 0.0
        idata = az.InferenceData(posterior=posterior)
        fitted = FittedVAR(
            idata=idata,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        ivar = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))

        pp = ivar.historical_decomposition().idata.posterior_predictive
        P = ivar.shock_matrix().values
        resid = reduced_form_residuals(posterior, var_data_2v, n_lags=1)
        eps = np.einsum("cdij,cdtj->cdti", np.linalg.inv(P), resid)
        impact = P[:, :, np.newaxis, :, :] * eps[:, :, :, np.newaxis, :]
        np.testing.assert_allclose(pp["hd"].values, impact, atol=1e-10)
        # With zero dynamics the baseline is the flat intercept path.
        intercept = posterior["intercept"].values
        expected_baseline = np.broadcast_to(intercept[:, :, np.newaxis, :], pp["baseline"].shape)
        np.testing.assert_allclose(pp["baseline"].values, expected_baseline, atol=1e-12)


class TestExogAndMultiLag:
    def test_additivity_with_exog_and_two_lags(self):
        """Additivity holds with exogenous regressors and n_lags > 1."""
        rng = np.random.default_rng(11)
        n_chains, n_draws, n_vars, n_lags, n_exog, t = 2, 25, 2, 2, 1, 40
        L = np.zeros((n_chains, n_draws, n_vars, n_vars))
        for c in range(n_chains):
            for d in range(n_draws):
                A = rng.standard_normal((n_vars, n_vars)) * 0.4
                L[c, d] = np.linalg.cholesky(A @ A.T + np.eye(n_vars))
        posterior = xr.Dataset({
            "B": xr.DataArray(
                rng.standard_normal((n_chains, n_draws, n_vars, n_vars * n_lags)) * 0.2,
                dims=["chain", "draw", "var", "coeff"],
            ),
            "B_exog": xr.DataArray(
                rng.standard_normal((n_chains, n_draws, n_vars, n_exog)),
                dims=["chain", "draw", "var", "exog"],
            ),
            "intercept": xr.DataArray(
                rng.standard_normal((n_chains, n_draws, n_vars)) * 0.01,
                dims=["chain", "draw", "var"],
            ),
            "L": xr.DataArray(L, dims=["chain", "draw", "var1", "var2"]),
        })
        data = VARData(
            endog=rng.standard_normal((t, n_vars)),
            endog_names=["y1", "y2"],
            exog=rng.standard_normal((t, n_exog)),
            exog_names=["z"],
            index=pd.date_range("2023-01-02", periods=t, freq="W-MON"),
        )
        fitted = FittedVAR(
            idata=az.InferenceData(posterior=posterior),
            n_lags=n_lags,
            data=data,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        ivar = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))

        pp = ivar.historical_decomposition().idata.posterior_predictive
        total = pp["hd"].sum("shock").values + pp["baseline"].values
        y = data.endog[n_lags:]
        np.testing.assert_allclose(total, np.broadcast_to(y, total.shape), atol=1e-8)


class TestWindowing:
    def test_start_end_slice_matches_full_run(self, identified_2v, var_data_2v):
        """start/end only slice the output; propagation runs from the sample start."""
        full = identified_2v.historical_decomposition().idata.posterior_predictive
        idx = var_data_2v.index[1:]
        sub = identified_2v.historical_decomposition(start=idx[50], end=idx[120]).idata.posterior_predictive
        np.testing.assert_array_equal(sub["hd"].values, full["hd"].values[:, :, 50:121])
        np.testing.assert_array_equal(sub["baseline"].values, full["baseline"].values[:, :, 50:121])


class _TimeVaryingVol:
    """Fake volatility process with a deterministically time-varying L_t.

    Delegates to `Constant` for the base factor and scales it by
    `1 + t / (2T)`, so `P_t` genuinely differs across `t` — the per-t
    propagation path cannot pass on a constant-vol alignment bug.
    """

    name = "fake-sv"
    is_time_varying = True

    def __init__(self, T_eff: int):
        self._constant = Constant()
        self._T = T_eff

    def build_pymc_latent(self, n_vars, T):  # pragma: no cover
        raise NotImplementedError

    def _scale(self, t: int) -> float:
        return 1.0 + t / (2.0 * self._T)

    def cholesky_at(self, posterior, t):
        idx = self._T - 1 if t is None else t
        return self._constant.cholesky_at(posterior, t=None) * self._scale(idx)

    def cholesky_path(self, posterior, T):
        L = self._constant.cholesky_at(posterior, t=None)  # (C, D, n, n)
        scales = np.array([self._scale(t) for t in range(T)])
        return L[:, :, np.newaxis, :, :] * scales[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

    def forecast_cholesky_path(self, posterior, steps, rng):  # pragma: no cover
        raise NotImplementedError


class TestPerTStochasticVolatility:
    def test_per_t_contributions_match_hand_recursion(self, synthetic_idata_2v, var_data_2v):
        """With genuinely time-varying P_t, HD equals the hand-rolled per-t recursion."""
        T_eff = var_data_2v.endog.shape[0] - 1
        identified = IdentifiedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=_TimeVaryingVol(T_eff),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        pp = identified.historical_decomposition().idata.posterior_predictive

        P_path = identified.shock_matrix(at="all").values  # (C, D, T, n, n)
        assert not np.allclose(P_path[:, :, 0], P_path[:, :, -1])  # the alignment is real

        resid = reduced_form_residuals(synthetic_idata_2v.posterior, var_data_2v, n_lags=1)
        eps = np.einsum("cdtij,cdtj->cdti", np.linalg.inv(P_path), resid)
        impact = P_path * eps[:, :, :, np.newaxis, :]
        B = synthetic_idata_2v.posterior["B"].values
        expected = np.zeros_like(impact)
        carry = np.zeros(impact.shape[:2] + impact.shape[3:])
        for t in range(impact.shape[2]):
            carry = impact[:, :, t] + np.einsum("cdij,cdjs->cdis", B, carry)
            expected[:, :, t] = carry
        np.testing.assert_allclose(pp["hd"].values, expected, atol=1e-10)

    def test_per_t_additivity_holds(self, synthetic_idata_2v, var_data_2v):
        """Additivity is exact under per-t P_t as well."""
        T_eff = var_data_2v.endog.shape[0] - 1
        identified = IdentifiedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=_TimeVaryingVol(T_eff),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        pp = identified.historical_decomposition().idata.posterior_predictive
        total = pp["hd"].sum("shock").values + pp["baseline"].values
        y = var_data_2v.endog[1:]
        np.testing.assert_allclose(total, np.broadcast_to(y, total.shape), atol=1e-8)


class TestResultSurface:
    def test_baseline_accessor_returns_frame(self, identified_2v, var_data_2v):
        result = identified_2v.historical_decomposition()
        frame = result.baseline()
        assert list(frame.columns) == ["y1", "y2"]
        assert len(frame) == var_data_2v.endog.shape[0] - 1
        assert frame.index.name == "time"

    def test_baseline_accessor_guards_missing_variable(self, rng):
        hd = xr.DataArray(
            rng.standard_normal((2, 5, 10, 2, 2)),
            dims=["chain", "draw", "time", "response", "shock"],
            coords={"response": ["y1", "y2"], "shock": ["y1", "y2"]},
            name="hd",
        )
        result = HistoricalDecompositionResult(
            idata=az.InferenceData(posterior_predictive=xr.Dataset({"hd": hd})),
            var_names=["y1", "y2"],
        )
        with pytest.raises(ValueError, match="baseline"):
            result.baseline()

    def test_cumulative_parameter_retired(self, identified_2v):
        with pytest.raises(TypeError):
            identified_2v.historical_decomposition(cumulative=True)
