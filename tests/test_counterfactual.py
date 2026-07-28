"""Invariant tests for IdentifiedVAR.counterfactual (PR 2 of the scenario stack).

Core identities (design + ADR-0005): reproduction with no edits; full-sample
duality with the historical decomposition; windowed duality (zero before the
window, persists after it, and is not the sliced full-sample contribution);
plus the condition-vocabulary guards.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from impulso._scenario import apply_shock_edits
from impulso.fitted import FittedVAR
from impulso.identification import Cholesky
from impulso.identified import IdentifiedVAR
from impulso.scenario import ShockPath, VariablePath
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


class TestVocabulary:
    def test_scalar_values_coerced_to_float(self):
        sp = ShockPath(shock="a", values=0)
        assert isinstance(sp.values, float)

    def test_array_values_readonly(self):
        sp = ShockPath(shock="a", values=np.arange(3.0))
        assert isinstance(sp.values, np.ndarray)
        with pytest.raises(ValueError, match="read-only"):
            sp.values[0] = 9.0

    def test_two_dimensional_values_rejected(self):
        with pytest.raises(ValueError, match="1-D"):
            ShockPath(shock="a", values=np.zeros((2, 2)))

    def test_empty_values_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            ShockPath(shock="a", values=np.array([]))

    def test_start_after_end_rejected_at_construction(self):
        with pytest.raises(ValueError, match="after end"):
            ShockPath(shock="a", values=0.0, start="2005-01-01", end="2004-01-01")

    def test_string_timestamps_coerced(self):
        sp = ShockPath(shock="a", values=0.0, start="2004-01-01")
        assert sp.start == pd.Timestamp("2004-01-01")

    def test_variable_path_allows_nan(self):
        vp = VariablePath(variable="x", values=np.array([1.0, np.nan, 2.0]))
        assert np.isnan(vp.values[1])


class TestReproduction:
    def test_no_edits_reproduce_observations(self, identified_2v, var_data_2v):
        """The engine's reproduction identity: shocks=[] gives back the data."""
        cf = identified_2v.counterfactual(shocks=[])
        draws = cf.idata.posterior_predictive["counterfactual"].values
        y = var_data_2v.endog[1:]
        np.testing.assert_allclose(draws, np.broadcast_to(y, draws.shape), atol=1e-8)

    def test_scalar_and_array_edits_agree(self, identified_2v, var_data_2v):
        idx = var_data_2v.index[1:]
        start, end = idx[40], idx[59]
        a = identified_2v.counterfactual(shocks=[ShockPath(shock="y1", values=0.0, start=start, end=end)])
        b = identified_2v.counterfactual(shocks=[ShockPath(shock="y1", values=np.zeros(20), start=start, end=end)])
        np.testing.assert_array_equal(
            a.idata.posterior_predictive["counterfactual"].values,
            b.idata.posterior_predictive["counterfactual"].values,
        )


class TestHDDuality:
    def test_full_sample_zero_edit_equals_hd_contribution(self, identified_2v, var_data_2v):
        """actual - counterfactual(shock j zeroed) equals HD contribution of j, per draw."""
        cf = identified_2v.counterfactual(shocks=[ShockPath(shock="y1", values=0.0)])
        hd = identified_2v.historical_decomposition()
        diff = var_data_2v.endog[1:][None, None] - cf.idata.posterior_predictive["counterfactual"].values
        contribution = hd.idata.posterior_predictive["hd"].sel(shock="y1").values
        np.testing.assert_allclose(diff, contribution, atol=1e-8)

    def test_windowed_duality(self, identified_2v, var_data_2v):
        """Windowed zero-edit: zero before, persists after, ≠ sliced HD contribution."""
        idx = var_data_2v.index[1:]
        t0, t1 = 80, 120  # edit window covers positions t0..t1-1
        cf = identified_2v.counterfactual(shocks=[ShockPath(shock="y1", values=0.0, start=idx[t0], end=idx[t1 - 1])])
        diff = var_data_2v.endog[1:][None, None] - cf.idata.posterior_predictive["counterfactual"].values

        np.testing.assert_allclose(diff[:, :, :t0], 0.0, atol=1e-10)
        assert np.abs(diff[:, :, t0:t1]).max() > 1e-6
        assert np.abs(diff[:, :, t1 : t1 + 5]).max() > 1e-8

        hd = identified_2v.historical_decomposition()
        contribution = hd.idata.posterior_predictive["hd"].sel(shock="y1").values
        assert not np.allclose(diff[:, :, t0:t1], contribution[:, :, t0:t1], atol=1e-6)


class _TimeVaryingVol:
    """Deterministically time-varying volatility (see test_hd_propagation)."""

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
        L = self._constant.cholesky_at(posterior, t=None)
        scales = np.array([self._scale(t) for t in range(T)])
        return L[:, :, np.newaxis, :, :] * scales[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

    def forecast_cholesky_path(self, posterior, steps, rng):  # pragma: no cover
        raise NotImplementedError


class TestPerTStochasticVolatility:
    def test_duality_holds_under_time_varying_p(self, synthetic_idata_2v, var_data_2v):
        """Full-sample duality survives genuinely per-t shock matrices."""
        T_eff = var_data_2v.endog.shape[0] - 1
        identified = IdentifiedVAR.model_construct(
            idata=synthetic_idata_2v,
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=_TimeVaryingVol(T_eff),
            scheme=Cholesky(ordering=["y1", "y2"]),
        )
        cf = identified.counterfactual(shocks=[ShockPath(shock="y1", values=0.0)])
        hd = identified.historical_decomposition()
        diff = var_data_2v.endog[1:][None, None] - cf.idata.posterior_predictive["counterfactual"].values
        contribution = hd.idata.posterior_predictive["hd"].sel(shock="y1").values
        np.testing.assert_allclose(diff, contribution, atol=1e-8)


class TestGuards:
    def test_unknown_shock_errors(self, identified_2v):
        with pytest.raises(ValueError, match="Unknown shock"):
            identified_2v.counterfactual(shocks=[ShockPath(shock="oil", values=0.0)])

    def test_unidentified_shock_errors(self):
        eps = np.zeros((1, 1, 10, 2))
        index = pd.date_range("2000-01-01", periods=10, freq="QS")
        with pytest.raises(ValueError, match="rotation-arbitrary"):
            apply_shock_edits(
                eps,
                [ShockPath(shock="unidentified_0", values=0.0)],
                index,
                ["target", "unidentified_0"],
                object(),
            )

    def test_empty_window_errors(self, identified_2v, var_data_2v):
        beyond = var_data_2v.index[-1] + pd.offsets.QuarterBegin(4)
        with pytest.raises(ValueError, match="zero periods"):
            identified_2v.counterfactual(shocks=[ShockPath(shock="y1", values=0.0, start=beyond)])

    def test_wrong_length_array_errors(self, identified_2v, var_data_2v):
        idx = var_data_2v.index[1:]
        with pytest.raises(ValueError, match="length"):
            identified_2v.counterfactual(shocks=[ShockPath(shock="y1", values=np.zeros(5), start=idx[10], end=idx[29])])

    def test_nan_values_error(self, identified_2v, var_data_2v):
        idx = var_data_2v.index[1:]
        with pytest.raises(ValueError, match="NaN"):
            identified_2v.counterfactual(
                shocks=[ShockPath(shock="y1", values=np.array([0.0, np.nan, 0.0]), start=idx[10], end=idx[12])]
            )

    def test_overlapping_edits_error(self, identified_2v, var_data_2v):
        idx = var_data_2v.index[1:]
        with pytest.raises(ValueError, match="overlapping"):
            identified_2v.counterfactual(
                shocks=[
                    ShockPath(shock="y1", values=0.0, start=idx[10], end=idx[30]),
                    ShockPath(shock="y1", values=1.0, start=idx[25], end=idx[40]),
                ]
            )

    def test_lag_trim_clamp_warns(self, identified_2v, var_data_2v):
        start = var_data_2v.index[0]  # precedes the lag-trimmed index
        with pytest.warns(UserWarning, match="clamps forward"):
            identified_2v.counterfactual(
                shocks=[ShockPath(shock="y1", values=0.0, start=start, end=var_data_2v.index[10])]
            )

    def test_unit_effect_scale_warns_on_nonzero_edits(self):
        class _ScaledScheme:
            scale = 10.0

        eps = np.zeros((1, 1, 10, 2))
        index = pd.date_range("2000-01-01", periods=10, freq="QS")
        with pytest.warns(UserWarning, match="one-standard-deviation"):
            apply_shock_edits(eps, [ShockPath(shock="a", values=2.0)], index, ["a", "b"], _ScaledScheme())

    def test_unit_effect_scale_silent_on_zero_edits(self):
        import warnings as _warnings

        class _ScaledScheme:
            scale = 10.0

        eps = np.zeros((1, 1, 10, 2))
        index = pd.date_range("2000-01-01", periods=10, freq="QS")
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            apply_shock_edits(eps, [ShockPath(shock="a", values=0.0)], index, ["a", "b"], _ScaledScheme())


class TestDisplaySlice:
    def test_start_end_slice_matches_full(self, identified_2v, var_data_2v):
        idx = var_data_2v.index[1:]
        edit = [ShockPath(shock="y1", values=0.0)]
        full = identified_2v.counterfactual(shocks=edit).idata.posterior_predictive
        sub = identified_2v.counterfactual(shocks=edit, start=idx[50], end=idx[120]).idata.posterior_predictive
        np.testing.assert_array_equal(sub["counterfactual"].values, full["counterfactual"].values[:, :, 50:121])
        np.testing.assert_array_equal(sub["actual"].values, full["actual"].values[50:121])


class TestResultSurface:
    def test_median_shape_and_labels(self, identified_2v, var_data_2v):
        frame = identified_2v.counterfactual(shocks=[]).median()
        assert list(frame.columns) == ["y1", "y2"]
        assert len(frame) == var_data_2v.endog.shape[0] - 1
        assert frame.index.name == "time"

    def test_actual_matches_data(self, identified_2v, var_data_2v):
        frame = identified_2v.counterfactual(shocks=[]).actual()
        np.testing.assert_allclose(frame.values, var_data_2v.endog[1:])

    def test_difference_matches_median_hd_contribution(self, identified_2v):
        """difference() of a full-sample zero-edit equals the median HD contribution."""
        cf = identified_2v.counterfactual(shocks=[ShockPath(shock="y1", values=0.0)])
        hd = identified_2v.historical_decomposition()
        expected = hd.idata.posterior_predictive["hd"].sel(shock="y1").median(dim=("chain", "draw"))
        np.testing.assert_allclose(cf.difference().values, expected.values.reshape(cf.difference().shape), atol=1e-8)

    def test_hdi_mirrors_median_layout(self, identified_2v):
        result = identified_2v.counterfactual(shocks=[])
        hdi = result.hdi(prob=0.9)
        assert hdi.lower.shape == result.median().shape
        assert hdi.prob == 0.9
        assert (hdi.upper.values >= hdi.lower.values).all()


class TestPlot:
    def test_returns_figure_with_panels_and_labels(self, identified_2v):
        result = identified_2v.counterfactual(shocks=[ShockPath(shock="y1", values=0.0)])
        fig = result.plot()
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert labels == ["actual", "counterfactual (median)", "89% HDI"]
        assert fig._suptitle.get_text() == "Historical Counterfactual"
