"""Tests for ZeroSignRestriction (Arias, Rubio-Ramirez & Waggoner, 2018)."""

import warnings

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from impulso._arviz_compat import make_idata
from impulso.identification import SignRestriction, ZeroSignRestriction
from impulso.protocols import IdentificationScheme

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def synthetic_idata_3v():
    """Synthetic InferenceData mimicking a fitted 3-var VAR(1).

    Same recipe as the shared `synthetic_idata_2v` fixture, one variable
    wider: 2 chains, 30 draws, 3 variables, 1 lag.
    """
    rng = np.random.default_rng(7)
    n_chains, n_draws, n_vars, n_lags = 2, 30, 3, 1

    B = rng.standard_normal((n_chains, n_draws, n_vars, n_vars * n_lags)) * 0.2
    intercept = rng.standard_normal((n_chains, n_draws, n_vars)) * 0.01

    sigma = np.zeros((n_chains, n_draws, n_vars, n_vars))
    L = np.zeros_like(sigma)
    for c in range(n_chains):
        for d in range(n_draws):
            A = rng.standard_normal((n_vars, n_vars)) * 0.5
            sigma[c, d] = A @ A.T + np.eye(n_vars)
            L[c, d] = np.linalg.cholesky(sigma[c, d])

    posterior = xr.Dataset({
        "B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"]),
        "intercept": xr.DataArray(intercept, dims=["chain", "draw", "var"]),
        "Sigma": xr.DataArray(
            sigma,
            dims=["chain", "draw", "var1", "var2"],
            coords={"var1": ["y1", "y2", "y3"], "var2": ["y1", "y2", "y3"]},
        ),
        "L": xr.DataArray(L, dims=["chain", "draw", "var1", "var2"]),
    })
    return make_idata(posterior=posterior)


def _quiet(**kwargs) -> ZeroSignRestriction:
    """Build a scheme, suppressing the unsigned-shock UserWarning.

    Most numerics tests deliberately leave column signs unpinned; the
    warning is asserted on directly in `TestValidation`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ZeroSignRestriction(**kwargs)


# --------------------------------------------------------------------------
# Validation (n-independent, at construction)
# --------------------------------------------------------------------------


class TestValidation:
    def test_frozen(self):
        scheme = _quiet(shock_names=["s1", "s2"], zero_restrictions={"y1": ["s2"]})
        with pytest.raises(ValidationError):
            scheme.n_rotations = 5

    def test_satisfies_protocol(self):
        scheme = _quiet(shock_names=["s1", "s2"], zero_restrictions={"y1": ["s2"]})
        assert isinstance(scheme, IdentificationScheme)

    def test_samples_rotations_flag(self):
        scheme = _quiet(shock_names=["s1", "s2"], zero_restrictions={"y1": ["s2"]})
        assert scheme._samples_rotations is True

    def test_empty_shock_names(self):
        with pytest.raises(ValidationError, match="at least one structural shock"):
            ZeroSignRestriction(shock_names=[], zero_restrictions={"y1": ["s2"]})

    def test_duplicate_shock_names(self):
        with pytest.raises(ValidationError, match="Duplicate shock names"):
            ZeroSignRestriction(shock_names=["s1", "s1"], zero_restrictions={"y1": ["s1"]})

    def test_reserved_prefix(self):
        with pytest.raises(ValidationError, match="reserved prefix"):
            ZeroSignRestriction(shock_names=["s1", "unidentified_1"], zero_restrictions={"y1": ["s1"]})

    def test_unknown_shock_in_zero_restrictions(self):
        with pytest.raises(ValidationError, match="unknown shock"):
            ZeroSignRestriction(shock_names=["s1"], zero_restrictions={"y1": ["nope"]})

    def test_unknown_shock_in_sign_restrictions(self):
        with pytest.raises(ValidationError, match="unknown shock"):
            ZeroSignRestriction(shock_names=["s1"], sign_restrictions={"y1": {"nope": "+"}})

    def test_conflicting_zero_and_sign_cell(self):
        with pytest.raises(ValidationError, match="conflict"):
            ZeroSignRestriction(
                shock_names=["s1", "s2"],
                zero_restrictions={"y1": ["s2"]},
                sign_restrictions={"y1": {"s2": "+"}},
            )

    def test_bad_sign_token(self):
        with pytest.raises(ValidationError, match="non-sign token"):
            ZeroSignRestriction(shock_names=["s1"], sign_restrictions={"y1": {"s1": "positive"}})

    def test_both_dicts_empty(self):
        with pytest.raises(ValidationError, match="at least one of zero_restrictions"):
            ZeroSignRestriction(shock_names=["s1"])

    def test_named_shock_without_sign_warns(self):
        with pytest.warns(UserWarning, match="no sign restriction"):
            ZeroSignRestriction(shock_names=["s1", "s2"], zero_restrictions={"y1": ["s2"]})

    def test_fully_signed_shocks_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ZeroSignRestriction(
                shock_names=["s1", "s2"],
                zero_restrictions={"y1": ["s2"]},
                sign_restrictions={"y2": {"s1": "+", "s2": "+"}},
            )


# --------------------------------------------------------------------------
# identify() entry checks (need n_vars)
# --------------------------------------------------------------------------


def _chol(n_vars: int, n_draws: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = np.zeros((1, n_draws, n_vars, n_vars))
    for d in range(n_draws):
        A = rng.standard_normal((n_vars, n_vars))
        sigma[0, d] = A @ A.T + np.eye(n_vars)
    return np.linalg.cholesky(sigma)


class TestIdentifyEntryChecks:
    def test_rank_condition_n2_two_zeros(self):
        """n = 2, one shock with 2 zeros: z_1 = 2 > n - 1 = 1."""
        scheme = _quiet(shock_names=["s1", "s2"], zero_restrictions={"y1": ["s2"], "y2": ["s2"]})
        with pytest.raises(ValueError, match="at most"):
            scheme.identify(_chol(2), ["y1", "y2"])

    def test_rank_condition_n3_two_shocks_two_zeros(self):
        """n = 3 with z = (2, 2): the second sorted position needs z <= 1."""
        scheme = _quiet(
            shock_names=["s1", "s2", "s3"],
            zero_restrictions={"y1": ["s2", "s3"], "y2": ["s2", "s3"]},
        )
        with pytest.raises(ValueError, match=r"n - j"):
            scheme.identify(_chol(3), ["y1", "y2", "y3"])

    def test_unknown_variable(self):
        scheme = _quiet(shock_names=["s1", "s2"], zero_restrictions={"nope": ["s2"]})
        with pytest.raises(ValueError, match="unknown variable"):
            scheme.identify(_chol(2), ["y1", "y2"])

    def test_too_many_shocks(self):
        scheme = _quiet(shock_names=["s1", "s2", "s3"], zero_restrictions={"y1": ["s2"]})
        with pytest.raises(ValueError, match="at most n_vars structural shocks"):
            scheme.identify(_chol(2), ["y1", "y2"])

    def test_horizon_without_posterior(self):
        scheme = _quiet(
            shock_names=["s1", "s2"],
            zero_restrictions={"y1": ["s2"]},
            sign_restrictions={"y2": {"s1": "+", "s2": "+"}},
            restriction_horizon=2,
        )
        with pytest.raises(ValueError, match="requires the full posterior"):
            scheme.identify(_chol(2), ["y1", "y2"])


# --------------------------------------------------------------------------
# Numerics — the Cholesky-degeneracy anchor is the go/no-go gate
# --------------------------------------------------------------------------


class TestCholeskyDegeneracyAnchor:
    """Full triangular zeros collapse every null space to one dimension.

    With z_j = n - j for every shock the construction has no freedom left:
    each column is determined up to its sign, P is lower triangular, and
    P P' = Sigma. The only lower-triangular square root of Sigma is the
    Cholesky factor up to column signs, so |P| must reproduce
    |cholesky(Sigma)| exactly and acceptance must be 1.0.
    """

    def test_matches_cholesky_in_absolute_value(self, synthetic_idata_3v):
        L = synthetic_idata_3v.posterior["L"].values
        sigma = synthetic_idata_3v.posterior["Sigma"].values
        scheme = _quiet(
            shock_names=["s1", "s2", "s3"],
            zero_restrictions={"y1": ["s2", "s3"], "y2": ["s3"]},
            n_rotations=1,
            random_seed=0,
        )
        P = scheme.identify(L, ["y1", "y2", "y3"])

        assert scheme._last_diagnostics["zero_sign_acceptance_rate"] == 1.0
        expected = np.abs(np.linalg.cholesky(sigma))
        max_dev = float(np.max(np.abs(np.abs(P) - expected)))
        assert max_dev < 1e-8, f"max deviation from |cholesky(Sigma)| = {max_dev:g}"

    def test_diagonal_signs_pin_the_cholesky_factor_exactly(self, synthetic_idata_3v):
        L = synthetic_idata_3v.posterior["L"].values
        sigma = synthetic_idata_3v.posterior["Sigma"].values
        scheme = ZeroSignRestriction(
            shock_names=["s1", "s2", "s3"],
            zero_restrictions={"y1": ["s2", "s3"], "y2": ["s3"]},
            sign_restrictions={"y1": {"s1": "+"}, "y2": {"s2": "+"}, "y3": {"s3": "+"}},
            # Each column's sign is an independent coin flip in the degenerate
            # (1-dimensional null space) case, so a candidate satisfies all
            # three diagonal pins with probability 1/8. The budget has to
            # cover that: P(no acceptance in 300 tries) is around 1e-17.
            n_rotations=300,
            random_seed=0,
        )
        P = scheme.identify(L, ["y1", "y2", "y3"])

        assert scheme._last_diagnostics["zero_sign_acceptance_rate"] == 1.0
        np.testing.assert_allclose(P, np.linalg.cholesky(sigma), atol=1e-8)


class TestClosedFormAndInvariants:
    def test_n2_analytic_column(self, synthetic_idata_2v):
        """One zero on a 2-var VAR identifies that column up to sign."""
        L = synthetic_idata_2v.posterior["L"].values
        sigma = synthetic_idata_2v.posterior["Sigma"].values
        scheme = _quiet(
            shock_names=["s1", "s2"],
            zero_restrictions={"y1": ["s2"]},
            n_rotations=1,
            random_seed=3,
        )
        P = scheme.identify(L, ["y1", "y2"])

        s11 = sigma[..., 0, 0]
        s12 = sigma[..., 0, 1]
        s22 = sigma[..., 1, 1]
        expected_lower = np.sqrt(s22 - s12**2 / s11)
        np.testing.assert_allclose(np.abs(P[..., 0, 1]), 0.0, atol=1e-8)
        np.testing.assert_allclose(np.abs(P[..., 1, 1]), expected_lower, atol=1e-8)

    def test_zero_cells_are_exact(self, synthetic_idata_3v):
        """A generic single zero restriction is satisfied to SVD precision."""
        L = synthetic_idata_3v.posterior["L"].values
        scheme = _quiet(
            shock_names=["s1", "s2"],
            zero_restrictions={"y2": ["s1"]},
            n_rotations=50,
            random_seed=11,
        )
        P = scheme.identify(L, ["y1", "y2", "y3"])
        assert np.nanmax(np.abs(P[..., 1, 0])) < 1e-10
        assert scheme._last_diagnostics["zero_sign_max_zero_violation"] < 1e-10

    def test_reproduces_sigma(self, synthetic_idata_3v):
        """P P' = Sigma on every accepted draw — Q is exactly orthogonal."""
        L = synthetic_idata_3v.posterior["L"].values
        sigma = synthetic_idata_3v.posterior["Sigma"].values
        scheme = _quiet(
            shock_names=["s1", "s2"],
            zero_restrictions={"y2": ["s1"]},
            sign_restrictions={"y1": {"s1": "+"}, "y3": {"s2": "-"}},
            n_rotations=200,
            random_seed=5,
        )
        P = scheme.identify(L, ["y1", "y2", "y3"])
        finite = np.isfinite(P).all(axis=(-2, -1))
        assert finite.any()
        np.testing.assert_allclose(P[finite] @ np.swapaxes(P[finite], -1, -2), sigma[finite], atol=1e-10)

    def test_columns_land_in_user_order(self, synthetic_idata_3v):
        """Internal sorting is invisible: zeros appear under the named shock."""
        L = synthetic_idata_3v.posterior["L"].values
        # s3 is the most-restricted shock, so construction reorders — but the
        # zero cells must still be at (y1, s3), (y2, s3) and (y1, s2).
        scheme = _quiet(
            shock_names=["s1", "s2", "s3"],
            zero_restrictions={"y1": ["s2", "s3"], "y2": ["s3"]},
            n_rotations=1,
            random_seed=0,
        )
        P = scheme.identify(L, ["y1", "y2", "y3"])
        assert np.max(np.abs(P[..., 0, 2])) < 1e-10
        assert np.max(np.abs(P[..., 1, 2])) < 1e-10
        assert np.max(np.abs(P[..., 0, 1])) < 1e-10
        assert np.min(np.abs(P[..., 0, 0])) > 1e-8


# --------------------------------------------------------------------------
# Horizon sign restrictions
# --------------------------------------------------------------------------


class TestHorizonSigns:
    def test_signs_hold_at_every_horizon(self, synthetic_idata_3v):
        from impulso._linalg import lag_matrices
        from impulso._ma import compute_ma_phi

        horizon = 3
        L = synthetic_idata_3v.posterior["L"].values
        B = synthetic_idata_3v.posterior["B"].values
        scheme = _quiet(
            shock_names=["s1", "s2"],
            zero_restrictions={"y2": ["s1"]},
            sign_restrictions={"y1": {"s1": "+"}, "y3": {"s2": "-"}},
            restriction_horizon=horizon,
            n_rotations=500,
            random_seed=17,
        )
        # Random synthetic B makes the horizon signs hard, so a fair share of
        # draws end as NaN. That is the documented policy, not a failure —
        # the loop below only inspects the accepted draws.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            P = scheme.identify(L, ["y1", "y2", "y3"], posterior=synthetic_idata_3v.posterior)

        n_checked = 0
        for c in range(P.shape[0]):
            for d in range(P.shape[1]):
                if not np.isfinite(P[c, d]).all():
                    continue
                n_checked += 1
                Phi = compute_ma_phi(lag_matrices(B[c, d], 1), horizon)
                for h in range(horizon + 1):
                    irf = Phi[h] @ P[c, d]
                    assert irf[0, 0] >= -1e-12
                    assert irf[2, 1] <= 1e-12
        assert n_checked > 0

    def test_zero_restriction_is_impact_only(self, synthetic_idata_3v):
        """Zeros bind at h = 0; later horizons are free."""
        L = synthetic_idata_3v.posterior["L"].values
        scheme = _quiet(
            shock_names=["s1", "s2"],
            zero_restrictions={"y2": ["s1"]},
            sign_restrictions={"y1": {"s1": "+"}},
            restriction_horizon=2,
            n_rotations=500,
            random_seed=23,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            P = scheme.identify(L, ["y1", "y2", "y3"], posterior=synthetic_idata_3v.posterior)
        assert np.nanmax(np.abs(P[..., 1, 0])) < 1e-10


# --------------------------------------------------------------------------
# Distributional equivalence with SignRestriction when there are no zeros
# --------------------------------------------------------------------------


class TestSignOnlyEquivalence:
    def test_matches_sign_restriction_without_zeros(self, synthetic_idata_2v):
        """No zeros => the recursion is Gram-Schmidt on Gaussians, i.e. Haar.

        `SignRestriction` draws from SO(n) via `special_ortho_group`; this
        scheme draws from O(n). With one named shock the second column is
        unidentified, so the two accepted-P laws coincide (a reflection only
        flips the unpinned column). Tolerances are coarse on purpose: this
        is a two-sample Monte Carlo comparison on 2 x 50 draws, so the
        sampling error on an acceptance rate near 0.5 is order 0.05 and the
        means of the restricted cells are noisier still.
        """
        L = synthetic_idata_2v.posterior["L"].values
        zs = ZeroSignRestriction(
            shock_names=["s"],
            sign_restrictions={"y1": {"s": "+"}, "y2": {"s": "-"}},
            n_rotations=200,
            random_seed=101,
        )
        sr = SignRestriction(
            restrictions={"y1": {"s": "+"}, "y2": {"s": "-"}},
            n_rotations=200,
            random_seed=101,
        )
        P_zs = zs.identify(L, ["y1", "y2"])
        P_sr = sr.identify(L, ["y1", "y2"])

        rate_zs = zs._last_diagnostics["zero_sign_acceptance_rate"]
        rate_sr = sr._last_acceptance_rate
        assert abs(rate_zs - rate_sr) < 0.15

        for row in (0, 1):
            m_zs = float(np.nanmean(P_zs[..., row, 0]))
            m_sr = float(np.nanmean(P_sr[..., row, 0]))
            assert abs(m_zs - m_sr) < 0.15, f"row {row}: {m_zs:g} vs {m_sr:g}"


# --------------------------------------------------------------------------
# Failure policy and diagnostics
# --------------------------------------------------------------------------


class TestFailurePolicy:
    def _impossible(self, **kwargs) -> ZeroSignRestriction:
        # y1 and y2 must both rise on impact from s1 while s1 is also
        # restricted to zero on y3 — feasible in principle, but with
        # n_rotations=1 the single candidate almost never complies.
        return ZeroSignRestriction(
            shock_names=["s1", "s2"],
            zero_restrictions={"y3": ["s1"]},
            sign_restrictions={"y1": {"s1": "+", "s2": "+"}, "y2": {"s1": "+", "s2": "-"}},
            n_rotations=1,
            random_seed=1,
            **kwargs,
        )

    def test_nan_and_summary_warning(self, synthetic_idata_3v):
        L = synthetic_idata_3v.posterior["L"].values
        scheme = self._impossible()
        with pytest.warns(UserWarning, match="NaN"):
            P = scheme.identify(L, ["y1", "y2", "y3"])
        assert np.isnan(P).any()
        assert scheme._last_diagnostics["zero_sign_failed_draws"] > 0
        assert 0.0 <= scheme._last_diagnostics["zero_sign_acceptance_rate"] < 1.0

    def test_raise_policy(self, synthetic_idata_3v):
        L = synthetic_idata_3v.posterior["L"].values
        scheme = self._impossible(on_failure="raise")
        with pytest.raises(ValueError, match="No admissible rotation"):
            scheme.identify(L, ["y1", "y2", "y3"])

    def test_no_fallback_to_cholesky(self, synthetic_idata_3v):
        """Failed draws are NaN, never silently the unrotated factor."""
        L = synthetic_idata_3v.posterior["L"].values
        scheme = self._impossible()
        with pytest.warns(UserWarning):
            P = scheme.identify(L, ["y1", "y2", "y3"])
        failed = ~np.isfinite(P).all(axis=(-2, -1))
        assert failed.any()
        assert np.isnan(P[failed]).all()

    def test_diagnostics_keys(self, synthetic_idata_3v):
        L = synthetic_idata_3v.posterior["L"].values
        scheme = _quiet(
            shock_names=["s1", "s2"],
            zero_restrictions={"y2": ["s1"]},
            n_rotations=10,
            random_seed=2,
        )
        scheme.identify(L, ["y1", "y2", "y3"])
        assert set(scheme._last_diagnostics) == {
            "zero_sign_acceptance_rate",
            "zero_sign_failed_draws",
            "zero_sign_failed_fraction",
            "zero_sign_mean_attempts",
            "zero_sign_max_zero_violation",
        }
        assert scheme._last_diagnostics["zero_sign_mean_attempts"] >= 1.0

    def test_does_not_set_sign_restriction_acceptance_rate(self, synthetic_idata_3v):
        """The scheme must not attach the misnamed SignRestriction attr."""
        L = synthetic_idata_3v.posterior["L"].values
        scheme = _quiet(shock_names=["s1", "s2"], zero_restrictions={"y2": ["s1"]}, n_rotations=10)
        scheme.identify(L, ["y1", "y2", "y3"])
        assert getattr(scheme, "_last_acceptance_rate", None) is None


# --------------------------------------------------------------------------
# Pipeline integration
# --------------------------------------------------------------------------


class TestPipeline:
    def test_importable_from_impulso(self):
        import impulso

        assert impulso.ZeroSignRestriction is ZeroSignRestriction
        assert "ZeroSignRestriction" in impulso.__all__

    def test_shock_coords_padding(self):
        scheme = _quiet(shock_names=["s1"], zero_restrictions={"y2": ["s1"]})
        assert scheme.shock_coords(3) == ["s1", "unidentified_1", "unidentified_2"]

    def test_diagnostics_reach_shock_matrix_attrs(self, synthetic_idata_3v):
        import pandas as pd

        from impulso.data import VARData
        from impulso.identified import IdentifiedVAR
        from impulso.volatility import Constant

        rng = np.random.default_rng(0)
        index = pd.date_range("2000-01-01", periods=60, freq="QS")
        data = VARData(endog=rng.standard_normal((60, 3)), endog_names=["y1", "y2", "y3"], index=index)
        scheme = _quiet(
            shock_names=["s1"],
            zero_restrictions={"y2": ["s1"]},
            sign_restrictions={"y1": {"s1": "+"}},
            n_rotations=100,
            random_seed=4,
        )
        identified = IdentifiedVAR(
            idata=synthetic_idata_3v,
            n_lags=1,
            data=data,
            var_names=["y1", "y2", "y3"],
            volatility=Constant(),
            scheme=scheme,
        )
        sm = identified.shock_matrix()
        assert "zero_sign_acceptance_rate" in sm.attrs
        assert "sign_restriction_acceptance_rate" not in sm.attrs
        assert list(sm.coords["shock"].values) == ["s1", "unidentified_1", "unidentified_2"]

        with pytest.warns(UserWarning, match="rotation-arbitrary"):
            fevd = identified.fevd(horizon=4)
        da = fevd.idata.posterior_predictive["fevd"]
        assert np.isnan(da.sel(shock="unidentified_1").values).all()
        assert np.isnan(da.sel(shock="unidentified_2").values).all()
        assert not np.isnan(da.sel(shock="s1").values).any()
