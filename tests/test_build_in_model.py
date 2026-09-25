"""Tests for `VAR.build_in_model` (issues 08a, 08b, 09a, 09b).

`VAR._build_pymc_model` becomes a thin wrapper: it opens a fresh
`pymc.Model`, converts a `VARData` into arrays, and delegates to a new
public `VAR.build_in_model`, which registers a VAR into whichever PyMC
model is active on entry. Several kinds of test live here:

* Parity tests (`TestWrapperLogpParity`, `TestLagDesignMatrixUsage`) exercise
  only the pre-existing `_build_pymc_model` surface and must pass unchanged
  whether the graph is built inline or delegated to `build_in_model` — they
  are not `xfail`-marked.
* Tests of the new public method (`TestBuildInModel`) exercise
  `VAR.build_in_model` directly.
* `TestInterceptEquations` (issue 08b) exercises the `intercept_equations`
  argument: which equations get an intercept term, the `var_intercept`
  coord that a strict subset needs, and the two edge cases (`None`/every
  name given -> today's behaviour unchanged; every name excluded -> no
  intercept variable at all).
* `TestSymbolicEndog` (issue 09a) passes the observed endogenous block as a
  symbolic tensor (`pytensor.shared` or `pm.Data`). The likelihood becomes a
  `pm.Potential` over `ErrorDistribution.logp`, and `endog_scales` is
  required because `ar1_residual_sd` needs concrete data.
* `TestLatentSeries` (issue 09b) declares latent endogenous series via
  `latent_names`. `build_in_model` generates their paths non-centred, from
  standard-normal innovations, and returns them; the observed block's
  likelihood is conditional on those innovations.
"""

import numpy as np
import pandas as pd
import pytest

from impulso.data import VARData
from impulso.spec import VAR


def _make_data(
    rng: np.random.Generator,
    n_vars: int = 2,
    T: int = 60,
    exog_names: list[str] | None = None,
) -> VARData:
    """A small VARData instance, deterministic given `rng`."""
    endog = rng.standard_normal((T, n_vars))
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    endog_names = [f"y{i + 1}" for i in range(n_vars)]
    if exog_names:
        exog = rng.standard_normal((T, len(exog_names)))
        return VARData(endog=endog, endog_names=endog_names, exog=exog, exog_names=exog_names, index=index)
    return VARData(endog=endog, endog_names=endog_names, index=index)


def _model_logp(model, seed: int = 0) -> float:
    """Total joint log-probability of `model` at a fixed, reproducible point.

    `initial_point(random_seed=...)` is deterministic given the model's
    graph and the seed, so this is a fixed parameter point in the sense
    the acceptance criteria mean, not a random draw that happens to be
    seeded.
    """
    point = model.initial_point(random_seed=seed)
    return float(model.compile_logp()(point))


def _pinned_intercept_prior_logp(full_model, point, pinned: list[int]) -> float:
    """Prior log-density of the `pinned` elements of `full_model`'s intercept at `point`.

    A model that excludes an equation's intercept has no prior term for it,
    whereas the equivalent full model with that intercept pinned to `0.0`
    still carries the prior density of `0.0`. Subtracting this makes the
    two joint log-probabilities comparable.
    """
    (elementwise,) = full_model.compile_logp(vars=[full_model["intercept"]], sum=False)(point)
    return float(np.sum(np.asarray(elementwise)[pinned]))


class TestWrapperLogpParity:
    """Pins `_build_pymc_model`'s numeric output across the 08a refactor.

    The expected values were captured against the pre-refactor
    `_build_pymc_model` (which built the whole graph inline). If
    `build_in_model` changes what graph gets built — a different prior, a
    dropped node, a reordered coordinate — these numbers move and the test
    fails; a pure delegation leaves them untouched. Deliberately independent
    of `build_in_model`: this test exercises only the public `VAR.fit` /
    `_build_pymc_model` surface, so it is not marked `xfail` and must pass
    both before and after the refactor.
    """

    def test_gaussian_no_exog(self, rng):
        data = _make_data(rng)
        model, _ = VAR(lags=1)._build_pymc_model(data)
        assert _model_logp(model) == pytest.approx(-225.00961968371863)

    def test_gaussian_with_exog(self, rng):
        data = _make_data(rng, exog_names=["z"])
        model, _ = VAR(lags=1)._build_pymc_model(data)
        assert _model_logp(model) == pytest.approx(-235.5508133506178)

    def test_student_t_no_exog(self, rng):
        data = _make_data(rng)
        model, _ = VAR(lags=1, error_dist="student_t")._build_pymc_model(data)
        assert _model_logp(model) == pytest.approx(-226.43783005847177)

    def test_student_t_with_exog(self, rng):
        data = _make_data(rng, exog_names=["z"])
        model, _ = VAR(lags=1, error_dist="student_t")._build_pymc_model(data)
        assert _model_logp(model) == pytest.approx(-236.97902372537095)


class TestLagDesignMatrixUsage:
    """`_build_pymc_model` (and, once it exists, `build_in_model`) must use
    the shared `build_lag_design_matrix` from issue 02, not a private
    re-implementation of lag stacking. Not `xfail`-marked: the wrapper
    already routes through the shared builder on `main`.
    """

    def test_wrapper_calls_shared_lag_design_matrix_builder(self, rng, monkeypatch):
        import impulso.spec as spec_module

        data = _make_data(rng)
        calls = []
        original = spec_module.build_lag_design_matrix

        def _spy(endog, n_lags, exog=None):
            calls.append((endog, n_lags, exog))
            return original(endog, n_lags, exog)

        monkeypatch.setattr(spec_module, "build_lag_design_matrix", _spy)

        VAR(lags=1)._build_pymc_model(data)

        assert len(calls) == 1
        got_endog, got_n_lags, got_exog = calls[0]
        np.testing.assert_array_equal(got_endog, data.endog)
        assert got_n_lags == 1
        assert got_exog is None


class TestBuildInModel:
    """Direct tests of the new public `VAR.build_in_model` (issue 08a)."""

    def test_direct_call_matches_wrapper_logp(self, rng):
        """Calling `build_in_model` directly inside a fresh model gives the
        same log-probability as the wrapper (acceptance criterion 2)."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        wrapper_model, n_lags = spec._build_pymc_model(data)
        wrapper_logp = _model_logp(wrapper_model)

        with pm.Model(coords={"time": data.index[n_lags:]}) as direct_model:
            spec.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=n_lags,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
            )
        direct_logp = _model_logp(direct_model)

        assert direct_logp == pytest.approx(wrapper_logp)

    def test_direct_call_matches_wrapper_logp_with_exog_and_student_t(self, rng):
        import pymc as pm

        data = _make_data(rng, exog_names=["z"])
        spec = VAR(lags=1, error_dist="student_t")

        wrapper_model, n_lags = spec._build_pymc_model(data)
        wrapper_logp = _model_logp(wrapper_model)

        with pm.Model(coords={"time": data.index[n_lags:]}) as direct_model:
            spec.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=n_lags,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
            )
        direct_logp = _model_logp(direct_model)

        assert direct_logp == pytest.approx(wrapper_logp)

    def test_returns_handles_with_intercept_b_bexog_l_and_likelihood(self, rng):
        import pymc as pm

        data = _make_data(rng, exog_names=["z"])
        spec = VAR(lags=1)

        with pm.Model():
            handles = spec.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=1,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
            )

        assert handles.intercept is not None
        assert handles.B is not None
        assert handles.B_exog is not None
        assert handles.L is not None
        assert handles.obs is not None

    def test_returns_none_b_exog_when_no_exog(self, rng):
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model():
            handles = spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
            )

        assert handles.B_exog is None

    def test_registers_into_the_active_model_context(self, rng):
        """No prefix argument: `build_in_model` writes into `pymc.modelcontext(None)`."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model() as model:
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
            )

        assert "intercept" in model.named_vars
        assert "B" in model.named_vars
        assert "obs" in model.named_vars

    def test_uses_shared_lag_design_matrix_builder(self, rng, monkeypatch):
        import pymc as pm

        import impulso.spec as spec_module

        data = _make_data(rng)
        calls = []
        original = spec_module.build_lag_design_matrix

        def _spy(endog, n_lags, exog=None):
            calls.append((endog, n_lags, exog))
            return original(endog, n_lags, exog)

        monkeypatch.setattr(spec_module, "build_lag_design_matrix", _spy)

        with pm.Model():
            VAR(lags=1).build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
            )

        assert len(calls) == 1
        got_endog, got_n_lags, got_exog = calls[0]
        np.testing.assert_array_equal(got_endog, data.endog)
        assert got_n_lags == 1
        assert got_exog is None

    def test_nested_named_model_prefixes_variables_but_not_coords(self, rng):
        """Inside `pm.Model(name=...)`, every Impulso variable, deterministic
        and the likelihood come out prefixed; coords land unprefixed on the
        root model (acceptance criterion 3; PyMC does not prefix coords —
        see `prototype/REPORT.md`)."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model() as root, pm.Model(name="p"):
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
            )

        assert "p::intercept" in root.named_vars
        assert "p::B" in root.named_vars
        assert "p::Sigma" in root.named_vars
        assert "p::obs" in root.named_vars
        assert "intercept" not in root.named_vars
        assert "var" in root.coords
        assert list(root.coords["var"]) == data.endog_names

    def test_unnested_names_are_unchanged(self, rng):
        """Outside a nested named model, names carry no prefix."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model() as model:
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
            )

        assert "intercept" in model.named_vars
        assert "B" in model.named_vars
        assert "obs" in model.named_vars

    def test_time_coord_length_mismatch_raises(self, rng):
        """Two VARs embedded in the same root model must not silently share
        a stale `time` coordinate when their likelihoods have a different
        number of rows (review round 1). Coords are not prefixed by a
        nested `pm.Model(name=...)`, so the second `build_in_model` call
        would otherwise reuse the first call's `time` coordinate at the
        wrong length, giving the second likelihood a wrong shape that only
        surfaces much later (e.g. in `sample_prior_predictive`)."""
        import pymc as pm

        data_a = _make_data(rng, T=41)  # n_lags=1 -> 40 likelihood rows
        data_b = _make_data(rng, T=31)  # n_lags=1 -> 30 likelihood rows
        spec = VAR(lags=1)

        with pm.Model() as root:
            with pm.Model(name="a"):
                spec.build_in_model(
                    endog=data_a.endog,
                    exog=None,
                    n_lags=1,
                    endog_names=data_a.endog_names,
                )
            with pytest.raises(ValueError, match="time"), pm.Model(name="b"):
                spec.build_in_model(
                    endog=data_b.endog,
                    exog=None,
                    n_lags=1,
                    endog_names=data_b.endog_names,
                )

        # The first VAR's own registration is untouched by the second call's failure.
        assert "a::obs" in root.named_vars

    def test_time_coord_matching_length_is_fine(self, rng):
        """Equal-length `time` coords — including the wrapper's real dates —
        are not a collision; only a length mismatch is rejected."""
        import pymc as pm

        data_a = _make_data(rng, T=41)
        data_b = _make_data(rng, T=41)  # same T, different values -- fine
        spec = VAR(lags=1)

        with pm.Model() as root:
            with pm.Model(name="a"):
                spec.build_in_model(
                    endog=data_a.endog,
                    exog=None,
                    n_lags=1,
                    endog_names=data_a.endog_names,
                )
            with pm.Model(name="b"):
                spec.build_in_model(
                    endog=data_b.endog,
                    exog=None,
                    n_lags=1,
                    endog_names=data_b.endog_names,
                )

        assert "a::obs" in root.named_vars
        assert "b::obs" in root.named_vars

    def test_endog_scales_overrides_the_default_ar1_residual_sd(self, rng):
        """`endog_scales=None` computes sigma from the data; a caller-supplied
        array is used as-is instead (issue 08a's `endog_scales` argument)."""
        import pymc as pm

        from impulso.spec import _exog_prior_sigma

        data = _make_data(rng, exog_names=["z"])
        spec = VAR(lags=1)
        custom_scales = np.array([2.5, 7.0])

        with pm.Model():
            handles_default = spec.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=1,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
            )
        with pm.Model():
            handles_custom = spec.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=1,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
                endog_scales=custom_scales,
            )

        default_sigma = handles_default.B_exog.owner.op.dist_params(handles_default.B_exog.owner)[1].eval()
        custom_sigma = handles_custom.B_exog.owner.op.dist_params(handles_custom.B_exog.owner)[1].eval()

        expected_custom = _exog_prior_sigma(custom_scales, data.exog[1:], spec.exog_prior_scale)
        np.testing.assert_allclose(custom_sigma, expected_custom)
        assert not np.allclose(default_sigma, custom_sigma)

    def test_endog_scales_feeds_the_minnesota_cross_lag_prior(self, rng):
        """`endog_scales` is the same sigma `Prior.build_priors` scales the
        Minnesota cross-lag entries by (docs/adr/0015), not only the exog prior."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)
        custom_scales = np.array([2.5, 7.0])

        with pm.Model():
            handles = spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
                endog_scales=custom_scales,
            )

        got = handles.B.owner.op.dist_params(handles.B.owner)[1].eval()
        expected = spec.resolved_prior.build_priors(n_vars=2, n_lags=1, sigma=custom_scales)["B_sigma"]
        np.testing.assert_allclose(got, expected)

    def test_endog_scales_are_validated(self, rng):
        """A caller-supplied scale gets the same zero/non-finite guard as the
        data-derived one (issue 07b), naming the offending column."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model(), pytest.raises(ValueError, match="y2"):
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
                endog_scales=np.array([1.0, 0.0]),
            )

    def test_endog_scales_validation_error_names_endog_scales_not_ar1_residual_sd(self, rng):
        """A bad caller-supplied `endog_scales` gets a message naming the actual
        source; it must not blame `ar1_residual_sd`, which never ran (issue 08c)."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model(), pytest.raises(ValueError) as exc_info:
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
                endog_scales=np.array([1.0, 0.0]),
            )
        message = str(exc_info.value)
        assert "endog_scales" in message
        assert "ar1_residual_sd" not in message

    def test_endog_scales_as_a_list_of_the_right_length_is_accepted(self, rng):
        """A plain Python list, not just an ndarray, is coerced and accepted
        when its length matches `n_vars` (issue 08c)."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model():
            handles = spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
                endog_scales=[2.5, 7.0],
            )
        assert handles.B is not None

    def test_endog_scales_wrong_length_raises_a_clear_shape_error(self, rng):
        """`endog_scales` longer than `n_vars` raises a `ValueError` naming
        `endog_scales` and its shape, not an opaque `TypeError` from a raw
        list hitting numpy comparisons downstream (issue 08c)."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model(), pytest.raises(ValueError) as exc_info:
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
                endog_scales=[1.0, 2.0, 3.0],
            )
        message = str(exc_info.value)
        assert "endog_scales" in message
        assert "shape" in message


class TestInterceptEquations:
    """`build_in_model(..., intercept_equations=...)` (issue 08b).

    `intercept_equations=None` (the default) means every equation gets an
    intercept — today's behaviour, unchanged. An explicit list restricts
    the intercept's free variable to that subset; equations left out get a
    literal zero in `mu` instead of a term. Naming a subset that is not,
    in fact, every equation (regardless of what order it lists them in)
    needs its own coord (`var_intercept`) because the free variable is
    then shorter than `n_vars`; naming every equation collapses back to
    today's `dims="var"` intercept so existing posteriors and `FittedVAR`
    keep reading it unchanged.
    """

    def test_unknown_name_raises(self, rng):
        import pymc as pm

        data = _make_data(rng)  # endog_names = ["y1", "y2"]
        spec = VAR(lags=1)

        with pm.Model(), pytest.raises(ValueError, match="bogus"):
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
                intercept_equations=["y1", "bogus"],
            )

    def test_duplicate_name_raises(self, rng):
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model(), pytest.raises(ValueError, match="y1"):
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
                intercept_equations=["y1", "y1"],
            )

    def test_partial_subset_uses_var_intercept_coord_and_dims(self, rng):
        """A strict subset gets its own `var_intercept` coord, ordered like
        `endog_names` (canonical order), not like the caller's list."""
        import pymc as pm

        data = _make_data(rng, n_vars=3)  # y1, y2, y3
        spec = VAR(lags=1)

        with pm.Model() as model:
            handles = spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
                intercept_equations=["y3", "y1"],
            )

        assert "var_intercept" in model.coords
        assert list(model.coords["var_intercept"]) == ["y1", "y3"]
        assert model.named_vars_to_dims["intercept"] == ("var_intercept",)
        assert handles.intercept is not None

    def test_explicit_full_list_keeps_dims_var_no_new_coord(self, rng):
        """Naming every equation — in any order — collapses to today's
        `dims="var"` intercept; no `var_intercept` coord is registered, and
        the log-probability matches the default (`None`) build exactly."""
        import pymc as pm

        data = _make_data(rng, n_vars=3)
        spec = VAR(lags=1)

        with pm.Model() as default_model:
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
            )
        with pm.Model() as explicit_model:
            handles = spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
                intercept_equations=list(reversed(data.endog_names)),
            )

        assert "var_intercept" not in explicit_model.coords
        assert explicit_model.named_vars_to_dims["intercept"] == ("var",)
        assert handles.intercept is not None

        point = default_model.initial_point(random_seed=7)
        default_logp = float(default_model.compile_logp()(point))
        explicit_logp = float(explicit_model.compile_logp()(point))
        assert explicit_logp == pytest.approx(default_logp)

    def test_excluded_equation_gets_a_literal_zero_not_a_missing_term(self, rng):
        """Building with only `y1` intercepted must give the same
        log-probability as building with both intercepted and `y2`'s pinned
        to exactly `0.0` — i.e. the excluded equation's `mu` really does add
        a literal zero rather than just omitting the term some other way.
        The full model's joint log-probability also carries `y2`'s intercept
        prior density at `0.0`, which the partial model has no term for, so
        that is subtracted before comparing."""
        import pymc as pm

        data = _make_data(rng)  # y1, y2
        spec = VAR(lags=1)

        with pm.Model() as partial_model:
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
                intercept_equations=["y1"],
            )
        with pm.Model() as full_model:
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
            )

        partial_point = partial_model.initial_point(random_seed=5)
        full_point = dict(partial_point)
        full_point["intercept"] = np.array([partial_point["intercept"][0], 0.0])

        partial_logp = float(partial_model.compile_logp()(partial_point))
        full_logp = float(full_model.compile_logp()(full_point))
        pinned_prior = _pinned_intercept_prior_logp(full_model, full_point, pinned=[1])
        assert partial_logp == pytest.approx(full_logp - pinned_prior)

    def test_excluded_equation_with_exog_gets_a_literal_zero(self, rng):
        """Same as above, but with an exogenous block present too, so the
        `B_exog` branch of `mu`'s construction is covered."""
        import pymc as pm

        data = _make_data(rng, exog_names=["z"])  # y1, y2
        spec = VAR(lags=1)

        with pm.Model() as partial_model:
            spec.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=1,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
                intercept_equations=["y1"],
            )
        with pm.Model() as full_model:
            spec.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=1,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
            )

        partial_point = partial_model.initial_point(random_seed=5)
        full_point = dict(partial_point)
        full_point["intercept"] = np.array([partial_point["intercept"][0], 0.0])

        partial_logp = float(partial_model.compile_logp()(partial_point))
        full_logp = float(full_model.compile_logp()(full_point))
        pinned_prior = _pinned_intercept_prior_logp(full_model, full_point, pinned=[1])
        assert partial_logp == pytest.approx(full_logp - pinned_prior)

    def test_all_excluded_gives_none_intercept_and_no_coord(self, rng):
        """Excluding every equation is allowed: no intercept variable is
        registered at all, and `handles.intercept` is `None`."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model() as model:
            handles = spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
                intercept_equations=[],
            )

        assert handles.intercept is None
        assert "intercept" not in model.named_vars
        assert "var_intercept" not in model.coords

    def test_all_excluded_matches_full_model_with_intercept_pinned_to_zero(self, rng):
        """No free intercept at all must give the same log-probability as
        the default build with every intercept pinned to `0.0`, once that
        build's intercept prior density at `0.0` is subtracted."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model() as excluded_model:
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
                intercept_equations=[],
            )
        with pm.Model() as full_model:
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
            )

        excluded_point = excluded_model.initial_point(random_seed=3)
        full_point = dict(excluded_point)
        full_point["intercept"] = np.zeros(2)

        excluded_logp = float(excluded_model.compile_logp()(excluded_point))
        full_logp = float(full_model.compile_logp()(full_point))
        pinned_prior = _pinned_intercept_prior_logp(full_model, full_point, pinned=[0, 1])
        assert excluded_logp == pytest.approx(full_logp - pinned_prior)

    def test_default_none_matches_current_wrapper_logp(self, rng):
        """Acceptance criterion 1: omitting `intercept_equations` gives the
        exact same log-probability as the pinned issue-08a wrapper values —
        default behaviour is unchanged."""
        data = _make_data(rng)
        model, _ = VAR(lags=1)._build_pymc_model(data)
        assert _model_logp(model) == pytest.approx(-225.00961968371863)


def _build(spec, endog, data, **kwargs):
    """`spec.build_in_model` on `data`'s names/exog, with `endog` swapped in."""
    return spec.build_in_model(
        endog=endog,
        exog=data.exog,
        n_lags=1,
        endog_names=data.endog_names,
        exog_names=data.exog_names,
        **kwargs,
    )


class TestSymbolicEndog:
    """`build_in_model` with the observed endog block as a PyTensor variable (issue 09a)."""

    def test_pm_data_endog_compiles(self, rng):
        import pymc as pm

        from impulso import ar1_residual_sd

        data = _make_data(rng)
        with pm.Model() as model:
            endog = pm.Data("endog", data.endog)
            _build(VAR(lags=1), endog, data, endog_scales=ar1_residual_sd(data.endog))

        assert np.isfinite(_model_logp(model))

    @pytest.mark.parametrize("error_dist", ["gaussian", "student_t"])
    @pytest.mark.parametrize("exog_names", [None, ["z"]])
    def test_logp_matches_numpy_path(self, rng, error_dist, exog_names):
        """With the tensor fixed to the data and `endog_scales` equal to the
        data's own `ar1_residual_sd`, the Potential contributes the same
        density the observed RV does, so the joint log-probabilities agree."""
        import pymc as pm
        import pytensor

        from impulso import ar1_residual_sd

        data = _make_data(rng, exog_names=exog_names)
        spec = VAR(lags=1, error_dist=error_dist)

        with pm.Model() as numpy_model:
            _build(spec, data.endog, data)
        with pm.Model() as symbolic_model:
            _build(spec, pytensor.shared(data.endog), data, endog_scales=ar1_residual_sd(data.endog))

        assert _model_logp(symbolic_model) == pytest.approx(_model_logp(numpy_model))

    def test_logp_matches_numpy_path_with_pm_data(self, rng):
        import pymc as pm

        from impulso import ar1_residual_sd

        data = _make_data(rng, exog_names=["z"])
        spec = VAR(lags=1, error_dist="student_t")

        with pm.Model() as numpy_model:
            _build(spec, data.endog, data)
        with pm.Model() as symbolic_model:
            endog = pm.Data("endog", data.endog)
            _build(spec, endog, data, endog_scales=ar1_residual_sd(data.endog))

        assert _model_logp(symbolic_model) == pytest.approx(_model_logp(numpy_model))

    def test_likelihood_is_a_potential_named_obs(self, rng):
        """The handles' `obs` is the registered `pm.Potential`, named like
        the numpy path's observed RV."""
        import pymc as pm
        import pytensor

        data = _make_data(rng)
        with pm.Model() as model:
            handles = _build(VAR(lags=1), pytensor.shared(data.endog), data, endog_scales=np.ones(2))

        assert handles.obs is model["obs"]
        assert handles.obs in model.potentials
        assert not model.observed_RVs

    def test_symbolic_endog_without_endog_scales_raises(self, rng):
        import pymc as pm
        import pytensor

        data = _make_data(rng)
        with pm.Model(), pytest.raises(ValueError, match="endog_scales is required"):
            _build(VAR(lags=1), pytensor.shared(data.endog), data)

    def test_endog_scales_flow_into_both_priors(self, rng):
        import pymc as pm
        import pytensor

        from impulso.spec import _exog_prior_sigma

        data = _make_data(rng, exog_names=["z"])
        spec = VAR(lags=1)
        scales = np.array([2.5, 7.0])

        with pm.Model():
            handles = _build(spec, pytensor.shared(data.endog), data, endog_scales=scales)

        b_sigma = handles.B.owner.op.dist_params(handles.B.owner)[1].eval()
        b_exog_sigma = handles.B_exog.owner.op.dist_params(handles.B_exog.owner)[1].eval()
        expected_b = spec.resolved_prior.build_priors(n_vars=2, n_lags=1, sigma=scales)["B_sigma"]
        expected_b_exog = _exog_prior_sigma(scales, data.exog[1:], spec.exog_prior_scale)
        np.testing.assert_allclose(b_sigma, expected_b)
        np.testing.assert_allclose(b_exog_sigma, expected_b_exog)

    def test_intercept_equations_on_symbolic_path(self, rng):
        """Excluding an equation's intercept works the same on the symbolic
        path: same coord and dims, same log-probability as the numpy path."""
        import pymc as pm
        import pytensor

        from impulso import ar1_residual_sd

        data = _make_data(rng, n_vars=3)
        spec = VAR(lags=1)

        with pm.Model() as numpy_model:
            _build(spec, data.endog, data, intercept_equations=["y1", "y3"])
        with pm.Model() as symbolic_model:
            handles = _build(
                spec,
                pytensor.shared(data.endog),
                data,
                endog_scales=ar1_residual_sd(data.endog),
                intercept_equations=["y1", "y3"],
            )

        assert list(symbolic_model.coords["var_intercept"]) == ["y1", "y3"]
        assert symbolic_model.named_vars_to_dims["intercept"] == ("var_intercept",)
        assert handles.intercept is not None
        assert _model_logp(symbolic_model) == pytest.approx(_model_logp(numpy_model))

    def test_static_length_registers_time_coord(self, rng):
        """A tensor with a known static length gets the same positional
        `time` coord as the numpy path."""
        import pymc as pm
        import pytensor.tensor as pt

        data = _make_data(rng)
        with pm.Model() as model:
            _build(VAR(lags=1), pt.constant(data.endog), data, endog_scales=np.ones(2))

        assert len(model.coords["time"]) == data.endog.shape[0] - 1

    def test_static_length_mismatch_with_existing_time_coord_raises(self, rng):
        import pymc as pm
        import pytensor.tensor as pt

        data = _make_data(rng)
        with pm.Model(coords={"time": range(5)}), pytest.raises(ValueError, match="'time' coordinate"):
            _build(VAR(lags=1), pt.constant(data.endog), data, endog_scales=np.ones(2))

    def test_unknown_length_leaves_time_coord_alone(self, rng):
        """`pm.Data` has no static length, and a Potential carries no dims, so
        no `time` coord is registered or checked."""
        import pymc as pm

        data = _make_data(rng)
        with pm.Model(coords={"time": range(5)}) as model:
            endog = pm.Data("endog", data.endog)
            _build(VAR(lags=1), endog, data, endog_scales=np.ones(2))

        assert len(model.coords["time"]) == 5

    def test_pmd_data_values_works(self, rng):
        """The contract pymc-marketing uses: `pmd.Data(...).values`, a plain
        `TensorVariable`, is accepted and matches the numpy path's logp."""
        pmd = pytest.importorskip("pymc.dims")
        import pymc as pm

        from impulso import ar1_residual_sd

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model() as numpy_model:
            _build(spec, data.endog, data)
        with pm.Model(coords={"date": range(data.endog.shape[0]), "series": data.endog_names}) as symbolic_model:
            endog = pmd.Data("endog", data.endog, dims=("date", "series"))
            _build(spec, endog.values, data, endog_scales=ar1_residual_sd(data.endog))

        assert _model_logp(symbolic_model) == pytest.approx(_model_logp(numpy_model))

    def test_raw_xtensor_endog_raises_pointing_at_values(self, rng):
        pmd = pytest.importorskip("pymc.dims")
        import pymc as pm

        data = _make_data(rng)
        with pm.Model(coords={"date": range(data.endog.shape[0]), "series": data.endog_names}):
            endog = pmd.Data("endog", data.endog, dims=("date", "series"))
            with pytest.raises(TypeError, match=r"\.values"):
                _build(VAR(lags=1), endog, data, endog_scales=np.ones(2))

    def test_non_2d_symbolic_endog_raises(self, rng):
        import pymc as pm
        import pytensor

        data = _make_data(rng)
        with pm.Model(), pytest.raises(ValueError, match="must be 2-D"):
            _build(VAR(lags=1), pytensor.shared(data.endog[:, 0]), data, endog_scales=np.ones(2))

    def test_static_column_count_mismatch_raises(self, rng):
        import pymc as pm
        import pytensor.tensor as pt

        data = _make_data(rng, n_vars=3)
        with pm.Model(), pytest.raises(ValueError, match="3 columns but endog_names has 2 names"):
            spec = VAR(lags=1)
            spec.build_in_model(
                endog=pt.constant(data.endog),
                exog=None,
                n_lags=1,
                endog_names=data.endog_names[:2],
                endog_scales=np.ones(2),
            )


_LATENT_XFAIL = pytest.mark.xfail(strict=True, reason="issue 09b")


def _latent_setup(rng: np.random.Generator, n_lags: int = 2, T: int = 40, with_exog: bool = True):
    """Observed block, exog and names for a VAR with one latent series `b` first."""
    obs = rng.standard_normal((T, 2))
    exog = rng.standard_normal((T, 1)) if with_exog else None
    return {
        "endog": obs,
        "exog": exog,
        "n_lags": n_lags,
        "endog_names": ["b", "y1", "y2"],
        "exog_names": ["x"] if with_exog else None,
        "endog_scales": np.array([0.5, 1.0, 2.0]),
        "latent_names": ["b"],
    }


def _perturbed_point(model, rng: np.random.Generator) -> dict:
    """`model`'s initial point with every value variable moved off its default.

    The initial point puts every innovation and initial value at zero, which
    would make the path trivially zero; a random perturbation exercises every
    term of the recursion.
    """
    point = model.initial_point(random_seed=0)
    return {name: value + 0.3 * rng.standard_normal(np.shape(value)) for name, value in point.items()}


def _evaluate(model, point: dict, names: list[str]) -> dict[str, np.ndarray]:
    fn = model.compile_fn([model[name] for name in names], inputs=model.value_vars, on_unused_input="ignore")
    return dict(zip(names, (np.asarray(v) for v in fn(point)), strict=True))


def _intercept_vector(values: dict, intercept_mask: np.ndarray) -> np.ndarray:
    full = np.zeros(intercept_mask.size)
    if "intercept" in values:
        full[intercept_mask] = values["intercept"]
    return full


def _numpy_latent_path(values: dict, obs: np.ndarray, exog: np.ndarray | None, n_lags: int, intercept: np.ndarray):
    """Hand-rolled VAR recursion for the latent block (index 0).

    The latent equation is `b_t = c_b + sum_l A_l[b, :] full_{t-l} + B_exog[b] x_t + L[b, b] z_t`,
    with `full = [b, y1, y2]` and `B` lag-major over `full`.
    """
    B, L, z, init = values["B"], values["L"], values["latent_innovations"], values["latent_init"]
    T, n_vars = obs.shape[0], obs.shape[1] + 1
    full = np.zeros((T, n_vars))
    full[:, 1:] = obs
    full[:n_lags, 0] = init[:, 0]
    for t in range(n_lags, T):
        x_lag = np.concatenate([full[t - lag] for lag in range(1, n_lags + 1)])
        value = intercept[0] + B[0] @ x_lag + L[0, 0] * z[t - n_lags, 0]
        if exog is not None:
            value += values["B_exog"][0] @ exog[t]
        full[t, 0] = value
    return full


class TestLatentSeries:
    """`build_in_model(latent_names=...)`: non-centred latent series (issue 09b)."""

    @_LATENT_XFAIL
    @pytest.mark.parametrize("symbolic", [False, True])
    def test_returned_path_matches_numpy_recursion(self, rng, symbolic):
        import pymc as pm
        import pytensor

        kwargs = _latent_setup(rng)
        obs = kwargs["endog"]
        if symbolic:
            kwargs["endog"] = pytensor.shared(obs)
        with pm.Model() as model:
            handles = VAR(lags=2).build_in_model(**kwargs)

        assert handles.latent.name == "latent"
        point = _perturbed_point(model, rng)
        values = _evaluate(
            model, point, ["latent", "B", "B_exog", "L", "intercept", "latent_init", "latent_innovations"]
        )
        assert values["latent"].shape == (obs.shape[0], 1)
        expected = _numpy_latent_path(values, obs, kwargs["exog"], 2, values["intercept"])
        np.testing.assert_allclose(values["latent"][:, 0], expected[:, 0], rtol=1e-10, atol=1e-10)

    @_LATENT_XFAIL
    @pytest.mark.parametrize("with_exog", [False, True])
    def test_conditional_logp_plus_innovations_equals_joint_var_logp(self, rng, with_exog):
        """Change of variables z -> latent residual `e_b = L[b, b] z`: the
        conditional observed-block density plus the standard-normal density of
        z, minus the Jacobian `sum_t log L[b, b]`, is the joint MvN density of
        the full stacked VAR residuals."""
        import pymc as pm
        from scipy import stats

        kwargs = _latent_setup(rng, with_exog=with_exog)
        obs, exog, n_lags = kwargs["endog"], kwargs["exog"], kwargs["n_lags"]
        with pm.Model() as model:
            VAR(lags=n_lags).build_in_model(**kwargs)

        point = _perturbed_point(model, rng)
        names = ["latent", "B", "L", "intercept", "latent_innovations", "obs"]
        if with_exog:
            names.append("B_exog")
        values = _evaluate(model, point, names)

        full = np.column_stack([values["latent"], obs])
        B, L, z = values["B"], values["L"], values["latent_innovations"]
        rows = range(n_lags, full.shape[0])
        x_lag = np.array([np.concatenate([full[t - lag] for lag in range(1, n_lags + 1)]) for t in rows])
        mu = values["intercept"] + x_lag @ B.T
        if with_exog:
            mu += exog[n_lags:] @ values["B_exog"].T
        resid = full[n_lags:] - mu
        joint = stats.multivariate_normal(mean=np.zeros(3), cov=L @ L.T).logpdf(resid).sum()

        n_rows = len(rows)
        conditional = float(values["obs"])
        innovations = stats.norm.logpdf(z).sum()
        jacobian = -n_rows * np.log(L[0, 0])
        assert conditional + innovations + jacobian == pytest.approx(joint, rel=1e-10)

    @_LATENT_XFAIL
    def test_latent_series_can_be_excluded_from_intercept_equations(self, rng):
        import pymc as pm

        kwargs = _latent_setup(rng)
        with pm.Model() as model:
            handles = VAR(lags=2).build_in_model(**kwargs, intercept_equations=["y1", "y2"])

        assert list(model.coords["var_intercept"]) == ["y1", "y2"]
        assert handles.intercept is not None
        point = _perturbed_point(model, rng)
        values = _evaluate(
            model, point, ["latent", "B", "B_exog", "L", "intercept", "latent_init", "latent_innovations"]
        )
        intercept = _intercept_vector(values, np.array([False, True, True]))
        expected = _numpy_latent_path(values, kwargs["endog"], kwargs["exog"], 2, intercept)
        np.testing.assert_allclose(values["latent"][:, 0], expected[:, 0], rtol=1e-10, atol=1e-10)

    @_LATENT_XFAIL
    def test_handles_carry_latent_names(self, rng):
        import pymc as pm

        with pm.Model():
            handles = VAR(lags=2).build_in_model(**_latent_setup(rng))

        assert handles.latent_names == ("b",)

    @_LATENT_XFAIL
    def test_nested_named_model_prefixes_latent_variables(self, rng):
        import pymc as pm

        with pm.Model() as root, pm.Model(name="brand"):
            handles = VAR(lags=2).build_in_model(**_latent_setup(rng))

        names = set(root.named_vars)
        for name in ["latent", "latent_init", "latent_innovations", "obs", "B", "L", "intercept"]:
            assert f"brand::{name}" in names
            assert name not in names
        assert handles.latent.name == "brand::latent"

    @_LATENT_XFAIL
    def test_latent_path_is_finite_at_the_initial_point(self, rng):
        import pymc as pm

        with pm.Model() as model:
            VAR(lags=2).build_in_model(**_latent_setup(rng))

        assert np.isfinite(_model_logp(model))

    @_LATENT_XFAIL
    def test_missing_endog_scales_raises(self, rng):
        import pymc as pm

        kwargs = _latent_setup(rng)
        kwargs["endog_scales"] = None
        with pm.Model(), pytest.raises(ValueError, match="endog_scales"):
            VAR(lags=2).build_in_model(**kwargs)

    @_LATENT_XFAIL
    def test_endog_scales_without_a_latent_entry_raises(self, rng):
        import pymc as pm

        kwargs = _latent_setup(rng)
        kwargs["endog_scales"] = np.array([1.0, 2.0])  # observed columns only
        with pm.Model(), pytest.raises(ValueError, match=r"endog_scales.*latent"):
            VAR(lags=2).build_in_model(**kwargs)

    @_LATENT_XFAIL
    def test_latent_name_not_at_start_of_endog_names_raises(self, rng):
        import pymc as pm

        kwargs = _latent_setup(rng)
        kwargs["endog_names"] = ["y1", "b", "y2"]
        with pm.Model(), pytest.raises(ValueError, match="first"):
            VAR(lags=2).build_in_model(**kwargs)

    @_LATENT_XFAIL
    def test_observed_column_count_mismatch_raises(self, rng):
        import pymc as pm

        kwargs = _latent_setup(rng)
        kwargs["endog"] = np.column_stack([rng.standard_normal(40), kwargs["endog"]])  # latent column passed too
        with pm.Model(), pytest.raises(ValueError, match="observed"):
            VAR(lags=2).build_in_model(**kwargs)

    @_LATENT_XFAIL
    def test_non_gaussian_errors_raise(self, rng):
        import pymc as pm

        with pm.Model(), pytest.raises(ValueError, match="Gaussian"):
            VAR(lags=2, error_dist="student_t").build_in_model(**_latent_setup(rng))

    @_LATENT_XFAIL
    def test_latent_init_sigma_sets_the_initial_value_prior(self, rng):
        import pymc as pm
        from scipy import stats

        with pm.Model() as model:
            VAR(lags=2).build_in_model(**_latent_setup(rng), latent_init_sigma=3.0)  # ty: ignore[unknown-argument]

        point = {"latent_init": np.array([[0.4], [-1.2]])}
        (logp,) = model.compile_logp(vars=[model["latent_init"]], sum=False)({
            **model.initial_point(random_seed=0),
            **point,
        })
        np.testing.assert_allclose(np.ravel(logp), stats.norm(0, 3.0).logpdf([0.4, -1.2]))

    @_LATENT_XFAIL
    @pytest.mark.slow
    def test_small_latent_var_samples(self):
        """One latent and one observed series, simulated from a stationary VAR(1).

        A latent series in a plain VAR has no data anchoring its scale (its
        innovation sd trades off against its loadings) and nothing yet keeps
        its own-lag inside the stationary region (issue 09c), so the setup is
        gentle: a weak loading, a tight prior on the latent innovation scale,
        and a stationary starting point. The check is that sampling runs and
        divergences stay a small fraction, not that the latent is recovered.
        """
        import pymc as pm

        from impulso.volatility import Constant, InnovationScalePrior

        rng = np.random.default_rng(7)
        T = 120
        A = np.array([[0.5, 0.0], [0.3, 0.3]])
        full = np.zeros((T, 2))
        for t in range(1, T):
            full[t] = A @ full[t - 1] + np.array([0.5, 0.3]) * rng.standard_normal(2)

        volatility = Constant(
            innovation_scale_priors=[
                InnovationScalePrior(family="halfnormal", scale=0.1),
                InnovationScalePrior(family="halfnormal", scale=0.5),
            ]
        )
        draws, chains = 200, 2
        with pm.Model():
            VAR(lags=1, volatility=volatility).build_in_model(
                endog=full[:, 1:],
                exog=None,
                n_lags=1,
                endog_names=["b", "y"],
                endog_scales=[0.5, 0.3],
                latent_names=["b"],  # ty: ignore[unknown-argument]
                latent_init_sigma=0.5,  # ty: ignore[unknown-argument]
                intercept_equations=["y"],
            )
            idata = pm.sample(
                draws=draws,
                tune=300,
                chains=chains,
                cores=1,
                random_seed=1,
                progressbar=False,
                nuts_sampler="pymc",
                target_accept=0.9,
                initvals={"B": np.array([[0.5, 0.0], [0.0, 0.5]])},
            )

        divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
        assert divergences < 0.1 * draws * chains
        assert np.all(np.isfinite(np.asarray(idata.posterior["latent"])))

    @_LATENT_XFAIL
    @pytest.mark.parametrize("n_lags", [1, 3])
    @pytest.mark.parametrize("n_latent", [1, 2])
    def test_path_and_joint_logp_across_lag_orders_and_latent_counts(self, rng, n_lags, n_latent):
        """The recursion and the change-of-variables identity hold for any
        lag order and any number of latent series, including the `n_lags == 1`
        and `n_latent == 1` edge cases whose static shapes scan must keep."""
        import pymc as pm
        from scipy import stats

        T, n_obs = 30, 2
        n_vars = n_latent + n_obs
        obs = rng.standard_normal((T, n_obs))
        exog = rng.standard_normal((T, 1))
        latent_names = [f"b{i}" for i in range(n_latent)]
        with pm.Model() as model:
            VAR(lags=n_lags).build_in_model(
                endog=obs,
                exog=exog,
                n_lags=n_lags,
                endog_names=[*latent_names, "y1", "y2"],
                exog_names=["x"],
                endog_scales=np.linspace(0.5, 2.0, n_vars),
                latent_names=latent_names,  # ty: ignore[unknown-argument]
            )

        point = _perturbed_point(model, rng)
        values = _evaluate(
            model, point, ["latent", "B", "B_exog", "L", "intercept", "latent_init", "latent_innovations", "obs"]
        )
        B, L, z, c = values["B"], values["L"], values["latent_innovations"], values["intercept"]

        full = np.zeros((T, n_vars))
        full[:, n_latent:] = obs
        full[:n_lags, :n_latent] = values["latent_init"]
        for t in range(n_lags, T):
            x_lag = np.concatenate([full[t - lag] for lag in range(1, n_lags + 1)])
            mean = c + B @ x_lag + values["B_exog"] @ exog[t]
            full[t, :n_latent] = mean[:n_latent] + L[:n_latent, :n_latent] @ z[t - n_lags]
        np.testing.assert_allclose(values["latent"], full[:, :n_latent], rtol=1e-10, atol=1e-10)

        x_lag = np.array([np.concatenate([full[t - lag] for lag in range(1, n_lags + 1)]) for t in range(n_lags, T)])
        resid = full[n_lags:] - (c + x_lag @ B.T + exog[n_lags:] @ values["B_exog"].T)
        joint = stats.multivariate_normal(mean=np.zeros(n_vars), cov=L @ L.T).logpdf(resid).sum()
        jacobian = -(T - n_lags) * np.log(np.diag(L)[:n_latent]).sum()
        total = float(values["obs"]) + stats.norm.logpdf(z).sum() + jacobian
        assert total == pytest.approx(joint, rel=1e-10)

    @_LATENT_XFAIL
    @pytest.mark.parametrize("n_lags", [1, 2])
    def test_compiles_under_nutpie_with_a_single_latent_series(self, rng, n_lags):
        """nutpie swaps each value variable for a reshaped slice of one flat
        vector, which can make a length-1 axis static that was unknown when
        the scan was built; scan then rejects the rebuilt node unless its
        input shapes are pinned."""
        import pymc as pm

        nutpie = pytest.importorskip("nutpie")
        with pm.Model() as model:
            VAR(lags=n_lags).build_in_model(
                endog=rng.standard_normal((30, 1)),
                exog=None,
                n_lags=n_lags,
                endog_names=["b", "y"],
                endog_scales=[1.0, 1.0],
                latent_names=["b"],  # ty: ignore[unknown-argument]
            )

        nutpie.compile_pymc_model(model)
