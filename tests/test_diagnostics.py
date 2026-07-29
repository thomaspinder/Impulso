"""Tests for the VAR-aware convergence and stability report."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from arviz import InferenceData

from impulso.conjugate import ConjugateVAR
from impulso.conjugate_volatility import PandemicBreak
from impulso.diagnostics import (
    BlockDiagnostics,
    ConvergenceReport,
    ConvergenceThresholds,
    StabilitySummary,
    assign_blocks,
    convergence_report,
)
from impulso.fitted import FittedVAR
from impulso.identification import Cholesky
from impulso.priors import NIWPrior
from impulso.sv.spec import StochasticVolatility
from impulso.volatility import Constant


def codes(report: ConvergenceReport) -> list[str]:
    return [message.code for message in report.messages]


def blocks_by_name(report: ConvergenceReport) -> dict[str, BlockDiagnostics]:
    return {block.block: block for block in report.blocks}


class TestHealthyPosterior:
    def test_status_passed_with_no_messages(self, make_var_posterior):
        report = convergence_report(make_var_posterior(), n_lags=1, var_names=["y1", "y2"])
        assert report.status == "passed"
        assert report.messages == []

    def test_headline_metrics_are_sane(self, make_var_posterior):
        report = convergence_report(make_var_posterior(), n_lags=1, var_names=["y1", "y2"])
        assert report.max_rhat is not None
        assert report.max_rhat < 1.01
        assert report.min_ess_bulk is not None
        assert report.min_ess_bulk > 400
        assert report.n_chains == 4
        assert report.n_draws == 200

    def test_stability_is_centred_on_half(self, make_var_posterior):
        report = convergence_report(make_var_posterior(), n_lags=1)
        assert report.stability.p_explosive == 0.0
        assert report.stability.median() == pytest.approx(0.5, abs=0.05)
        assert report.stability.n_vars == 2
        assert report.stability.n_lags == 1

    def test_blocks_present_in_canonical_order(self, make_var_posterior):
        report = convergence_report(make_var_posterior(), n_lags=1)
        assert [block.block for block in report.blocks] == ["coefficient", "intercept", "covariance"]

    def test_structural_zeros_do_not_break_the_covariance_block(self, make_var_posterior):
        # `L` is lower triangular, so L[y1, y2] is constant at zero and its
        # R-hat is NaN. The block still reports a finite worst value.
        block = blocks_by_name(convergence_report(make_var_posterior(), n_lags=1))["covariance"]
        assert block.max_rhat is not None
        assert np.isfinite(block.max_rhat)
        assert block.n_coordinates == 6  # sigma_sd (2) + L (2x2)

    def test_report_emits_no_python_warnings(self, make_var_posterior):
        # The report object is the channel; it never uses `warnings.warn`,
        # and it swallows the divide-by-zero notice ArviZ raises on the
        # structural zeros in `L`.
        import warnings

        idata = make_var_posterior()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            convergence_report(idata, n_lags=1)
        assert caught == []


class TestNonConvergedPosterior:
    @pytest.fixture
    def report(self, make_var_posterior):
        idata = make_var_posterior(bad_coord=(1, 0), divergences=0)
        return convergence_report(idata, n_lags=1, var_names=["y1", "y2"])

    def test_status_failed(self, report):
        assert report.status == "failed"
        assert report.max_rhat > 1.05

    def test_rhat_without_divergences_code_present(self, report):
        assert "rhat_without_divergences" in codes(report)

    def test_message_quotes_the_low_rank_mass_matrix_remedy(self, report):
        message = next(m for m in report.messages if m.code == "rhat_without_divergences")
        assert "low_rank_modified_mass_matrix" in message.message
        assert 'nuts_sampler="nutpie"' in message.message

    def test_worst_coordinate_named_with_coords(self, report):
        assert report.blocks[0].max_rhat_coord == "B[y2, L1.y1]"

    def test_worst_coordinate_named_without_coords(self, make_var_posterior):
        idata = make_var_posterior(bad_coord=(1, 0), coords=False)
        report = convergence_report(idata, n_lags=1, var_names=["y1", "y2"])
        assert report.blocks[0].max_rhat_coord == "B[y2, L1.y1]"

    def test_worst_coordinate_falls_back_to_positions(self, make_var_posterior):
        idata = make_var_posterior(bad_coord=(1, 0), coords=False)
        report = convergence_report(idata, n_lags=1, var_names=None)
        assert report.blocks[0].max_rhat_coord == "B[1, 0]"

    def test_healthy_blocks_are_not_implicated(self, report):
        intercept_rhat = blocks_by_name(report)["intercept"].max_rhat
        assert intercept_rhat is not None
        assert intercept_rhat < 1.01

    def test_divergences_present_suppresses_the_zero_divergence_message(self, make_var_posterior):
        idata = make_var_posterior(bad_coord=(1, 0), divergences=40)
        report = convergence_report(idata, n_lags=1)
        assert "rhat_without_divergences" not in codes(report)
        assert "divergences_present" in codes(report)
        assert report.status == "failed"


class TestExplosiveDraws:
    @pytest.fixture
    def report(self, make_var_posterior):
        idata = make_var_posterior(explosive_frac=0.15)
        return convergence_report(idata, n_lags=1, var_names=["y1", "y2"])

    def test_exact_stability_statistics(self, report):
        assert report.stability.p_explosive == pytest.approx(0.15)
        assert report.stability.max_radius == pytest.approx(1.2)
        assert report.stability.median() == pytest.approx(0.5)

    def test_explosive_code_raised_to_warning(self, report):
        message = next(m for m in report.messages if m.code == "explosive_draws")
        assert message.severity == "warning"
        assert message.block == "coefficient"

    def test_explosive_draws_never_fail_the_report(self, report):
        # Reserved-status decision: `failed` means sampler pathology only.
        # Near-unit-root mass is a legitimate posterior statement.
        assert report.status == "warnings"

    def test_below_threshold_explosive_mass_is_informational(self, make_var_posterior):
        idata = make_var_posterior(explosive_frac=0.02)
        report = convergence_report(idata, n_lags=1)
        message = next(m for m in report.messages if m.code == "explosive_draws")
        assert message.severity == "info"
        assert report.status == "passed"

    def test_no_explosive_message_when_all_draws_stable(self, make_var_posterior):
        report = convergence_report(make_var_posterior(), n_lags=1)
        assert "explosive_draws" not in codes(report)


class TestDivergences:
    def test_counts_are_exact(self, make_var_posterior):
        report = convergence_report(make_var_posterior(divergences=7), n_lags=1)
        assert report.divergences == 7
        assert report.n_transitions == 800
        assert report.divergence_rate == pytest.approx(7 / 800)
        assert report.sampler_stats_available is True

    def test_low_rate_warns_but_does_not_fail(self, make_var_posterior):
        report = convergence_report(make_var_posterior(divergences=3), n_lags=1)
        assert report.status == "warnings"
        assert next(m for m in report.messages if m.code == "divergences_present").severity == "warning"

    def test_rate_at_one_percent_fails(self, make_var_posterior):
        report = convergence_report(make_var_posterior(divergences=8), n_lags=1)
        assert report.divergence_rate == pytest.approx(0.01)
        assert report.status == "failed"

    def test_nutpie_shaped_stats_match_pymc_shaped(self, make_var_posterior):
        nutpie = convergence_report(make_var_posterior(divergences=5, nutpie_shaped=True), n_lags=1)
        pymc = convergence_report(make_var_posterior(divergences=5), n_lags=1)
        assert nutpie.divergences == pymc.divergences == 5
        assert nutpie.n_transitions == pymc.n_transitions == 800

    def test_warmup_divergences_are_ignored(self, make_var_posterior):
        idata = make_var_posterior(divergences=0, nutpie_shaped=True)
        assert "warmup_sample_stats" in idata.groups()
        report = convergence_report(idata, n_lags=1)
        assert report.divergences == 0
        assert report.status == "passed"


class TestEnergy:
    def test_healthy_energy_is_reported_without_a_message(self, make_var_posterior):
        report = convergence_report(make_var_posterior(), n_lags=1)
        assert report.ebfmi is not None
        assert len(report.ebfmi) == 4
        assert all(value > 0.3 for value in report.ebfmi)
        assert report.min_ebfmi == min(report.ebfmi)
        assert "low_ebfmi" not in codes(report)

    def test_slow_energy_exploration_warns(self, make_var_posterior):
        report = convergence_report(make_var_posterior(energy_rho=0.97), n_lags=1)
        assert report.min_ebfmi is not None
        assert report.min_ebfmi < 0.3
        message = next(m for m in report.messages if m.code == "low_ebfmi")
        assert message.severity == "warning"
        assert report.status == "warnings"

    def test_message_names_the_worst_chain(self, make_var_posterior):
        report = convergence_report(make_var_posterior(energy_rho=0.97), n_lags=1)
        worst = report.ebfmi.index(min(report.ebfmi))
        message = next(m for m in report.messages if m.code == "low_ebfmi")
        assert f"chain {worst}" in message.message

    def test_threshold_is_respected(self, make_var_posterior):
        idata = make_var_posterior(energy_rho=0.97)
        lenient = ConvergenceThresholds(ebfmi_warn=0.0)
        assert "low_ebfmi" not in codes(convergence_report(idata, n_lags=1, thresholds=lenient))

    def test_comparison_is_strict_at_the_threshold(self, make_var_posterior):
        idata = make_var_posterior(energy_rho=0.97)
        observed = convergence_report(idata, n_lags=1).min_ebfmi
        assert observed is not None
        on_threshold = ConvergenceThresholds(ebfmi_warn=observed)
        just_above = ConvergenceThresholds(ebfmi_warn=observed + 1e-12)
        assert "low_ebfmi" not in codes(convergence_report(idata, n_lags=1, thresholds=on_threshold))
        assert "low_ebfmi" in codes(convergence_report(idata, n_lags=1, thresholds=just_above))

    def test_low_ebfmi_never_fails_the_report(self, make_var_posterior):
        # An efficiency pathology, not evidence the draws are wrong.
        report = convergence_report(make_var_posterior(energy_rho=0.99), n_lags=1)
        assert report.status == "warnings"
        assert not any(m.severity == "failure" for m in report.messages)

    def test_nutpie_shaped_energy_matches_pymc_shaped(self, make_var_posterior):
        nutpie = convergence_report(make_var_posterior(nutpie_shaped=True), n_lags=1)
        pymc = convergence_report(make_var_posterior(), n_lags=1)
        assert nutpie.ebfmi == pytest.approx(pymc.ebfmi)

    def test_warmup_energy_is_ignored(self, make_var_posterior):
        # The warmup group carries a constant energy trace, whose BFMI is
        # undefined; reading it would erase the post-warmup diagnosis.
        idata = make_var_posterior(nutpie_shaped=True)
        assert "warmup_sample_stats" in idata.groups()
        report = convergence_report(idata, n_lags=1)
        assert report.min_ebfmi is not None
        assert report.min_ebfmi > 0.3


class TestMaxTreedepth:
    def test_no_saturation_is_reported_as_zero(self, make_var_posterior):
        report = convergence_report(make_var_posterior(), n_lags=1)
        assert report.treedepth_saturations == 0
        assert report.treedepth_saturation_rate == 0.0
        assert "treedepth_saturation" not in codes(report)

    def test_counts_are_exact_from_pymc_shaped_stats(self, make_var_posterior):
        report = convergence_report(make_var_posterior(treedepth_hits=40), n_lags=1)
        assert report.treedepth_saturations == 40
        assert report.treedepth_saturation_rate == pytest.approx(40 / 800)

    def test_counts_are_exact_from_nutpie_shaped_stats(self, make_var_posterior):
        report = convergence_report(make_var_posterior(treedepth_hits=40, nutpie_shaped=True), n_lags=1)
        assert report.treedepth_saturations == 40
        assert report.treedepth_saturation_rate == pytest.approx(40 / 800)

    def test_backends_agree(self, make_var_posterior):
        nutpie = convergence_report(make_var_posterior(treedepth_hits=13, nutpie_shaped=True), n_lags=1)
        pymc = convergence_report(make_var_posterior(treedepth_hits=13), n_lags=1)
        assert nutpie.treedepth_saturations == pymc.treedepth_saturations == 13
        assert codes(nutpie) == codes(pymc)

    def test_warmup_saturations_are_ignored(self, make_var_posterior):
        idata = make_var_posterior(treedepth_hits=0, nutpie_shaped=True)
        assert "warmup_sample_stats" in idata.groups()
        assert convergence_report(idata, n_lags=1).treedepth_saturations == 0

    def test_rate_below_the_threshold_is_silent(self, make_var_posterior):
        report = convergence_report(make_var_posterior(treedepth_hits=7), n_lags=1)
        assert report.treedepth_saturation_rate is not None
        assert report.treedepth_saturation_rate < 0.01
        assert "treedepth_saturation" not in codes(report)

    def test_rate_at_the_threshold_warns(self, make_var_posterior):
        report = convergence_report(make_var_posterior(treedepth_hits=8), n_lags=1)
        assert report.treedepth_saturation_rate == pytest.approx(0.01)
        message = next(m for m in report.messages if m.code == "treedepth_saturation")
        assert message.severity == "warning"
        assert report.status == "warnings"

    def test_saturation_never_fails_the_report(self, make_var_posterior):
        # Deep trees cost wall-clock time and autocorrelation, not correctness.
        report = convergence_report(make_var_posterior(treedepth_hits=800), n_lags=1)
        assert report.status == "warnings"
        assert not any(m.severity == "failure" for m in report.messages)

    def test_message_quotes_the_low_rank_mass_matrix_remedy(self, make_var_posterior):
        report = convergence_report(make_var_posterior(treedepth_hits=80), n_lags=1)
        message = next(m for m in report.messages if m.code == "treedepth_saturation")
        assert "max_treedepth" in message.message
        assert "low_rank_modified_mass_matrix" in message.message

    def test_custom_threshold_flips_the_message(self, make_var_posterior):
        idata = make_var_posterior(treedepth_hits=4)
        strict = ConvergenceThresholds(treedepth_warn_rate=0.001)
        assert "treedepth_saturation" not in codes(convergence_report(idata, n_lags=1))
        assert "treedepth_saturation" in codes(convergence_report(idata, n_lags=1, thresholds=strict))


class TestMissingEfficiencyStats:
    def test_stats_present_but_energy_and_treedepth_absent(self, make_var_posterior):
        idata = make_var_posterior()
        stripped = InferenceData(
            posterior=idata.posterior,
            sample_stats=idata.sample_stats.drop_vars(["energy", "reached_max_treedepth"]),
        )
        report = convergence_report(stripped, n_lags=1)
        assert report.sampler_stats_available is True
        assert report.divergences == 0
        assert report.ebfmi is None
        assert report.min_ebfmi is None
        assert report.treedepth_saturations is None
        assert report.treedepth_saturation_rate is None
        assert report.status == "passed"

    def test_constant_energy_is_reported_as_absent_not_as_a_pathology(self, make_var_posterior):
        idata = make_var_posterior()
        stats = idata.sample_stats.copy()
        stats["energy"] = (("chain", "draw"), np.zeros(stats["energy"].shape))
        report = convergence_report(InferenceData(posterior=idata.posterior, sample_stats=stats), n_lags=1)
        assert report.ebfmi is None
        assert "low_ebfmi" not in codes(report)
        assert report.status == "passed"


class TestMissingSamplerStats:
    @pytest.fixture
    def report(self, make_var_posterior):
        return convergence_report(make_var_posterior(divergences=None), n_lags=1)

    def test_availability_flag_and_none_counts(self, report):
        assert report.sampler_stats_available is False
        assert report.divergences is None
        assert report.n_transitions is None
        assert report.divergence_rate is None

    def test_efficiency_metrics_are_none_without_stats(self, report):
        assert report.ebfmi is None
        assert report.min_ebfmi is None
        assert report.treedepth_saturations is None
        assert report.treedepth_saturation_rate is None

    def test_message_is_informational_only(self, report):
        message = next(m for m in report.messages if m.code == "sampler_stats_missing")
        assert message.severity == "info"

    def test_status_not_degraded_by_missing_stats_alone(self, report):
        assert report.status == "passed"

    def test_zero_divergence_message_requires_stats(self, make_var_posterior):
        idata = make_var_posterior(bad_coord=(1, 0), divergences=None)
        report = convergence_report(idata, n_lags=1)
        assert "rhat_without_divergences" not in codes(report)


class TestSingleChain:
    @pytest.fixture
    def report(self, make_var_posterior):
        idata = make_var_posterior(n_chains=1, n_draws=800)
        return convergence_report(idata, n_lags=1, var_names=["y1", "y2"])

    def test_rhat_is_none_everywhere(self, report):
        assert report.max_rhat is None
        assert all(block.max_rhat is None and block.max_rhat_coord is None for block in report.blocks)

    def test_ess_still_computed(self, report):
        assert report.min_ess_bulk is not None
        assert report.min_ess_bulk > 0

    def test_single_chain_message_and_capped_status(self, report):
        assert "single_chain" in codes(report)
        assert report.status != "failed"


class TestBlockAssignment:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("B", "coefficient"),
            ("intercept", "intercept"),
            ("B_exog", "exog"),
            ("sigma_sd", "covariance"),
            ("tril_offdiag", "covariance"),
            ("L", "covariance"),
            ("Sigma", "covariance"),
            ("h", "volatility"),
            ("R_chol", "volatility"),
            ("R_chol_offdiag", "volatility"),
            ("v0_h", "volatility"),
            ("v1_sigma_eta", "volatility"),
            ("v12_phi", "volatility"),
            ("structural_shock_matrix", "identification"),
            ("P", "identification"),
            ("weird_thing", "other"),
            ("lambda_", "other"),
        ],
    )
    def test_variable_maps_to_expected_block(self, name, expected):
        posterior = xr.Dataset({name: (("chain", "draw"), np.zeros((2, 3)))})
        assert assign_blocks(posterior) == {expected: [name]}

    def test_unknown_variable_never_raises(self, make_var_posterior):
        idata = make_var_posterior(extra_vars={"weird_thing": (3,)})
        report = convergence_report(idata, n_lags=1)
        assert blocks_by_name(report)["other"].var_names == ["weird_thing"]

    def test_absent_blocks_are_omitted(self, make_var_posterior):
        report = convergence_report(make_var_posterior(), n_lags=1)
        assert "exog" not in blocks_by_name(report)
        assert "volatility" not in blocks_by_name(report)

    def test_canonical_order_in_to_dataframe(self, make_var_posterior):
        idata = make_var_posterior(extra_vars={"weird_thing": (), "B_exog": (1,), "h": (4, 2)})
        frame = convergence_report(idata, n_lags=1).to_dataframe()
        assert list(frame.index) == ["coefficient", "intercept", "exog", "covariance", "volatility", "other"]

    def test_variables_sorted_within_a_block(self, make_var_posterior):
        idata = make_var_posterior(extra_vars={"Sigma": (2, 2), "tril_offdiag": (1,)})
        assert blocks_by_name(convergence_report(idata, n_lags=1))["covariance"].var_names == [
            "L",
            "Sigma",
            "sigma_sd",
            "tril_offdiag",
        ]


class TestAdapterHook:
    def test_constant_claims_its_own_variables(self):
        assert Constant().posterior_var_names() == ("sigma_sd", "tril_offdiag", "L", "Sigma")

    def test_stochastic_volatility_claims_shared_variables(self):
        assert StochasticVolatility().posterior_var_names() == ("h", "R_chol", "R_chol_offdiag")

    def test_pandemic_break_claims_its_hyperparameters(self):
        assert PandemicBreak(start=3).posterior_var_names() == ("s_march", "s_april", "s_may", "rho")

    def test_claimed_names_route_to_volatility_for_a_time_varying_adapter(self):
        posterior = xr.Dataset({
            name: (("chain", "draw"), np.zeros((2, 3))) for name in ("s_march", "s_april", "s_may", "rho", "lambda_")
        })
        blocks = assign_blocks(posterior, volatility=PandemicBreak(start=3))
        assert blocks["volatility"] == ["rho", "s_april", "s_march", "s_may"]
        assert blocks["other"] == ["lambda_"]

    def test_claimed_names_route_to_covariance_for_a_constant_adapter(self):
        posterior = xr.Dataset({"custom_scale": (("chain", "draw"), np.zeros((2, 3)))})

        class _CustomConstant(Constant):
            def posterior_var_names(self) -> tuple[str, ...]:
                return ("custom_scale",)

        assert assign_blocks(posterior, volatility=_CustomConstant()) == {"covariance": ["custom_scale"]}

    def test_adapter_without_the_hook_is_fine(self):
        posterior = xr.Dataset({"custom_scale": (("chain", "draw"), np.zeros((2, 3)))})

        class _Bare:
            is_time_varying = False

        assert assign_blocks(posterior, volatility=_Bare()) == {"other": ["custom_scale"]}


class TestThresholds:
    def test_custom_thresholds_flip_status(self, make_var_posterior):
        idata = make_var_posterior()
        strict = ConvergenceThresholds(rhat_warn=1.0, rhat_fail=1.001)
        assert convergence_report(idata, n_lags=1).status == "passed"
        assert convergence_report(idata, n_lags=1, thresholds=strict).status == "failed"

    def test_comparison_is_strict_at_the_threshold(self, make_var_posterior):
        idata = make_var_posterior()
        observed = convergence_report(idata, n_lags=1).max_rhat
        assert observed is not None
        on_threshold = ConvergenceThresholds(rhat_warn=observed)
        just_below = ConvergenceThresholds(rhat_warn=observed - 1e-12)
        assert convergence_report(idata, n_lags=1, thresholds=on_threshold).status == "passed"
        assert convergence_report(idata, n_lags=1, thresholds=just_below).status == "warnings"

    def test_thresholds_echoed_on_the_report(self, make_var_posterior):
        custom = ConvergenceThresholds(ess_warn=10.0, explosive_warn=0.5)
        report = convergence_report(make_var_posterior(), n_lags=1, thresholds=custom)
        assert report.thresholds == custom

    def test_defaults_match_the_documented_values(self):
        thresholds = ConvergenceThresholds()
        assert (thresholds.rhat_warn, thresholds.rhat_fail) == (1.01, 1.05)
        assert (thresholds.ess_warn, thresholds.ess_fail) == (400.0, 100.0)
        assert thresholds.divergence_fail_rate == 0.01
        assert thresholds.ebfmi_warn == 0.3
        assert thresholds.treedepth_warn_rate == 0.01
        assert thresholds.explosive_warn == 0.05


class TestRendering:
    def test_to_dataframe_columns_and_index(self, make_var_posterior):
        frame = convergence_report(make_var_posterior(), n_lags=1).to_dataframe()
        assert frame.index.name == "block"
        assert list(frame.columns) == [
            "n_variables",
            "n_coordinates",
            "max_rhat",
            "max_rhat_coord",
            "min_ess_bulk",
            "min_ess_bulk_coord",
            "min_ess_tail",
            "min_ess_tail_coord",
        ]

    def test_summary_mentions_blocks_divergences_and_headlines(self, make_var_posterior):
        report = convergence_report(make_var_posterior(divergences=3), n_lags=1, var_names=["y1", "y2"])
        text = report.summary()
        assert "coefficient" in text
        assert "covariance" in text
        assert "divergences: 3" in text
        assert "explosive draws" in text
        assert "divergences_present" in text

    def test_summary_reports_unavailable_stats(self, make_var_posterior):
        text = convergence_report(make_var_posterior(divergences=None), n_lags=1).summary()
        assert "divergences: unavailable" in text
        assert "E-BFMI" not in text
        assert "max-treedepth" not in text

    def test_summary_carries_the_efficiency_headlines(self, make_var_posterior):
        text = convergence_report(make_var_posterior(treedepth_hits=8), n_lags=1).summary()
        assert "min E-BFMI:" in text
        assert "max-treedepth hits: 8 (1.00%)" in text

    def test_repr_is_one_line(self, make_var_posterior):
        text = repr(convergence_report(make_var_posterior(), n_lags=1))
        assert "\n" not in text
        assert text.startswith("ConvergenceReport(status='passed'")

    def test_stability_to_dataframe_is_a_single_row(self, make_var_posterior):
        frame = convergence_report(make_var_posterior(), n_lags=1).stability.to_dataframe()
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == 1
        assert frame.loc["stability", "p_explosive"] == 0.0

    def test_stability_hdi_brackets_the_median(self, make_var_posterior):
        stability = convergence_report(make_var_posterior(), n_lags=1).stability
        lower, upper = stability.hdi()
        assert lower <= stability.median() <= upper
        wide_lower, wide_upper = stability.hdi(prob=0.99)
        assert wide_lower <= lower and wide_upper >= upper


class TestStabilityArray:
    def test_radius_is_read_only(self, make_var_posterior):
        radius = convergence_report(make_var_posterior(), n_lags=1).stability.radius
        with pytest.raises(ValueError, match="read-only"):
            radius[0, 0] = 99.0

    def test_radius_shape_matches_the_posterior(self, make_var_posterior):
        stability = convergence_report(make_var_posterior(), n_lags=1).stability
        assert stability.radius.shape == (4, 200)
        assert stability.thinned_from is None

    def test_thinning_is_deterministic_and_recorded(self, make_var_posterior):
        idata = make_var_posterior(explosive_frac=0.15)
        first = convergence_report(idata, n_lags=1, stability_draws=50).stability
        second = convergence_report(idata, n_lags=1, stability_draws=50).stability
        assert first.thinned_from == 200
        assert first.radius.shape == (4, 50)
        np.testing.assert_array_equal(first.radius, second.radius)
        assert first.p_explosive == pytest.approx(0.15, abs=0.1)

    def test_thinning_larger_than_the_posterior_is_a_no_op(self, make_var_posterior):
        stability = convergence_report(make_var_posterior(), n_lags=1, stability_draws=10_000).stability
        assert stability.thinned_from is None
        assert stability.radius.shape == (4, 200)

    def test_eigenvalues_are_read_only(self, make_var_posterior):
        eigenvalues = convergence_report(make_var_posterior(), n_lags=1).stability.eigenvalues
        with pytest.raises(ValueError, match="read-only"):
            eigenvalues[0, 0] = 99.0

    def test_eigenvalues_are_pooled_capped_and_strided(self, make_var_posterior):
        # 4 x 200 = 800 pooled draws, capped at 200: a stride of 4.
        stability = convergence_report(make_var_posterior(), n_lags=1).stability
        assert stability.eigenvalues.shape == (200, 2)
        assert np.iscomplexobj(stability.eigenvalues)
        np.testing.assert_allclose(
            np.max(np.abs(stability.eigenvalues), axis=-1),
            stability.radius.reshape(-1)[::4],
        )

    def test_small_posteriors_keep_every_eigenvalue(self, make_var_posterior):
        stability = convergence_report(make_var_posterior(n_chains=1, n_draws=50), n_lags=1).stability
        assert stability.eigenvalues.shape == (50, 2)

    def test_eigenvalue_trailing_axis_is_the_companion_dimension(self, make_var_posterior):
        stability = convergence_report(make_var_posterior(n_vars=3, n_lags=2), n_lags=2).stability
        assert stability.eigenvalues.shape[-1] == 6

    def test_rejects_non_positive_thinning(self, make_var_posterior):
        with pytest.raises(ValueError, match="stability_draws must be positive"):
            convergence_report(make_var_posterior(), n_lags=1, stability_draws=0)


class TestDeterminism:
    def test_two_calls_agree(self, make_var_posterior):
        idata = make_var_posterior(bad_coord=(1, 0), divergences=3)
        first = convergence_report(idata, n_lags=1, var_names=["y1", "y2"])
        second = convergence_report(idata, n_lags=1, var_names=["y1", "y2"])
        assert first.to_dataframe().equals(second.to_dataframe())
        assert codes(first) == codes(second)
        assert first.status == second.status
        np.testing.assert_array_equal(first.stability.radius, second.stability.radius)


class TestUnsupportedPosteriors:
    def test_missing_posterior_group_raises(self):
        idata = InferenceData(prior=xr.Dataset({"B": (("chain", "draw"), np.zeros((2, 3)))}))
        with pytest.raises(ValueError, match="`posterior` group"):
            convergence_report(idata, n_lags=1)

    def test_missing_B_names_fitted_sv(self):
        posterior = xr.Dataset({"h": (("chain", "draw", "time"), np.zeros((2, 3, 4)))})
        with pytest.raises(ValueError, match="FittedSV"):
            convergence_report(InferenceData(posterior=posterior), n_lags=1)


class TestPipelineIntegration:
    @pytest.fixture
    def fitted(self, make_var_posterior, var_data_2v):
        return FittedVAR(
            idata=make_var_posterior(),
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )

    def test_fitted_var_method_delegates(self, fitted):
        report = fitted.convergence_report()
        assert isinstance(report, ConvergenceReport)
        assert report.status == "passed"
        assert report.blocks[0].max_rhat_coord.startswith("B[y")

    def test_fitted_var_method_forwards_options(self, fitted):
        report = fitted.convergence_report(
            thresholds=ConvergenceThresholds(rhat_warn=1.0, rhat_fail=1.0),
            hdi_prob=0.5,
            stability_draws=50,
        )
        assert report.status == "failed"
        assert report.stability.hdi_prob == 0.5
        assert report.stability.thinned_from == 200

    def test_identified_var_matches_fitted_var(self, fitted):
        identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        assert identified.convergence_report().to_dataframe().equals(fitted.convergence_report().to_dataframe())

    def test_legacy_shock_matrix_lands_in_identification_block(self, make_var_posterior, var_data_2v):
        fitted = FittedVAR(
            idata=make_var_posterior(extra_vars={"structural_shock_matrix": (2, 2)}),
            n_lags=1,
            data=var_data_2v,
            var_names=["y1", "y2"],
            volatility=Constant(),
        )
        report = fitted.convergence_report()
        assert blocks_by_name(report)["identification"].var_names == ["structural_shock_matrix"]

    def test_conjugate_fit_is_supported_with_honest_gaps(self, var_data_2v):
        fitted = ConjugateVAR(lags=1, prior=NIWPrior(), draws=1000, tune=20, seed=0).fit(var_data_2v)
        report = fitted.convergence_report()
        assert report.n_chains == 1
        assert report.max_rhat is None
        assert report.blocks[0].max_rhat_coord is None
        assert report.sampler_stats_available is False
        assert report.divergences is None
        assert {"single_chain", "sampler_stats_missing"} <= set(codes(report))
        # Coefficient and Cholesky draws are exact conditional draws, so their
        # effective sample size is nominal by construction.
        assert report.min_ess_bulk is not None
        assert report.min_ess_bulk > 400
        assert report.stability.n_vars == 2
        assert np.isfinite(report.stability.max_radius)
        # Nothing here is sampler pathology: no R-hat, no divergences, so the
        # single-chain note is the only thing keeping it off "passed".
        assert not any(message.severity == "failure" for message in report.messages)
        assert report.status == "warnings"

    def test_conjugate_posterior_labels_coordinates_without_coords(self, var_data_2v):
        fitted = ConjugateVAR(lags=1, prior=NIWPrior(), draws=1000, tune=20, seed=0).fit(var_data_2v)
        coord = fitted.convergence_report().blocks[0].min_ess_bulk_coord
        assert coord is not None
        assert coord.startswith("B[y")
        assert "L1." in coord


@pytest.mark.slow
class TestRealNUTSFit:
    def test_report_on_a_real_posterior(self, var_data_2v):
        from impulso.samplers import NUTSSampler
        from impulso.spec import VAR

        fitted = VAR(lags=1).fit(
            var_data_2v,
            sampler=NUTSSampler(draws=100, tune=100, chains=2, cores=1, random_seed=42, progressbar=False),
        )
        report = fitted.convergence_report()

        assert report.status in {"passed", "warnings", "failed"}
        assert isinstance(report.divergences, int)
        assert report.sampler_stats_available is True
        assert {"coefficient", "intercept", "covariance"} <= set(blocks_by_name(report))
        assert np.isfinite(report.stability.p_explosive)
        assert isinstance(report.stability, StabilitySummary)
        assert report.blocks[0].max_rhat_coord.startswith("B[y")
        assert "Convergence report" in report.summary()
