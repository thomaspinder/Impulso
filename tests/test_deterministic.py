"""Tests for deterministic regressors (`impulso.deterministic`).

The headline property is *continuation*: the rows `extend` writes for the
future must be exactly the rows `build` would have written had the sample
run that much longer. Everything else — trend anchoring, harmonic phase,
dummy calendars, column order, dtypes, the future index — falls out of it,
so `TestContinuationProperty` carries most of the weight here.
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from numpy.testing import assert_allclose

from impulso.data import VARData
from impulso.deterministic import (
    BreakDummy,
    DeterministicDesign,
    Fourier,
    SeasonalDummies,
    Trend,
    _format_period,
)

# --------------------------------------------------------------------------- #
# Index fixtures
# --------------------------------------------------------------------------- #

_FREQ_SPECS = {
    "MS": ("1980-01-01", 540),
    "QS": ("1980-01-01", 240),
    "D": ("2010-01-01", 1200),
    "YS": ("1900-01-01", 150),
}

#: Reserved tail length: continuation cases build on `index[: T + h]`.
_TAIL = 25


def _index_for(freq_key: str) -> pd.DatetimeIndex:
    start, periods = _FREQ_SPECS[freq_key]
    return pd.date_range(start, periods=periods, freq=freq_key)


#: Per-frequency designs that are full rank in-sample, keyed by a test id.
_DESIGNS_BY_FREQ: dict[str, dict[str, list]] = {
    "MS": {
        "trend1": [Trend(degree=1, scale=120.0)],
        "trend3": [Trend(degree=3, scale=120.0)],
        "fourier": [Fourier(period=12, order=2)],
        "month": [SeasonalDummies(season="month")],
        "level": [BreakDummy(date="2000-01-01")],
        "pulse": [BreakDummy(date="2000-01-01", kind="pulse")],
        "composed": [
            Trend(degree=1, scale=120.0),
            Fourier(period=12, order=2),
            SeasonalDummies(season="quarter"),
            BreakDummy(date="2000-01-01"),
        ],
    },
    "QS": {
        "trend2": [Trend(degree=2, scale=40.0)],
        "fourier": [Fourier(period=4, order=1)],
        "quarter": [SeasonalDummies(season="quarter")],
        "pulse": [BreakDummy(date="2000-01-01", kind="pulse")],
        "composed": [
            Trend(degree=1, scale=40.0),
            SeasonalDummies(season="quarter"),
            BreakDummy(date="2000-01-01"),
        ],
    },
    "D": {
        "trend1": [Trend(degree=1, scale=365.0)],
        "fourier": [Fourier(period=365.25, order=2)],
        "dayofweek": [SeasonalDummies(season="dayofweek")],
        "level": [BreakDummy(date="2012-01-01")],
        "composed": [
            Trend(degree=1, scale=365.0),
            Fourier(period=365.25, order=2),
            SeasonalDummies(season="dayofweek"),
            BreakDummy(date="2012-01-01"),
        ],
    },
    "YS": {
        "trend1": [Trend(degree=1, scale=10.0)],
        "fourier": [Fourier(period=11, order=2)],
        "level": [BreakDummy(date="1950-01-01")],
        "composed": [Trend(degree=1, scale=10.0), BreakDummy(date="1950-01-01")],
    },
}


def _continuation_cases():
    for freq_key, designs in _DESIGNS_BY_FREQ.items():
        for design_id, terms in designs.items():
            for h in (1, 3, 13, 25):
                yield pytest.param(freq_key, terms, h, id=f"{freq_key}-{design_id}-h{h}")


@pytest.fixture
def monthly_index():
    return _index_for("MS")


@pytest.fixture
def quarterly_index():
    return _index_for("QS")


@pytest.fixture
def daily_index():
    return _index_for("D")


@pytest.fixture
def annual_index():
    return _index_for("YS")


@pytest.fixture
def gappy_monthly_index(monthly_index):
    """Monthly index with a six-month hole and therefore no inferable freq."""
    return monthly_index.delete(range(36, 42))


@pytest.fixture
def simple_design():
    """All four term types, jointly full rank on a monthly index."""
    return DeterministicDesign(
        terms=[
            Trend(degree=1, scale=120.0),
            Fourier(period=12, order=2),
            SeasonalDummies(season="quarter"),
            BreakDummy(date="2000-01-01"),
        ]
    )


# --------------------------------------------------------------------------- #
# A. The continuation property
# --------------------------------------------------------------------------- #


class TestContinuationProperty:
    """`extend` must reproduce the rows `build` writes on a longer index."""

    @pytest.mark.parametrize(("freq_key", "terms", "h"), _continuation_cases())
    def test_extend_matches_build_on_longer_index(self, freq_key, terms, h):
        index = _index_for(freq_key)
        design = DeterministicDesign(terms=terms)
        cut = len(index) - _TAIL

        expected = design.build(index[: cut + h]).iloc[cut:]
        actual = design.extend(index[:cut], h)

        pdt.assert_frame_equal(expected, actual)

    @pytest.mark.parametrize(("freq_key", "terms", "h"), _continuation_cases())
    def test_future_index_matches_extend_index(self, freq_key, terms, h):
        index = _index_for(freq_key)
        design = DeterministicDesign(terms=terms)
        cut = len(index) - _TAIL

        assert design.extend(index[:cut], h).index.equals(design.future_index(index[:cut], h))

    def test_columns_and_dtypes_are_stable(self, monthly_index, simple_design):
        built = simple_design.build(monthly_index[:400])
        extended = simple_design.extend(monthly_index[:400], 12)

        assert list(built.columns) == simple_design.column_names
        assert list(extended.columns) == simple_design.column_names
        assert (built.dtypes == np.float64).all()
        assert (extended.dtypes == np.float64).all()

    def test_extend_rejects_nonpositive_steps(self, monthly_index, simple_design):
        with pytest.raises(ValueError, match="steps must be >= 1"):
            simple_design.extend(monthly_index, 0)
        with pytest.raises(ValueError, match="steps must be >= 1"):
            simple_design.future_index(monthly_index, -3)


# --------------------------------------------------------------------------- #
# B. Slicing invariance (and the deliberate exception)
# --------------------------------------------------------------------------- #


class TestSlicingInvariance:
    """Calendar terms ignore where the sample starts; trends deliberately do not."""

    @pytest.mark.parametrize(
        ("term_id", "term"),
        [
            ("fourier", Fourier(period=12, order=2)),
            ("month", SeasonalDummies(season="month")),
            ("level", BreakDummy(date="2000-01-01")),
            ("pulse", BreakDummy(date="2000-01-01", kind="pulse")),
        ],
    )
    def test_calendar_terms_are_slice_invariant(self, monthly_index, term_id, term):
        design = DeterministicDesign(terms=[term])
        cut = 60

        pdt.assert_frame_equal(design.build(monthly_index).iloc[cut:], design.build(monthly_index[cut:]))

    def test_trend_shifts_by_the_exact_offset(self, monthly_index):
        design = DeterministicDesign(terms=[Trend(degree=1, scale=1.0)])
        cut = 60

        on_full = design.build(monthly_index).iloc[cut:]["trend"].to_numpy()
        on_slice = design.build(monthly_index[cut:])["trend"].to_numpy()

        # The origin moved with the sample start, so the two differ by exactly
        # the number of periods dropped — an affine shift, not a bug.
        assert not np.array_equal(on_full, on_slice)
        assert_allclose(on_full - on_slice, float(cut))


# --------------------------------------------------------------------------- #
# C. Column-name contract
# --------------------------------------------------------------------------- #


class TestColumnNames:
    def test_trend_names(self):
        assert Trend(degree=1).column_names == ["trend"]
        assert Trend(degree=3).column_names == ["trend", "trend_squared", "trend_cubed"]

    def test_fourier_names(self):
        assert Fourier(period=12, order=2).column_names == [
            "sin(1,12)",
            "cos(1,12)",
            "sin(2,12)",
            "cos(2,12)",
        ]
        assert Fourier(period=365.25, order=1).column_names == ["sin(1,365.25)", "cos(1,365.25)"]

    @pytest.mark.parametrize(
        ("period", "expected"),
        [(12, "12"), (12.0, "12"), (365.25, "365.25"), (4, "4"), (52.18, "52.18")],
    )
    def test_format_period(self, period, expected):
        assert _format_period(period) == expected

    def test_seasonal_names_drop_the_reference_level(self):
        assert SeasonalDummies(season="quarter").column_names == ["quarter_2", "quarter_3", "quarter_4"]
        assert SeasonalDummies(season="quarter", reference=3).column_names == [
            "quarter_1",
            "quarter_2",
            "quarter_4",
        ]
        assert SeasonalDummies(season="dayofweek").column_names == [f"dow_{i}" for i in range(1, 7)]
        assert len(SeasonalDummies(season="month").column_names) == 11
        assert SeasonalDummies(season="month", drop_first=False).column_names == [f"month_{i}" for i in range(1, 13)]

    def test_break_names_use_the_resolved_timestamp(self):
        assert BreakDummy(date="2000-01-01").column_names == ["level_2000-01-01"]
        assert BreakDummy(date=pd.Timestamp("2000-03-15"), kind="pulse").column_names == ["pulse_2000-03-15"]

    def test_design_column_names_concatenate_in_term_order(self, simple_design):
        assert simple_design.column_names == [
            "trend",
            "sin(1,12)",
            "cos(1,12)",
            "sin(2,12)",
            "cos(2,12)",
            "quarter_2",
            "quarter_3",
            "quarter_4",
            "level_2000-01-01",
        ]

    def test_duplicate_column_names_are_rejected(self):
        with pytest.raises(ValueError, match="Duplicate column name 'trend'"):
            DeterministicDesign(terms=[Trend(degree=1), Trend(degree=1, scale=12.0)])

    def test_empty_design_is_rejected(self):
        with pytest.raises(ValueError, match="requires at least one term"):
            DeterministicDesign(terms=[])


# --------------------------------------------------------------------------- #
# D. Validation errors
# --------------------------------------------------------------------------- #


class TestConstructionValidation:
    @pytest.mark.parametrize("degree", [0, 4])
    def test_trend_degree_bounds(self, degree):
        with pytest.raises(ValueError, match="degree"):
            Trend(degree=degree)

    def test_trend_scale_must_be_positive(self):
        with pytest.raises(ValueError, match="scale"):
            Trend(degree=1, scale=0.0)

    def test_fourier_nyquist_limit(self):
        with pytest.raises(ValueError, match="Nyquist"):
            Fourier(period=12, order=7)

    def test_fourier_period_must_exceed_one(self):
        with pytest.raises(ValueError, match="period"):
            Fourier(period=1, order=1)

    def test_seasonal_reference_must_be_a_level(self):
        with pytest.raises(ValueError, match="not a valid month level"):
            SeasonalDummies(season="month", reference=13)

    def test_seasonal_reference_requires_drop_first(self):
        with pytest.raises(ValueError, match="only meaningful with drop_first=True"):
            SeasonalDummies(season="quarter", drop_first=False, reference=2)

    def test_unknown_season_rejected(self):
        with pytest.raises(ValueError, match="season"):
            SeasonalDummies(season="dayofyear")


class TestBuildValidation:
    def test_pulse_not_on_the_index_names_its_neighbours(self, monthly_index):
        design = DeterministicDesign(terms=[BreakDummy(date="1990-01-15", kind="pulse")])
        with pytest.raises(ValueError, match=r"1990-01-01 \(before\) and 1990-02-01 \(after\)"):
            design.build(monthly_index)

    def test_level_break_at_sample_start_is_rejected(self, monthly_index):
        design = DeterministicDesign(terms=[BreakDummy(date="1980-01-01")])
        with pytest.raises(ValueError, match="collinear with the intercept"):
            design.build(monthly_index)

    def test_level_break_after_sample_end_is_rejected(self, monthly_index):
        design = DeterministicDesign(terms=[BreakDummy(date="2050-01-01")])
        with pytest.raises(ValueError, match="never occurs in-sample"):
            design.build(monthly_index)

    def test_full_dummy_set_is_collinear_with_the_intercept(self, monthly_index):
        design = DeterministicDesign(terms=[SeasonalDummies(season="month", drop_first=False)])
        with pytest.raises(ValueError, match="drop_first=False"):
            design.build(monthly_index)

    def test_degenerate_top_harmonic_is_rejected(self, monthly_index):
        design = DeterministicDesign(terms=[Fourier(period=12, order=6)])
        with pytest.raises(ValueError, match="is identically"):
            design.build(monthly_index)

    def test_dummies_and_harmonics_of_the_same_cycle_clash(self, monthly_index):
        design = DeterministicDesign(terms=[SeasonalDummies(season="month"), Fourier(period=12, order=2)])
        with pytest.raises(ValueError, match="describe the same cycle"):
            design.build(monthly_index)

    def test_too_few_observations(self, monthly_index):
        design = DeterministicDesign(terms=[SeasonalDummies(season="month")])
        with pytest.raises(ValueError, match="Too few observations"):
            design.build(monthly_index[:8])

    def test_index_must_be_a_datetime_index(self, simple_design):
        with pytest.raises(TypeError, match="must be a pandas DatetimeIndex"):
            simple_design.build(pd.RangeIndex(10))

    def test_index_must_be_strictly_increasing(self, simple_design, monthly_index):
        shuffled = monthly_index[::-1]
        with pytest.raises(ValueError, match="strictly increasing"):
            simple_design.build(shuffled)

    def test_empty_index_rejected(self, simple_design, monthly_index):
        with pytest.raises(ValueError, match="must not be empty"):
            simple_design.build(monthly_index[:0])

    def test_break_date_of_an_unsupported_type_is_rejected(self):
        with pytest.raises(ValueError, match="date"):
            BreakDummy(date=12345)

    def test_a_term_whose_width_contradicts_its_names_is_caught(self, monthly_index):
        class BadTerm:
            """A custom term that promises two columns and delivers one."""

            @property
            def column_names(self):
                return ["a", "b"]

            def build(self, index, origin, alias):
                return np.zeros((len(index), 1))

        design = DeterministicDesign(terms=[BadTerm()])
        with pytest.raises(ValueError, match=r"BadTerm.build returned shape"):
            design.build(monthly_index)


# --------------------------------------------------------------------------- #
# E. Frequency resolution
# --------------------------------------------------------------------------- #


class TestFrequencyResolution:
    def test_explicit_freq_wins_over_a_gappy_index(self, gappy_monthly_index):
        design = DeterministicDesign(terms=[Trend(degree=1)], freq="MS")
        assert design.build(gappy_monthly_index).shape == (len(gappy_monthly_index), 1)

    def test_index_freq_is_used_when_present(self, monthly_index):
        assert monthly_index.freq is not None
        design = DeterministicDesign(terms=[Trend(degree=1)])
        assert_allclose(design.build(monthly_index)["trend"].to_numpy()[:4], [0, 1, 2, 3])

    def test_inference_accepted_when_it_regenerates_the_index(self, monthly_index):
        stripped = pd.DatetimeIndex(list(monthly_index))
        assert stripped.freq is None
        design = DeterministicDesign(terms=[Trend(degree=1)])
        assert_allclose(design.build(stripped)["trend"].to_numpy()[:4], [0, 1, 2, 3])

    def test_false_positive_inference_is_rejected(self):
        # pandas confidently infers WOM-1SAT here; it has no period equivalent,
        # so the design refuses rather than silently anchoring to nonsense.
        irregular = pd.DatetimeIndex(["2020-01-04", "2020-02-01", "2020-03-07"])
        assert pd.infer_freq(irregular) == "WOM-1SAT"
        design = DeterministicDesign(terms=[Trend(degree=1)])
        with pytest.raises(ValueError, match="no pandas period equivalent"):
            design.build(irregular)

    def test_inference_that_does_not_regenerate_the_index_is_rejected(self, monkeypatch, monthly_index):
        # `pd.infer_freq` is confident on short irregular indices, so the
        # candidate must reproduce the index before it is trusted. Forcing a
        # wrong-but-valid answer exercises that guard directly.
        stripped = pd.DatetimeIndex(list(monthly_index))
        monkeypatch.setattr(pd, "infer_freq", lambda index: "QS")
        design = DeterministicDesign(terms=[Trend(degree=1)])

        with pytest.raises(ValueError, match="does not reproduce it"):
            design.build(stripped)

    def test_unresolvable_frequency_errors(self):
        irregular = pd.DatetimeIndex(["2020-01-01", "2020-01-05", "2020-03-17", "2020-09-02"])
        design = DeterministicDesign(terms=[Trend(degree=1)])
        with pytest.raises(ValueError, match="Could not determine the sampling frequency"):
            design.build(irregular)

    def test_business_day_frequency_names_the_alternative(self):
        business = pd.date_range("2020-01-01", periods=60, freq="B")
        design = DeterministicDesign(terms=[Trend(degree=1)])
        with pytest.raises(ValueError, match=r'Business-day frequencies.*freq="D"'):
            design.build(business)

    @pytest.mark.parametrize("freq", ["MS", "ME", "QS", "QE", "YS", "YE", "D", "W", "h", "15D", "2h", "15min"])
    def test_extend_walks_the_sampling_offset(self, freq):
        index = pd.date_range("2000-01-03", periods=60, freq=freq)
        design = DeterministicDesign(terms=[Trend(degree=1)])

        expected_index = pd.date_range(index[-1], periods=6, freq=freq)[1:]
        extended = design.extend(index, 5)

        assert extended.index.equals(expected_index)
        assert design.future_index(index, 5).equals(expected_index)
        last = design.build(index)["trend"].to_numpy()[-1]
        assert_allclose(extended["trend"].to_numpy(), last + np.arange(1.0, 6.0))

    @pytest.mark.parametrize("freq", ["15D", "2h", "15min"])
    def test_multiplied_offsets_count_in_sampling_periods(self, freq):
        # pandas stores 15D ordinals in days and 2h ordinals in hours. Elapsed
        # time must still advance by one per observation, or `Fourier.period`
        # would silently mean something other than "cycle length in sampling
        # periods" — a 12-observation cycle on 2-hourly data is 24 hours.
        index = pd.date_range("2000-01-03", periods=60, freq=freq)
        design = DeterministicDesign(terms=[Trend(degree=1)])

        assert_allclose(design.build(index)["trend"].to_numpy(), np.arange(60.0))

    @pytest.mark.parametrize("freq", ["15D", "2h", "15min"])
    @pytest.mark.parametrize("h", [1, 3, 13])
    def test_continuation_holds_for_multiplied_offsets(self, freq, h):
        index = pd.date_range("2000-01-03", periods=60, freq=freq)
        design = DeterministicDesign(
            terms=[Trend(degree=1, scale=12.0), Fourier(period=12, order=2)],
        )
        cut = len(index) - 13

        pdt.assert_frame_equal(design.build(index[: cut + h]).iloc[cut:], design.extend(index[:cut], h))

    def test_extend_does_not_skip_a_period_off_anchor(self):
        # `pd.date_range` rolls an off-anchor start forward, so the walk's
        # first entry is already April here. Dropping it would forecast from
        # May and silently lose a month.
        index = pd.DatetimeIndex(["2000-01-01", "2000-02-01", "2000-03-15"])
        design = DeterministicDesign(terms=[Trend(degree=1)], freq="MS")

        future = design.future_index(index, 3)

        assert list(future.strftime("%Y-%m-%d")) == ["2000-04-01", "2000-05-01", "2000-06-01"]
        assert design.extend(index, 3).index.equals(future)
        # The trend keeps counting calendar months from the origin.
        assert_allclose(design.extend(index, 3)["trend"].to_numpy(), [3.0, 4.0, 5.0])

    def test_on_anchor_extend_is_unchanged(self):
        index = pd.date_range("2000-01-01", periods=3, freq="MS")
        design = DeterministicDesign(terms=[Trend(degree=1)])

        assert list(design.future_index(index, 3).strftime("%Y-%m-%d")) == [
            "2000-04-01",
            "2000-05-01",
            "2000-06-01",
        ]

    def test_future_index_override_is_honoured(self, monthly_index):
        design = DeterministicDesign(terms=[Trend(degree=1)])
        override = pd.DatetimeIndex(["2030-01-01", "2030-06-01", "2031-01-01"])

        extended = design.extend(monthly_index, 3, future_index=override)

        assert extended.index.equals(override)
        # Anchored to the estimation origin, so elapsed months are absolute.
        assert_allclose(extended["trend"].to_numpy(), [600.0, 605.0, 612.0])

    def test_future_index_length_must_match_steps(self, monthly_index):
        design = DeterministicDesign(terms=[Trend(degree=1)])
        override = pd.DatetimeIndex(["2030-01-01", "2030-02-01"])
        with pytest.raises(ValueError, match="future_index has length 2, but steps=3"):
            design.extend(monthly_index, 3, future_index=override)


# --------------------------------------------------------------------------- #
# F. Gaps
# --------------------------------------------------------------------------- #


class TestGaps:
    def test_trend_jumps_across_a_gap(self, monthly_index):
        gappy = monthly_index.delete(range(3, 6))
        design = DeterministicDesign(terms=[Trend(degree=1)], freq="MS")

        trend = design.build(gappy)["trend"].to_numpy()

        assert_allclose(trend[:8], [0, 1, 2, 6, 7, 8, 9, 10])

    def test_dummies_stay_calendar_correct_across_a_gap(self, gappy_monthly_index):
        design = DeterministicDesign(terms=[SeasonalDummies(season="month")], freq="MS")

        frame = design.build(gappy_monthly_index)

        months = gappy_monthly_index.month
        for position, month in enumerate(months[:60]):
            row = frame.iloc[position]
            if month == 1:  # the dropped reference level
                assert row.sum() == 0.0
            else:
                assert row[f"month_{month}"] == 1.0
                assert row.sum() == 1.0

    def test_continuation_holds_across_a_gap(self, gappy_monthly_index):
        design = DeterministicDesign(
            terms=[Trend(degree=1, scale=12.0), Fourier(period=12, order=2)],
            freq="MS",
        )
        cut = len(gappy_monthly_index) - 12

        expected = design.build(gappy_monthly_index).iloc[cut:]
        actual = design.extend(gappy_monthly_index[:cut], 12)

        assert_allclose(actual.to_numpy(), expected.to_numpy())


# --------------------------------------------------------------------------- #
# G. Finiteness
# --------------------------------------------------------------------------- #


class TestFiniteness:
    """Terms are total functions of a timestamp — they cannot emit NaN."""

    @pytest.mark.parametrize(("freq_key", "terms", "h"), _continuation_cases())
    def test_no_missing_values_anywhere(self, freq_key, terms, h):
        index = _index_for(freq_key)
        design = DeterministicDesign(terms=terms)
        cut = len(index) - _TAIL

        assert np.isfinite(design.build(index[:cut]).to_numpy()).all()
        assert np.isfinite(design.extend(index[:cut], h).to_numpy()).all()


# --------------------------------------------------------------------------- #
# H. VARData round trip
# --------------------------------------------------------------------------- #


def _endog_frame(index: pd.DatetimeIndex, n_vars: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.standard_normal((len(index), n_vars)),
        index=index,
        columns=[f"y{i + 1}" for i in range(n_vars)],
    )


class TestVARDataRoundTrip:
    def test_design_columns_become_exog_names(self, monthly_index, simple_design):
        endog = _endog_frame(monthly_index)
        frame = pd.concat([endog, simple_design.build(monthly_index)], axis=1)

        data = VARData.from_df(frame, endog=list(endog.columns), exog=simple_design.column_names)

        assert data.exog_names == simple_design.column_names
        assert data.exog.shape == (len(monthly_index), len(simple_design.column_names))
        assert data.exog.flags.writeable is False

    def test_misaligned_index_surfaces_as_the_nan_invariant(self, monthly_index, simple_design):
        endog = _endog_frame(monthly_index)
        # The documented failure mode: the design was built on the index that
        # survived a transform, but concatenated against the untrimmed endog.
        # Deterministic terms cannot themselves emit NaN, so any NaN in the
        # exog block is misalignment — and VARData catches it at construction.
        frame = pd.concat([endog, simple_design.build(monthly_index[3:])], axis=1)

        with pytest.raises(ValueError, match="exog contains NaN or Inf values"):
            VARData.from_df(frame, endog=list(endog.columns), exog=simple_design.column_names)


# --------------------------------------------------------------------------- #
# I. exog_future
# --------------------------------------------------------------------------- #


def _data_with_design(index, design, order=None):
    endog = _endog_frame(index)
    frame = pd.concat([endog, design.build(index)], axis=1)
    columns = design.column_names if order is None else order
    return VARData.from_df(frame, endog=list(endog.columns), exog=columns)


class TestExogFuture:
    def test_shape_and_dtype(self, monthly_index, simple_design):
        data = _data_with_design(monthly_index, simple_design)

        block = simple_design.exog_future(data, 6)

        assert block.shape == (6, len(simple_design.column_names))
        assert block.dtype == np.float64
        assert_allclose(block, simple_design.extend(monthly_index, 6).to_numpy())

    def test_columns_are_reordered_to_match_exog_names(self, monthly_index):
        design = DeterministicDesign(terms=[Trend(degree=1, scale=120.0), Fourier(period=12, order=2)])
        permuted = ["cos(1,12)", "trend", "sin(2,12)", "sin(1,12)", "cos(2,12)"]
        data = _data_with_design(monthly_index, design, order=permuted)
        assert data.exog_names == permuted

        block = design.exog_future(data, 4)

        assert_allclose(block, design.extend(monthly_index, 4)[permuted].to_numpy())
        # Positional forecasting would otherwise silently use the wrong column.
        assert not np.allclose(block, design.extend(monthly_index, 4).to_numpy())

    def test_name_mismatch_names_both_sets(self, monthly_index, simple_design):
        data = _data_with_design(monthly_index, simple_design)
        other = DeterministicDesign(terms=[Trend(degree=1, scale=120.0), Fourier(period=12, order=1)])

        with pytest.raises(ValueError, match="does not match the fitted exogenous block"):
            other.exog_future(data, 4)

    def test_data_without_exog_errors(self, monthly_index, simple_design):
        endog = _endog_frame(monthly_index)
        data = VARData.from_df(endog, endog=list(endog.columns))

        with pytest.raises(ValueError, match="fitted without exogenous regressors"):
            simple_design.exog_future(data, 4)

    def test_accepts_a_fitted_var(self, monthly_index, simple_design):
        fitted = _fitted_with_exog(monthly_index, simple_design)

        from_fitted = simple_design.exog_future(fitted, 5)
        from_data = simple_design.exog_future(fitted.data, 5)

        assert_allclose(from_fitted, from_data)

    def test_future_index_override_flows_through(self, monthly_index, simple_design):
        data = _data_with_design(monthly_index, simple_design)
        override = simple_design.future_index(monthly_index, 3)

        assert_allclose(
            simple_design.exog_future(data, 3, future_index=override),
            simple_design.exog_future(data, 3),
        )


# --------------------------------------------------------------------------- #
# J. Fast end-to-end (synthetic posterior, no MCMC)
# --------------------------------------------------------------------------- #


def _fitted_with_exog(index, design, n_lags: int = 1):
    """A FittedVAR over a synthetic posterior that carries `B_exog`."""
    import arviz as az
    import xarray as xr

    from impulso.fitted import FittedVAR
    from impulso.volatility import Constant

    data = _data_with_design(index, design)
    n_chains, n_draws, n_vars = 2, 20, 2
    n_exog = len(design.column_names)
    rng = np.random.default_rng(11)

    L = np.broadcast_to(np.eye(n_vars) * 0.2, (n_chains, n_draws, n_vars, n_vars)).copy()
    posterior = xr.Dataset({
        "B": xr.DataArray(
            rng.standard_normal((n_chains, n_draws, n_vars, n_vars * n_lags)) * 0.2,
            dims=["chain", "draw", "var", "coeff"],
        ),
        "intercept": xr.DataArray(
            rng.standard_normal((n_chains, n_draws, n_vars)) * 0.01,
            dims=["chain", "draw", "var"],
        ),
        "B_exog": xr.DataArray(
            rng.standard_normal((n_chains, n_draws, n_vars, n_exog)),
            dims=["chain", "draw", "var", "exog"],
            coords={"exog": design.column_names},
        ),
        "L": xr.DataArray(L, dims=["chain", "draw", "var1", "var2"]),
    })
    return FittedVAR.model_construct(
        idata=az.InferenceData(posterior=posterior),
        n_lags=n_lags,
        data=data,
        var_names=data.endog_names,
        volatility=Constant(),
        pymc_model=None,
    )


class TestForecastIntegrationFast:
    def test_forecast_consumes_the_generated_block(self, monthly_index, simple_design):
        fitted = _fitted_with_exog(monthly_index, simple_design)

        block = simple_design.exog_future(fitted, 6)
        forecast = fitted.forecast(steps=6, exog_future=block, include_shock_uncertainty=False)

        assert forecast.median().shape == (6, 2)

        zeroed = fitted.forecast(
            steps=6,
            exog_future=np.zeros_like(block),
            include_shock_uncertainty=False,
        )
        # If B_exog were ignored the two would coincide; the design must bite.
        assert not np.allclose(forecast.median().to_numpy(), zeroed.median().to_numpy())

    def test_posterior_exog_coord_matches_the_design(self, monthly_index, simple_design):
        fitted = _fitted_with_exog(monthly_index, simple_design)

        coord = list(fitted.idata.posterior["B_exog"].coords["exog"].values)

        assert coord == simple_design.column_names


# --------------------------------------------------------------------------- #
# K. Estimator boundary
# --------------------------------------------------------------------------- #


class TestConjugateEstimatorBoundary:
    def test_conjugate_var_rejects_a_deterministic_design(self, monthly_index, simple_design):
        from impulso.conjugate import ConjugateVAR
        from impulso.priors import NIWPrior

        data = _data_with_design(monthly_index, simple_design)
        estimator = ConjugateVAR(lags=1, prior=NIWPrior(), draws=2, tune=0, seed=0)

        with pytest.raises(ValueError, match="does not support exogenous regressors"):
            estimator.fit(data)


# --------------------------------------------------------------------------- #
# L. Slow integration — the onboarding recipe, end to end
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_deterministic_design_end_to_end():
    """The documented recipe, exercised against real MCMC.

    The body of this test is reproduced verbatim in
    `docs/how-to/deterministic-regressors.md`.
    """
    import numpy as np
    import pandas as pd

    from impulso import VAR, DeterministicDesign, Fourier, NUTSSampler, Trend, VARData

    # A short monthly two-variable sample standing in for climate anomalies.
    rng = np.random.default_rng(7)
    index = pd.date_range("2000-01-01", periods=120, freq="MS")
    endog = pd.DataFrame(
        rng.standard_normal((len(index), 2)).cumsum(axis=0) * 0.1,
        index=index,
        columns=["temperature", "precipitation"],
    )

    # One design, used for estimation and for forecasting.
    design = DeterministicDesign(
        terms=[Trend(degree=1, scale=120.0), Fourier(period=12, order=1)],
        freq="MS",
    )

    frame = pd.concat([endog, design.build(index)], axis=1)
    data = VARData.from_df(frame, endog=list(endog.columns), exog=design.column_names)

    fitted = VAR(lags=1).fit(data, sampler=NUTSSampler(draws=50, tune=50, chains=2, cores=1, random_seed=42))

    # The posterior labels B_exog with the design's own column names.
    assert list(fitted.idata.posterior["B_exog"].coords["exog"].values) == design.column_names

    forecast = fitted.forecast(steps=12, exog_future=design.exog_future(fitted, 12))

    assert forecast.median().shape == (12, 2)
