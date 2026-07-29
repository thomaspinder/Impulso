"""Tests for VARData."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from impulso.data import VARData


class TestVARDataConstruction:
    def test_basic_construction(self, sample_endog, sample_index, endog_names):
        data = VARData(
            endog=sample_endog,
            endog_names=endog_names,
            index=sample_index,
        )
        assert data.endog.shape == (100, 3)
        assert data.exog is None
        assert data.exog_names is None
        assert len(data.endog_names) == 3

    def test_with_exog(self, sample_endog, sample_index, endog_names):
        rng = np.random.default_rng(42)
        exog = rng.standard_normal((100, 1))
        data = VARData(
            endog=sample_endog,
            endog_names=endog_names,
            exog=exog,
            exog_names=["oil_price"],
            index=sample_index,
        )
        assert data.exog is not None
        assert data.exog.shape == (100, 1)

    def test_frozen(self, sample_endog, sample_index, endog_names):
        data = VARData(
            endog=sample_endog,
            endog_names=endog_names,
            index=sample_index,
        )
        with pytest.raises(ValidationError):
            data.endog = np.zeros((100, 3))

    def test_arrays_not_writeable(self, sample_endog, sample_index, endog_names):
        data = VARData(
            endog=sample_endog,
            endog_names=endog_names,
            index=sample_index,
        )
        with pytest.raises(ValueError):
            data.endog[0, 0] = 999.0


class TestVARDataValidation:
    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite(self, sample_index, endog_names, bad_val):
        bad = np.full((100, 3), 1.0)
        bad[0, 2] = bad_val
        with pytest.raises(ValueError, match="NaN or Inf"):
            VARData(endog=bad, endog_names=endog_names, index=sample_index)

    def test_rejects_single_variable(self, sample_index):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Minimum 2"):
            VARData(
                endog=rng.standard_normal((100, 1)),
                endog_names=["gdp"],
                index=sample_index,
            )

    def test_rejects_mismatched_names(self, sample_endog, sample_index):
        with pytest.raises(ValueError, match="endog_names length"):
            VARData(endog=sample_endog, endog_names=["a", "b"], index=sample_index)

    def test_rejects_mismatched_index(self, sample_endog, endog_names):
        short_index = pd.date_range("2000-01-01", periods=50, freq="QS")
        with pytest.raises(ValueError, match="index length"):
            VARData(endog=sample_endog, endog_names=endog_names, index=short_index)

    def test_rejects_exog_names_without_exog(self, sample_endog, sample_index, endog_names):
        with pytest.raises(ValueError, match="exog_names provided without exog"):
            VARData(
                endog=sample_endog,
                endog_names=endog_names,
                exog_names=["oil"],
                index=sample_index,
            )

    def test_rejects_exog_without_names(self, sample_endog, sample_index, endog_names):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="exog_names required"):
            VARData(
                endog=sample_endog,
                endog_names=endog_names,
                exog=rng.standard_normal((100, 1)),
                index=sample_index,
            )


class TestVARDataExogVariation:
    """Exactly-constant exog columns are collinear with the intercept (#192)."""

    @pytest.mark.parametrize("fill", [1.0, 0.0, -3.5])
    def test_rejects_constant_exog_column(self, sample_endog, sample_index, endog_names, fill):
        with pytest.raises(ValueError, match=r"constant columns: 'level'"):
            VARData(
                endog=sample_endog,
                endog_names=endog_names,
                exog=np.full((100, 1), fill),
                exog_names=["level"],
                index=sample_index,
            )

    def test_error_names_every_constant_column_and_points_at_the_fix(
        self, sample_endog, sample_index, endog_names, rng
    ):
        exog = np.column_stack([np.ones(100), rng.standard_normal(100), np.zeros(100)])
        with pytest.raises(ValueError, match=r"constant columns: 'ones', 'zeros'") as exc:
            VARData(
                endog=sample_endog,
                endog_names=endog_names,
                exog=exog,
                exog_names=["ones", "oil", "zeros"],
                index=sample_index,
            )
        assert "collinear with the intercept" in str(exc.value)

    def test_accepts_column_that_varies_only_once(self, sample_endog, sample_index, endog_names):
        """A step dummy varies within the sample, so it is identified."""
        dummy = np.zeros(100)
        dummy[75:] = 1.0
        data = VARData(
            endog=sample_endog,
            endog_names=endog_names,
            exog=dummy[:, None],
            exog_names=["break"],
            index=sample_index,
        )
        assert data.exog is not None
        assert data.exog.shape == (100, 1)

    def test_accepts_near_constant_column(self, sample_endog, sample_index, endog_names, rng):
        """Only exactly-constant columns are rejected; tiny variation is legal."""
        col = 1.0 + 1e-13 * rng.standard_normal(100)
        data = VARData(
            endog=sample_endog,
            endog_names=endog_names,
            exog=col[:, None],
            exog_names=["almost_flat"],
            index=sample_index,
        )
        assert data.exog is not None

    def test_nonfinite_exog_still_reports_nan_not_constant(self, sample_endog, sample_index, endog_names):
        bad = np.full((100, 1), np.nan)
        with pytest.raises(ValueError, match="NaN or Inf"):
            VARData(
                endog=sample_endog,
                endog_names=endog_names,
                exog=bad,
                exog_names=["broken"],
                index=sample_index,
            )

    def test_from_df_rejects_constant_exog_column(self, rng):
        index = pd.date_range("2000-01-01", periods=100, freq="QS")
        df = pd.DataFrame(rng.standard_normal((100, 3)), columns=["gdp", "inflation", "rate"], index=index)
        df["const"] = 1.0
        with pytest.raises(ValueError, match=r"constant columns: 'const'"):
            VARData.from_df(df, endog=["gdp", "inflation", "rate"], exog=["const"])


class TestVARDataNameUniqueness:
    def test_rejects_duplicate_endog_names(self, sample_endog, sample_index):
        with pytest.raises(ValueError, match=r"endog_names must be unique, got duplicates: 'gdp'"):
            VARData(
                endog=sample_endog,
                endog_names=["gdp", "gdp", "rate"],
                index=sample_index,
            )

    def test_duplicate_endog_message_lists_every_duplicate(self, sample_index, rng):
        endog = rng.standard_normal((100, 4))
        with pytest.raises(ValueError, match=r"duplicates: 'gdp', 'rate'"):
            VARData(
                endog=endog,
                endog_names=["gdp", "rate", "gdp", "rate"],
                index=sample_index,
            )

    def test_rejects_duplicate_exog_names(self, sample_endog, sample_index, endog_names, rng):
        with pytest.raises(ValueError, match=r"exog_names must be unique, got duplicates: 'oil'"):
            VARData(
                endog=sample_endog,
                endog_names=endog_names,
                exog=rng.standard_normal((100, 2)),
                exog_names=["oil", "oil"],
                index=sample_index,
            )

    def test_rejects_endog_exog_overlap(self, sample_endog, sample_index, endog_names, rng):
        with pytest.raises(ValueError, match=r"endog_names and exog_names must not overlap, got shared names: 'gdp'"):
            VARData(
                endog=sample_endog,
                endog_names=endog_names,
                exog=rng.standard_normal((100, 1)),
                exog_names=[endog_names[0]],
                index=sample_index,
            )

    def test_accepts_unique_names(self, sample_endog, sample_index, endog_names, rng):
        data = VARData(
            endog=sample_endog,
            endog_names=endog_names,
            exog=rng.standard_normal((100, 2)),
            exog_names=["oil", "fx"],
            index=sample_index,
        )
        assert data.endog_names == endog_names
        assert data.exog_names == ["oil", "fx"]


class TestVARDataFromDFNameUniqueness:
    @staticmethod
    def _frame(columns: list[str], rng) -> pd.DataFrame:
        index = pd.date_range("2000-01-01", periods=100, freq="QS")
        return pd.DataFrame(
            rng.standard_normal((100, len(columns))),
            columns=columns,
            index=index,
        )

    def test_from_df_rejects_duplicated_endog_selection(self, rng):
        df = self._frame(["gdp", "inflation", "rate"], rng)
        with pytest.raises(ValueError, match=r"endog_names must be unique, got duplicates: 'gdp'"):
            VARData.from_df(df, endog=["gdp", "gdp", "rate"])

    def test_from_df_rejects_endog_exog_overlap(self, rng):
        df = self._frame(["gdp", "inflation", "rate"], rng)
        with pytest.raises(ValueError, match=r"must not overlap, got shared names: 'rate'"):
            VARData.from_df(df, endog=["gdp", "inflation", "rate"], exog=["rate"])

    def test_from_df_rejects_duplicate_dataframe_labels(self, rng):
        df = self._frame(["gdp", "gdp", "rate"], rng)
        with pytest.raises(ValueError, match=r"duplicate column labels for selected variables: 'gdp'"):
            VARData.from_df(df, endog=["gdp", "rate"])

    def test_from_df_ignores_duplicate_labels_outside_selection(self, rng):
        df = self._frame(["gdp", "inflation", "rate", "rate"], rng)
        data = VARData.from_df(df, endog=["gdp", "inflation"])
        assert data.endog.shape == (100, 2)


class TestVARDataFromDF:
    def test_from_df_endog_only(self):
        rng = np.random.default_rng(42)
        index = pd.date_range("2000-01-01", periods=100, freq="QS")
        df = pd.DataFrame(
            rng.standard_normal((100, 3)),
            columns=["gdp", "inflation", "rate"],
            index=index,
        )
        data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])
        assert data.endog.shape == (100, 3)
        assert data.endog_names == ["gdp", "inflation", "rate"]
        assert data.exog is None

    def test_from_df_with_exog(self):
        rng = np.random.default_rng(42)
        index = pd.date_range("2000-01-01", periods=100, freq="QS")
        df = pd.DataFrame(
            rng.standard_normal((100, 4)),
            columns=["gdp", "inflation", "rate", "oil"],
            index=index,
        )
        data = VARData.from_df(df, endog=["gdp", "inflation", "rate"], exog=["oil"])
        assert data.exog is not None
        assert data.exog.shape == (100, 1)
        assert data.exog_names == ["oil"]

    def test_from_df_requires_datetime_index(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(TypeError, match="DatetimeIndex"):
            VARData.from_df(df, endog=["a", "b"])
