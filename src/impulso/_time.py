"""Calendar helpers for building forecast axes from observed indices."""

import pandas as pd


def infer_index_freq(index: pd.Index | None) -> pd.offsets.BaseOffset | None:
    """Detect the sampling frequency of an observed index.

    The index's own `freq` attribute wins; when it is unset, `pd.infer_freq`
    is consulted. Returns `None` for non-datetime indices, irregular dates,
    or samples too short to infer from (fewer than three observations).

    Args:
        index: Observed index, typically `VARData.index` / `SVData.index`.

    Returns:
        A pandas offset, or `None` when no frequency is detectable.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return None
    if index.freq is not None:
        return index.freq
    try:
        inferred = pd.infer_freq(index)
    except (TypeError, ValueError):
        # <3 observations, or a non-datetime-like index sneaking through.
        return None
    return pd.tseries.frequencies.to_offset(inferred) if inferred is not None else None


def forecast_index(index: pd.Index | None, steps: int) -> pd.Index:
    """Build the out-of-sample axis continuing `index` for `steps` periods.

    When a frequency is detectable on `index`, the axis is a `DatetimeIndex`
    named `time` starting one offset after the last observation. Otherwise it
    falls back to a step-numbered `RangeIndex` (`0..steps-1`, named `step`),
    which is the axis undated inputs have always produced.

    Args:
        index: Observed index, or `None` for a step-numbered axis.
        steps: Number of forecast steps.

    Returns:
        A `DatetimeIndex` continuing the calendar, or a `RangeIndex` of steps.
    """
    freq = infer_index_freq(index)
    if freq is None or index is None:
        return pd.RangeIndex(steps, name="step")
    return pd.date_range(start=index[-1] + freq, periods=steps, freq=freq, name="time")
