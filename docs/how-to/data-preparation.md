# Preparing Data for VARData

This guide shows how to construct a `VARData` container from common data formats.

## From a pandas DataFrame

The simplest path is `VARData.from_df`. Your DataFrame must have a `DatetimeIndex`:

```python
import pandas as pd
from impulso import VARData

df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])
```

## With exogenous variables

Pass column names for exogenous variables separately:

```python
data = VARData.from_df(
    df,
    endog=["gdp", "inflation", "rate"],
    exog=["oil_price"],
)
```

Trends, seasonal cycles and dated breaks do not need to come from a file —
build them from the index itself with a
[deterministic design](deterministic-regressors.md), which also produces the
matching future block for forecasting:

```python
from impulso import DeterministicDesign, Fourier, Trend

design = DeterministicDesign(terms=[Trend(degree=1, scale=120.0), Fourier(period=12, order=2)])
frame = pd.concat([df, design.build(df.index)], axis=1)
data = VARData.from_df(frame, endog=["gdp", "inflation", "rate"], exog=design.column_names)
```

## From NumPy arrays

If you already have arrays, pass them directly:

```python
import numpy as np

data = VARData(
    endog=endog_array,           # shape (T, n), n >= 2
    endog_names=["gdp", "inflation", "rate"],
    index=pd.date_range("2000-01-01", periods=T, freq="QS"),
)
```

## Validation rules

`VARData` enforces these constraints at construction time:

- At least 2 endogenous variables
- No `NaN` or `Inf` values
- `endog_names` length must match number of columns
- `index` length must match number of rows
- If `exog` is provided, `exog_names` is required (and vice versa)
- Every `exog` column must vary within the sample — a constant column is collinear
  with the intercept the VAR already includes, so its coefficient is not identified.
  Drop it, or encode a level shift as a dummy that changes value inside the sample.
- Arrays are copied and made read-only — the original data is never modified
