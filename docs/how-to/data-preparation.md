# Preparing Data for VARData

This guide shows how to construct a `VARData` container from common data formats.

## From a pandas DataFrame

The simplest path is `VARData.from_df`. Your DataFrame must have a `DatetimeIndex`:

```python
import pandas as pd
from litterman import VARData

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
- Arrays are copied and made read-only — the original data is never modified
