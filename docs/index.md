# Litterman

**Bayesian Vector Autoregression in Python.**

```python
import pandas as pd
from litterman import VARData

df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])
# Validated, immutable, ready for modeling
```

## Features

- **Validated data containers** — `VARData` catches shape mismatches, missing values, and type errors at construction time
- **Immutable types** — all objects are frozen after creation, preventing accidental mutation
- **Economist-friendly API** — think in variables and lags, not tensors and MCMC chains

## Installation

```bash
pip install litterman
```
