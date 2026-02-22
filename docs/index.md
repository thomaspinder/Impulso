# Litterman

**Bayesian Vector Autoregression in Python.**

```python
import pandas as pd
from litterman import VAR, VARData

df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])

fitted = VAR(lags="bic", prior="minnesota").fit(data)
forecast = fitted.forecast(steps=8)
forecast.median()  # point forecasts
forecast.hdi()     # credible intervals
```

## Features

- **Validated data containers** — `VARData` catches shape mismatches, missing values, and type errors at construction time
- **Immutable types** — all objects are frozen after creation, preventing accidental mutation
- **Economist-friendly API** — think in variables and lags, not tensors and MCMC chains
- **Minnesota prior** — smart defaults with tunable hyperparameters for shrinkage
- **Automatic lag selection** — AIC, BIC, and Hannan-Quinn criteria
- **PyMC backend** — full Bayesian estimation with NUTS sampling
- **Probabilistic forecasts** — posterior median, HDI credible intervals, tidy DataFrames

## Installation

```bash
pip install litterman
```
