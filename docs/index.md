# Impulso

**Bayesian Vector Autoregression in Python.**

```python
import pandas as pd
from impulso import VAR, VARData
from impulso.identification import Cholesky

# Load data
df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])

# Estimate
fitted = VAR(lags="bic", prior="minnesota").fit(data)

# Forecast
forecast = fitted.forecast(steps=8)
forecast.median()  # point forecasts
forecast.hdi()  # credible intervals

# Structural analysis
identified = fitted.set_identification_strategy(Cholesky(ordering=["gdp", "inflation", "rate"]))
irf = identified.impulse_response(horizon=20)
irf.plot()
```


<section class="consulting-cta">
    <p>We currently have some <strong>availability for consulting</strong> on how Bayesian modelling, vector autoregressions, and impulso can be integrated into your team's macroeconomic and financial forecasting work. If this sounds relevant, <a href="https://calendly.com/hello-1761-izqw/15-minute-meeting-clone-1">book an introductory call</a>. These calls are for consulting inquiries only. For technical usage questions and free community support, please use GitHub Discussions and the documentation below.</p>
</section>


## Features

- **Validated data containers** — `VARData` catches shape mismatches, missing values, and type errors at construction time
- **Immutable pipeline** — `VARData` -> `VAR` -> `FittedVAR` -> `IdentifiedVAR`, each stage frozen after creation
- **Economist-friendly API** — think in variables and lags, not tensors and MCMC chains
- **Minnesota prior** — smart defaults with tunable hyperparameters for shrinkage
- **Automatic lag selection** — AIC, BIC, and Hannan-Quinn criteria
- **PyMC backend** — full Bayesian estimation with NUTS sampling
- **Probabilistic forecasts** — posterior median, HDI credible intervals, tidy DataFrames
- **Structural identification** — Cholesky and sign restriction schemes
- **Built-in plotting** — IRF, FEVD, forecast, and historical decomposition plots

## Installation

```bash
pip install impulso
```

## Learn more

- [Quickstart tutorial](tutorials/quickstart.md) — fit your first Bayesian VAR
- [Forecasting tutorial](tutorials/forecasting.md) — produce probabilistic forecasts
- [Structural analysis tutorial](tutorials/structural-analysis.md) — impulse responses and variance decompositions
- [API Reference](reference/index.md) — complete module documentation
