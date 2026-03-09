# Impulso

**Bayesian Vector Autoregression in Python.**

=== "Fast (Conjugate)"

    ```python
    import pandas as pd
    from impulso import ConjugateVAR, VARData

    # Load data
    df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
    data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])

    # Estimate with data-driven prior selection (no MCMC needed)
    fitted = ConjugateVAR(lags=4, prior="minnesota_optimized").fit(data)

    # Forecast
    forecast = fitted.forecast(steps=8)
    forecast.median()  # point forecasts
    forecast.hdi()     # credible intervals
    ```

=== "Flexible (NUTS)"

    ```python
    import pandas as pd
    from impulso import VAR, VARData
    from impulso.identification import Cholesky

    # Load data
    df = pd.read_csv("macro_data.csv", index_col="date", parse_dates=True)
    data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])

    # Estimate with NUTS (supports arbitrary priors)
    fitted = VAR(lags="bic", prior="minnesota").fit(data)

    # Structural analysis
    identified = fitted.set_identification_strategy(
        Cholesky(ordering=["gdp", "inflation", "rate"])
    )
    irf = identified.impulse_response(horizon=20)
    irf.plot()
    ```

## Features

- **Validated data containers** — `VARData` catches shape mismatches, missing values, and type errors at construction time
- **Immutable pipeline** — `VARData` -> `VAR` -> `FittedVAR` -> `IdentifiedVAR`, each stage frozen after creation
- **Economist-friendly API** — think in variables and lags, not tensors and MCMC chains
- **Two estimation paths** — conjugate NIW sampling (instant, no MCMC) or full NUTS via PyMC (flexible, supports custom priors)
- **Data-driven prior selection** — automatic Minnesota hyperparameter optimisation via marginal likelihood (Giannone, Lenza & Primiceri, 2015)
- **Dummy observation priors** — encode persistence and unit root beliefs via sum-of-coefficients and single-unit-root priors
- **Minnesota prior** — smart defaults with tunable hyperparameters for shrinkage
- **Automatic lag selection** — AIC, BIC, and Hannan-Quinn criteria
- **Probabilistic forecasts** — posterior median, HDI credible intervals, tidy DataFrames
- **Conditional forecasting** — constrain future variable paths or structural shock paths for policy analysis
- **Structural identification** — Cholesky, sign restriction, and Blanchard-Quah long-run schemes
- **Built-in plotting** — IRF, FEVD, forecast, and historical decomposition plots

## Installation

```bash
pip install impulso
```

## Learn more

- [Fast estimation with ConjugateVAR](how-to/conjugate-estimation.md) — conjugate sampling, data-driven priors, dummy observations
- [Conditional forecasting](how-to/conditional-forecasting.md) — constrain future paths for policy analysis
- [Long-run restrictions](how-to/long-run-restrictions.md) — Blanchard-Quah structural identification
- [API Reference](reference/index.md) — complete module documentation
