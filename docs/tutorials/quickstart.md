# Fitting Your First Bayesian VAR


A vector autoregression (VAR) models multiple time series jointly, capturing how each variable depends on its own past values and the past values of all other variables in the system. This makes it a natural tool for macroeconomic analysis, where GDP, inflation, and interest rates evolve together. Bayesian estimation adds regularisation through prior distributions – critical when the number of parameters grows quickly with lags and variables – and provides full posterior uncertainty over every coefficient and forecast. For more background, see the [Bayesian VAR explanation](../explanation/bayesian-var.md).

``` python
import arviz as az
import numpy as np
import pandas as pd

from impulso import VAR, VARData, select_lag_order
from impulso.samplers import NUTSSampler
```

## Simulate a small macro economy

We simulate a VAR(1) with three variables – quarterly GDP growth, inflation, and a short-term interest rate – so that we know the true coefficients and can check whether the model recovers them.

The true coefficient matrix `A_true` embeds a simple macroeconomic story:

| Variable | Own lag | Cross-variable effects |
|----|----|----|
| **GDP growth** | Persistent at 0.6 | Reacts negatively to last quarter’s interest rate (-0.1) |
| **Inflation** | Persistent at 0.5 | Follows GDP with a one-quarter lag (0.2) |
| **Interest rate** | Persistent at 0.4 | Reacts to inflation (0.15) – a simplified Taylor-rule channel |

``` python
rng = np.random.default_rng(42)
T = 200
n_vars = 3

# True VAR(1) coefficient matrix
# Rows: [gdp_growth, inflation, rate]
# Columns: [gdp_growth_lag1, inflation_lag1, rate_lag1]
A_true = np.array([
    [0.6, 0.0, -0.1],  # GDP: persistent, negatively affected by rate
    [0.2, 0.5, 0.0],  # Inflation: follows GDP, persistent
    [0.0, 0.15, 0.4],  # Rate: reacts to inflation (Taylor rule), persistent
])

y = np.zeros((T, n_vars))
for t in range(1, T):
    y[t] = A_true @ y[t - 1] + rng.standard_normal(n_vars) * 0.1
```

## Load data into VARData

`VARData` is the entry point for all data in Impulso. It validates the shape of your arrays, checks for NaN and Inf values, and makes the underlying arrays read-only so that data cannot be accidentally mutated after construction.

> [!TIP]
>
> ### Already have a DataFrame?
>
> Use `VARData.from_df(df, endog=["col1", "col2"])` to build the dataset directly from a pandas DataFrame without constructing arrays manually.

``` python
index = pd.date_range("2000-01-01", periods=T, freq="QS")
data = VARData(endog=y, endog_names=["gdp_growth", "inflation", "rate"], index=index)
```

## Select lag order

Before estimating the Bayesian VAR, we need to choose the number of lags. `select_lag_order` fits OLS VARs at each candidate lag length and computes three information criteria:

- **AIC** (Akaike): tends to overfit by selecting too many lags.
- **BIC** (Bayesian / Schwarz): penalises complexity more heavily and is conservative.
- **HQ** (Hannan-Quinn): sits between AIC and BIC in penalisation strength.

Since the true DGP is VAR(1), BIC should recover lag 1.

``` python
ic = select_lag_order(data, max_lags=8)
print(
    f"AIC selects {ic.aic} lag(s), BIC selects {ic.bic} lag(s), HQ selects {ic.hq} lag(s)"
)

print(ic.summary().to_markdown())
```

AIC selects 3 lag(s), BIC selects 1 lag(s), HQ selects 1 lag(s) \| lag \| aic \| bic \| hq \| \|——:\|———:\|———:\|———:\| \| 1 \| -13.9833 \| -13.7847 \| -13.9029 \| \| 2 \| -13.9792 \| -13.6304 \| -13.838 \| \| 3 \| -13.9922 \| -13.4922 \| -13.7898 \| \| 4 \| -13.9362 \| -13.2839 \| -13.6721 \| \| 5 \| -13.8713 \| -13.0657 \| -13.5451 \| \| 6 \| -13.7903 \| -12.8301 \| -13.4015 \| \| 7 \| -13.7737 \| -12.658 \| -13.3219 \| \| 8 \| -13.6892 \| -12.4167 \| -13.1738 \|

BIC selects 1 lag – correctly recovering the true DGP order. The criteria table above shows the penalty-adjusted fit at each lag. BIC’s stronger complexity penalty makes it a sensible default when the goal is parsimony.

## Specify the model

`VAR(lags=1, prior="minnesota")` creates a model specification. The Minnesota prior, introduced by Doan, Impulso, and Sims (1984), shrinks each variable’s own lags toward a random walk and cross-variable lags toward zero. This regularisation is especially valuable when the number of parameters grows quadratically with the number of variables and linearly with lags. For more detail, see the [Minnesota prior explanation](../explanation/minnesota-prior.md).

``` python
spec = VAR(lags=1, prior="minnesota")
spec
```

    VAR(lags=1, max_lags=None, prior='minnesota')

## Estimate the model

Calling `.fit()` builds a PyMC model under the hood, places the Minnesota prior on the coefficient matrix, and samples the posterior using NUTS (the No-U-Turn Sampler). We use a fixed `random_seed` for reproducibility. More draws improve the posterior approximation but take longer.

``` python
sampler = NUTSSampler(draws=500, tune=500, chains=2, cores=1, random_seed=42)
fitted = spec.fit(data, sampler=sampler)
fitted
```

<style>
    :root {
        --column-width-1: 40%; /* Progress column width */
        --column-width-2: 15%; /* Chain column width */
        --column-width-3: 15%; /* Divergences column width */
        --column-width-4: 15%; /* Step Size column width */
        --column-width-5: 15%; /* Gradients/Draw column width */
    }
&#10;    .nutpie {
        max-width: 800px;
        margin: 10px auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        //color: #333;
        //background-color: #fff;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-radius: 8px;
        font-size: 14px; /* Smaller font size for a more compact look */
    }
    .nutpie table {
        width: 100%;
        border-collapse: collapse; /* Remove any extra space between borders */
    }
    .nutpie th, .nutpie td {
        padding: 8px 10px; /* Reduce padding to make table more compact */
        text-align: left;
        border-bottom: 1px solid #888;
    }
    .nutpie th {
        //background-color: #f0f0f0;
    }
&#10;    .nutpie th:nth-child(1) { width: var(--column-width-1); }
    .nutpie th:nth-child(2) { width: var(--column-width-2); }
    .nutpie th:nth-child(3) { width: var(--column-width-3); }
    .nutpie th:nth-child(4) { width: var(--column-width-4); }
    .nutpie th:nth-child(5) { width: var(--column-width-5); }
&#10;    .nutpie progress {
        width: 100%;
        height: 15px; /* Smaller progress bars */
        border-radius: 5px;
    }
    progress::-webkit-progress-bar {
        background-color: #eee;
        border-radius: 5px;
    }
    progress::-webkit-progress-value {
        background-color: #5cb85c;
        border-radius: 5px;
    }
    progress::-moz-progress-bar {
        background-color: #5cb85c;
        border-radius: 5px;
    }
    .nutpie .progress-cell {
        width: 100%;
    }
&#10;    .nutpie p strong { font-size: 16px; font-weight: bold; }
&#10;    @media (prefers-color-scheme: dark) {
        .nutpie {
            //color: #ddd;
            //background-color: #1e1e1e;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .nutpie table, .nutpie th, .nutpie td {
            border-color: #555;
            color: #ccc;
        }
        .nutpie th {
            background-color: #2a2a2a;
        }
        .nutpie progress::-webkit-progress-bar {
            background-color: #444;
        }
        .nutpie progress::-webkit-progress-value {
            background-color: #3178c6;
        }
        .nutpie progress::-moz-progress-bar {
            background-color: #3178c6;
        }
    }
</style>

<div class="nutpie">
    <p><strong>Sampler Progress</strong></p>
    <p>Total Chains: <span id="total-chains">2</span></p>
    <p>Active Chains: <span id="active-chains">0</span></p>
    <p>
        Finished Chains:
        <span id="active-chains">2</span>
    </p>
    <p>Sampling for now</p>
    <p>
        Estimated Time to Completion:
        <span id="eta">now</span>
    </p>
&#10;    <progress
        id="total-progress-bar"
        max="2000"
        value="2000">
    </progress>
    <table>
        <thead>
            <tr>
                <th>Progress</th>
                <th>Draws</th>
                <th>Divergences</th>
                <th>Step Size</th>
                <th>Gradients/Draw</th>
            </tr>
        </thead>
        <tbody id="chain-details">
            &#10;                <tr>
                    <td class="progress-cell">
                        <progress
                            max="1000"
                            value="1000">
                        </progress>
                    </td>
                    <td>1000</td>
                    <td>0</td>
                    <td>0.74</td>
                    <td>7</td>
                </tr>
            &#10;                <tr>
                    <td class="progress-cell">
                        <progress
                            max="1000"
                            value="1000">
                        </progress>
                    </td>
                    <td>1000</td>
                    <td>0</td>
                    <td>0.82</td>
                    <td>7</td>
                </tr>
            &#10;            </tr>
        </tbody>
    </table>
</div>

    FittedVAR(n_lags=1, data=VARData(endog_names=['gdp_growth', 'inflation', 'rate'], exog_names=None), var_names=['gdp_growth', 'inflation', 'rate'], has_exog=False)

## Inspect the posterior

The fitted model stores the full posterior in ArviZ `InferenceData` format. `az.summary()` shows posterior means, standard deviations, and highest density intervals for each parameter. We can check whether the posterior recovers the true DGP coefficients.

``` python
az.summary(fitted.idata, var_names=["B", "intercept"])
```

|  | mean | sd | hdi_3% | hdi_97% | mcse_mean | mcse_sd | ess_bulk | ess_tail | r_hat |
|----|----|----|----|----|----|----|----|----|----|
| B\[0, 0\] | 0.645 | 0.056 | 0.543 | 0.748 | 0.001 | 0.002 | 2321.0 | 694.0 | 1.00 |
| B\[0, 1\] | 0.009 | 0.041 | -0.068 | 0.084 | 0.001 | 0.002 | 2237.0 | 668.0 | 1.00 |
| B\[0, 2\] | 0.019 | 0.044 | -0.069 | 0.095 | 0.001 | 0.002 | 1931.0 | 745.0 | 1.00 |
| B\[1, 0\] | 0.062 | 0.037 | -0.003 | 0.135 | 0.001 | 0.001 | 1889.0 | 730.0 | 1.00 |
| B\[1, 1\] | 0.638 | 0.056 | 0.536 | 0.747 | 0.001 | 0.002 | 2235.0 | 626.0 | 1.00 |
| B\[1, 2\] | -0.030 | 0.042 | -0.115 | 0.042 | 0.001 | 0.001 | 2443.0 | 727.0 | 1.01 |
| B\[2, 0\] | -0.041 | 0.040 | -0.120 | 0.029 | 0.001 | 0.001 | 1947.0 | 787.0 | 1.00 |
| B\[2, 1\] | 0.021 | 0.038 | -0.060 | 0.086 | 0.001 | 0.001 | 1651.0 | 793.0 | 1.00 |
| B\[2, 2\] | 0.537 | 0.059 | 0.429 | 0.649 | 0.001 | 0.002 | 2193.0 | 854.0 | 1.00 |
| intercept\[0\] | -0.001 | 0.008 | -0.016 | 0.012 | 0.000 | 0.000 | 1500.0 | 800.0 | 1.00 |
| intercept\[1\] | -0.006 | 0.007 | -0.020 | 0.007 | 0.000 | 0.000 | 2181.0 | 784.0 | 1.00 |
| intercept\[2\] | -0.001 | 0.007 | -0.014 | 0.011 | 0.000 | 0.000 | 2040.0 | 844.0 | 1.01 |

The posterior means for the `B` coefficients should be close to the true values in `A_true`. For example, the GDP-on-GDP-lag coefficient should be near 0.6 and the GDP-on-rate-lag coefficient near -0.1. The 94% HDIs give you a credible range – if they contain the true value, the model is well calibrated. The intercepts should be near zero since the DGP has no intercept.

## What’s next

With a fitted model in hand, you can:

- **Forecast**: produce probabilistic multi-step-ahead forecasts – see the [Forecasting tutorial](forecasting.md)
- **Identify structural shocks**: apply Cholesky or sign-restriction identification to study causal impulse responses – see the [Structural Analysis tutorial](structural-analysis.md)

<section class="consulting-cta">

<p>

We currently have some <strong>availability for consulting</strong> on how Bayesian modelling, vector autoregressions, and impulso can be integrated into your team’s macroeconomic and financial forecasting work. If this sounds relevant, <a href="https://calendly.com/hello-1761-izqw/15-minute-meeting-clone-1">book an introductory call</a>. These calls are for consulting inquiries only. For technical usage questions and free community support, please use GitHub Discussions and the documentation.
</p>

</section>
