# Probabilistic Forecasts


Conventional VARs produce point forecasts. A Bayesian VAR produces a full posterior predictive distribution over future paths. This means every forecast comes with calibrated uncertainty — wide bands when the model is unsure, narrow when the data are informative.

``` python
import numpy as np
import pandas as pd

from impulso import VAR, VARData
from impulso.samplers import NUTSSampler
```

## Setup

We repeat the data-generating process from the [quickstart tutorial](quickstart.md). The DGP is a VAR(1) with three macro variables — GDP growth, inflation, and an interest rate. If you’ve already worked through that notebook, the setup code below will be familiar.

``` python
rng = np.random.default_rng(42)
T = 200
n_vars = 3

A_true = np.array([
    [0.6, 0.0, -0.1],
    [0.2, 0.5, 0.0],
    [0.0, 0.15, 0.4],
])

y = np.zeros((T, n_vars))
for t in range(1, T):
    y[t] = A_true @ y[t - 1] + rng.standard_normal(n_vars) * 0.1

index = pd.date_range("2000-01-01", periods=T, freq="QS")
data = VARData(endog=y, endog_names=["gdp_growth", "inflation", "rate"], index=index)

sampler = NUTSSampler(draws=500, tune=500, chains=2, cores=1, random_seed=42)
fitted = VAR(lags=1, prior="minnesota").fit(data, sampler=sampler)
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

## Point forecasts

Call `.forecast(steps=8)` to produce an 8-step-ahead forecast. The result is a `ForecastResult` object that holds the full posterior predictive draws. The `.median()` method extracts the central tendency — the posterior median at each horizon.

``` python
fcast = fitted.forecast(steps=8)
fcast.median()
```

|     | gdp_growth | inflation | rate      |
|-----|------------|-----------|-----------|
| 0   | -0.013904  | 0.072168  | -0.053514 |
| 1   | -0.009715  | 0.041279  | -0.027897 |
| 2   | -0.006648  | 0.020815  | -0.014629 |
| 3   | -0.005040  | 0.008232  | -0.008310 |
| 4   | -0.003237  | -0.000271 | -0.005325 |
| 5   | -0.001996  | -0.005426 | -0.003812 |
| 6   | -0.001585  | -0.008575 | -0.003110 |
| 7   | -0.001419  | -0.010643 | -0.002580 |

Each row is a forecast horizon (1 through 8 quarters ahead). The values converge toward the unconditional mean of the process as the horizon increases — a hallmark of stationary VARs.

## Credible intervals

The `.hdi()` method computes the highest density interval at a given probability level. An 89% HDI means 89% of the posterior forecast mass falls within these bounds. We use 89% rather than 95% following the ArviZ convention — it avoids the false precision of round numbers.

``` python
hdi = fcast.hdi(prob=0.89)

print("Lower bounds:")
print(hdi.lower)
print("\nUpper bounds:")
print(hdi.upper)
```

    Lower bounds:
       gdp_growth  inflation      rate
    0   -0.029950   0.053070 -0.071551
    1   -0.035203   0.015770 -0.049894
    2   -0.034322  -0.007591 -0.037466
    3   -0.034431  -0.022282 -0.030866
    4   -0.037699  -0.032357 -0.027206
    5   -0.037708  -0.038754 -0.028479
    6   -0.037572  -0.038684 -0.025764
    7   -0.037351  -0.041122 -0.024178

    Upper bounds:
       gdp_growth  inflation      rate
    0    0.003275   0.089627 -0.037677
    1    0.013563   0.067530 -0.006803
    2    0.023137   0.051853  0.008579
    3    0.027775   0.041415  0.014705
    4    0.027247   0.033236  0.018755
    5    0.028361   0.028338  0.017704
    6    0.029564   0.028623  0.020326
    7    0.030857   0.026834  0.021439

The intervals widen at longer horizons. This is expected: uncertainty compounds over time because each forecast step propagates parameter uncertainty forward.

## Visualise the forecast

The `.plot()` method produces a fan chart showing the median forecast with shaded credible bands for each variable.

``` python
fig = fcast.plot()
```

![](forecasting_files/figure-commonmark/cell-7-output-1.png)

The fan chart shows the posterior median (line) and 89% HDI (shaded region) for each variable. The bands widen at longer horizons, reflecting compounding uncertainty. GDP growth and the interest rate show the widest bands, consistent with their stronger cross-variable dependencies in the DGP.

## Tidy export

For downstream analysis or dashboarding, `.to_dataframe()` returns the median forecast in a tidy DataFrame format.

``` python
fcast.to_dataframe()
```

|      | gdp_growth | inflation | rate      |
|------|------------|-----------|-----------|
| step |            |           |           |
| 0    | -0.013904  | 0.072168  | -0.053514 |
| 1    | -0.009715  | 0.041279  | -0.027897 |
| 2    | -0.006648  | 0.020815  | -0.014629 |
| 3    | -0.005040  | 0.008232  | -0.008310 |
| 4    | -0.003237  | -0.000271 | -0.005325 |
| 5    | -0.001996  | -0.005426 | -0.003812 |
| 6    | -0.001585  | -0.008575 | -0.003110 |
| 7    | -0.001419  | -0.010643 | -0.002580 |

## Summary

Bayesian VAR forecasts provide more than point predictions. The full posterior predictive distribution lets you quantify and communicate forecast uncertainty honestly. For structural questions — what happens to inflation when the central bank raises rates? — see the [Structural Analysis tutorial](structural-analysis.md).

<section class="consulting-cta">

<p>

We currently have some <strong>availability for consulting</strong> on how Bayesian modelling, vector autoregressions, and impulso can be integrated into your team’s macroeconomic and financial forecasting work. If this sounds relevant, <a href="https://calendly.com/hello-1761-izqw/15-minute-meeting-clone-1">book an introductory call</a>. These calls are for consulting inquiries only. For technical usage questions and free community support, please use GitHub Discussions and the documentation.
</p>

</section>
