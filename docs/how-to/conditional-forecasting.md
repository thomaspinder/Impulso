# Conditional Forecasting

Standard forecasts let the model speak freely — given the historical data, where do the variables go next? But policy analysis often needs to answer a different question: **what happens if we assume a specific path for one variable?**

For example:
- "What happens to GDP and inflation if the central bank raises the policy rate by 25bp at each of the next 4 meetings?"
- "What is the inflation outlook if oil prices stay at \$80/barrel for the next year?"

Conditional forecasting answers these questions. It finds the **smallest set of shocks** consistent with the assumed path, then traces out the implications for all other variables. This implements the algorithm of Waggoner & Zha (1999).

## Defining conditions

A `ForecastCondition` specifies the variable, the periods (0-indexed forecast steps), and the target values:

```python
from impulso import ForecastCondition

# "The policy rate will be 5.25 at steps 0, 1, 2, and 3"
rate_path = ForecastCondition(
    variable="rate",
    periods=[0, 1, 2, 3],
    values=[5.25, 5.50, 5.75, 6.00],
)
```

You can specify multiple conditions on different variables:

```python
# Also condition on oil prices
oil_path = ForecastCondition(
    variable="oil",
    periods=[0, 1, 2, 3],
    values=[80.0, 80.0, 80.0, 80.0],
)
```

!!! note "Periods are 0-indexed"
    Period 0 is the first forecast step (one step ahead of the last observation). Period 3 is four steps ahead.

## Reduced-form conditional forecasts

On a `FittedVAR` (before identification), conditional forecasts use the Cholesky factor of the residual covariance to define the shock space:

```python
from impulso import VAR, VARData, ForecastCondition

data = VARData.from_df(df, endog=["gdp", "inflation", "rate"])
fitted = VAR(lags=4, prior="minnesota").fit(data)

# Condition on a rising rate path
rate_path = ForecastCondition(
    variable="rate",
    periods=[0, 1, 2, 3],
    values=[5.25, 5.50, 5.75, 6.00],
)

result = fitted.conditional_forecast(steps=8, conditions=[rate_path])
```

The result is a `ConditionalForecastResult` — a subclass of `ForecastResult` that also stores the conditions. You can inspect it the same way:

```python
# Posterior median conditional forecast
result.median()

# HDI credible intervals
result.hdi(prob=0.89)

# Plot
result.plot()
```

The constrained variable ("rate") will hit its target values exactly at the specified periods. The unconstrained variables ("gdp", "inflation") show the model's best prediction given those constraints — how the economy would evolve under the assumed rate path.

!!! tip "Interpreting the results"
    The conditional forecast answers: "What is the most likely path for all variables, given that `rate` follows the specified path?" The uncertainty bands for unconstrained variables reflect both parameter uncertainty (different posterior draws give different answers) and the uncertainty about which combination of shocks would produce the assumed path.

## Structural conditional forecasts

If you have an identified model, you can also condition on **structural shock paths**. This is useful for scenario analysis: "What happens if there are no supply shocks for the next 4 quarters?"

```python
from impulso.identification import Cholesky

identified = fitted.set_identification_strategy(
    Cholesky(ordering=["gdp", "inflation", "rate"])
)

# Condition on zero supply shocks
no_supply_shocks = ForecastCondition(
    variable="gdp",  # shock named after the first variable in ordering
    periods=[0, 1, 2, 3],
    values=[0.0, 0.0, 0.0, 0.0],
)

result = identified.conditional_forecast(
    steps=8,
    conditions=[rate_path],
    shock_conditions=[no_supply_shocks],
)
```

Here, `conditions` constrain observable variable paths (as before), and `shock_conditions` constrain structural shock paths. You can use either or both.

!!! warning "Shock naming"
    Shock names correspond to the variable names in the identification scheme's ordering. With `Cholesky(ordering=["gdp", "inflation", "rate"])`, the shocks are named "gdp", "inflation", and "rate".

## Degrees of freedom

Each condition uses up one degree of freedom per constrained period. The total number of constraints cannot exceed `steps * n_variables` (the total number of shock values to be determined). In practice, keeping constraints well below this limit gives more stable results — the system becomes increasingly sensitive as you approach the maximum.

!!! tip "Start simple"
    Begin with conditions on one variable at a few periods. Add more constraints incrementally and check that results remain sensible. Over-constraining can produce large, implausible shocks.
