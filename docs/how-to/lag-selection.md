# Choosing Lag Order

Litterman provides two ways to set the lag order for your VAR model.

## Fixed lag order

If you know the lag order you want, pass an integer:

```python
from impulso import VAR

spec = VAR(lags=4, prior="minnesota")
```

## Automatic selection via information criteria

Pass a criterion name (`"aic"`, `"bic"`, or `"hq"`) and Litterman selects the optimal lag via OLS:

```python
spec = VAR(lags="bic", prior="minnesota")
```

You can cap the search range:

```python
spec = VAR(lags="aic", max_lags=12, prior="minnesota")
```

## Inspecting the criteria table

Use `select_lag_order` directly to see all criteria values:

```python
from impulso import select_lag_order

ic = select_lag_order(data, max_lags=8)
print(f"AIC selects {ic.aic} lags, BIC selects {ic.bic} lags")
ic.summary()  # Returns a DataFrame with all criteria by lag
```
