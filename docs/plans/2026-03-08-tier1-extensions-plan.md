# Tier 1 Extensions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the 5 foundation extensions that make Impulso competitive with the BVAR R package and the ECB's BEAR toolbox.

**Architecture:** Five layers built bottom-up: (1) dummy observation priors on VARData, (2) ConjugateVAR model class with direct NIW sampling, (3) GLP hierarchical prior selection on ConjugateVAR, (4) long-run Blanchard-Quah identification, (5) conditional forecasting on FittedVAR and IdentifiedVAR.

**Tech Stack:** NumPy, SciPy (invwishart, optimize), ArviZ, xarray, Pydantic v2, pytest

**Design doc:** `docs/plans/2026-03-08-tier1-extensions-design.md`

---

## Task 1: Dummy Observation Priors — Tests

**Files:**
- Create: `tests/test_dummy_observations.py`

**Step 1: Write tests for `with_dummy_observations()`**

```python
"""Tests for dummy observation priors on VARData."""

import numpy as np
import pandas as pd
import pytest

from impulso.data import VARData


@pytest.fixture
def var_data():
    rng = np.random.default_rng(42)
    T, n = 100, 3
    endog = rng.standard_normal((T, n))
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=endog, endog_names=["gdp", "inflation", "rate"], index=index)


class TestDummyObservationPriors:
    def test_sum_of_coefficients_appends_n_vars_rows(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, mu=5.0)
        assert augmented.endog.shape[0] == var_data.endog.shape[0] + 3

    def test_single_unit_root_appends_one_row(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, delta=1.0)
        assert augmented.endog.shape[0] == var_data.endog.shape[0] + 1

    def test_both_dummies_append_correct_rows(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, mu=5.0, delta=1.0)
        assert augmented.endog.shape[0] == var_data.endog.shape[0] + 4

    def test_sum_of_coefficients_values(self, var_data):
        mu = 5.0
        augmented = var_data.with_dummy_observations(n_lags=4, mu=mu)
        y_bar = var_data.endog.mean(axis=0)
        dummy_rows = augmented.endog[var_data.endog.shape[0] :]
        # Each row i should have y_bar[i] / mu in position i, zeros elsewhere
        for i in range(3):
            expected = np.zeros(3)
            expected[i] = y_bar[i] / mu
            np.testing.assert_allclose(dummy_rows[i], expected)

    def test_single_unit_root_values(self, var_data):
        delta = 1.0
        augmented = var_data.with_dummy_observations(n_lags=4, delta=delta)
        y_bar = var_data.endog.mean(axis=0)
        dummy_row = augmented.endog[-1]
        np.testing.assert_allclose(dummy_row, y_bar / delta)

    def test_preserves_original_data(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, mu=5.0)
        np.testing.assert_array_equal(augmented.endog[: var_data.endog.shape[0]], var_data.endog)

    def test_returns_new_vardata(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, mu=5.0)
        assert augmented is not var_data
        assert isinstance(augmented, VARData)

    def test_index_extended(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, mu=5.0)
        assert len(augmented.index) == augmented.endog.shape[0]

    def test_endog_names_preserved(self, var_data):
        augmented = var_data.with_dummy_observations(n_lags=4, mu=5.0)
        assert augmented.endog_names == var_data.endog_names

    def test_raises_if_neither_mu_nor_delta(self, var_data):
        with pytest.raises(ValueError, match="At least one"):
            var_data.with_dummy_observations(n_lags=4)

    def test_raises_if_mu_not_positive(self, var_data):
        with pytest.raises(ValueError, match="mu must be"):
            var_data.with_dummy_observations(n_lags=4, mu=-1.0)

    def test_raises_if_delta_not_positive(self, var_data):
        with pytest.raises(ValueError, match="delta must be"):
            var_data.with_dummy_observations(n_lags=4, delta=0.0)

    def test_raises_if_n_lags_not_positive(self, var_data):
        with pytest.raises(ValueError, match="n_lags must be"):
            var_data.with_dummy_observations(n_lags=0, mu=5.0)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_dummy_observations.py -v`
Expected: FAIL — `VARData has no attribute 'with_dummy_observations'`

**Step 3: Commit test file**

```bash
git add tests/test_dummy_observations.py
git commit -m "test: add tests for dummy observation priors"
```

---

## Task 2: Dummy Observation Priors — Implementation

**Files:**
- Modify: `src/impulso/data.py:12-102` (add method to VARData)

**Step 1: Implement `with_dummy_observations()`**

Add this method to the `VARData` class in `src/impulso/data.py`, after the `from_df` classmethod (after line 101):

```python
def with_dummy_observations(
    self,
    n_lags: int,
    mu: float | None = None,
    delta: float | None = None,
) -> "VARData":
    """Return new VARData with dummy observations appended.

    Dummy observations encode beliefs about unit roots and persistence,
    following Doan, Litterman & Sims (1984) and Sims (1993).

    Args:
        n_lags: Number of VAR lags (needed to construct dummy rows).
        mu: Sum-of-coefficients hyperparameter. Larger = weaker prior.
            Encodes belief that sum of own-lag coefficients is close to 1.
        delta: Single-unit-root hyperparameter. Larger = weaker prior.
            Encodes belief that variables persist at initial levels.

    Returns:
        New VARData with dummy observations appended to endog.
    """
    if mu is None and delta is None:
        raise ValueError("At least one of mu or delta must be provided")
    if mu is not None and mu <= 0:
        raise ValueError(f"mu must be strictly positive, got {mu}")
    if delta is not None and delta <= 0:
        raise ValueError(f"delta must be strictly positive, got {delta}")
    if n_lags < 1:
        raise ValueError(f"n_lags must be >= 1, got {n_lags}")

    n_vars = self.endog.shape[1]
    y_bar = self.endog.mean(axis=0)
    dummy_rows = []

    # Sum-of-coefficients dummies: n_vars rows
    if mu is not None:
        soc = np.zeros((n_vars, n_vars))
        np.fill_diagonal(soc, y_bar / mu)
        dummy_rows.append(soc)

    # Single-unit-root dummy: 1 row
    if delta is not None:
        sur = (y_bar / delta).reshape(1, n_vars)
        dummy_rows.append(sur)

    dummies = np.vstack(dummy_rows)
    new_endog = np.vstack([self.endog, dummies])

    # Extend index with synthetic dates
    freq = self.index.freq or pd.tseries.frequencies.to_offset(pd.infer_freq(self.index))
    n_dummy = dummies.shape[0]
    extra_index = pd.date_range(
        start=self.index[-1] + freq, periods=n_dummy, freq=freq
    )
    new_index = self.index.append(extra_index)

    # Handle exog: pad with zeros for dummy rows
    new_exog = None
    if self.exog is not None:
        exog_padding = np.zeros((n_dummy, self.exog.shape[1]))
        new_exog = np.vstack([self.exog, exog_padding])

    return VARData(
        endog=new_endog,
        endog_names=self.endog_names,
        exog=new_exog,
        exog_names=self.exog_names,
        index=new_index,
    )
```

**Step 2: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_dummy_observations.py -v`
Expected: All PASS

**Step 3: Run full test suite**

Run: `uv run python -m pytest -m "not slow" -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/impulso/data.py
git commit -m "feat: add dummy observation priors to VARData"
```

---

## Task 3: ConjugateVAR — Tests

**Files:**
- Create: `tests/test_conjugate.py`

**Step 1: Write tests for ConjugateVAR**

```python
"""Tests for ConjugateVAR (direct NIW posterior sampling)."""

import arviz as az
import numpy as np
import pandas as pd
import pytest

from impulso.data import VARData
from impulso.priors import MinnesotaPrior


@pytest.fixture
def stable_var_data():
    """VAR(1) DGP with known stable coefficients."""
    rng = np.random.default_rng(42)
    T, n = 200, 2
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2"], index=index)


class TestConjugateVARConstruction:
    def test_basic_construction(self):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=2)
        assert cvar.lags == 2
        assert cvar.draws == 2000

    def test_custom_prior(self):
        from impulso.conjugate import ConjugateVAR

        prior = MinnesotaPrior(tightness=0.2, cross_shrinkage=0.3)
        cvar = ConjugateVAR(lags=2, prior=prior)
        assert cvar.prior == prior

    def test_frozen(self):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=2)
        with pytest.raises(Exception):
            cvar.lags = 4

    def test_rejects_negative_draws(self):
        from impulso.conjugate import ConjugateVAR

        with pytest.raises(Exception):
            ConjugateVAR(lags=2, draws=0)


class TestConjugateVARFit:
    def test_fit_returns_fitted_var(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR
        from impulso.fitted import FittedVAR

        cvar = ConjugateVAR(lags=1, draws=100, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        assert isinstance(fitted, FittedVAR)

    def test_idata_has_required_variables(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=100, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        assert "B" in fitted.idata.posterior
        assert "intercept" in fitted.idata.posterior
        assert "Sigma" in fitted.idata.posterior

    def test_posterior_shapes(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        n_draws = 100
        cvar = ConjugateVAR(lags=2, draws=n_draws, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        B = fitted.idata.posterior["B"].values
        assert B.shape == (1, n_draws, 2, 4)  # (chains=1, draws, n_vars, n_vars*n_lags)
        intercept = fitted.idata.posterior["intercept"].values
        assert intercept.shape == (1, n_draws, 2)
        sigma = fitted.idata.posterior["Sigma"].values
        assert sigma.shape == (1, n_draws, 2, 2)

    def test_sigma_positive_definite(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=100, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        sigma = fitted.idata.posterior["Sigma"].values
        for d in range(sigma.shape[1]):
            eigvals = np.linalg.eigvalsh(sigma[0, d])
            assert np.all(eigvals > 0), f"Draw {d} has non-positive eigenvalue"

    def test_sigma_symmetric(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=100, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        sigma = fitted.idata.posterior["Sigma"].values
        np.testing.assert_allclose(sigma, np.swapaxes(sigma, -2, -1), atol=1e-10)

    def test_reproducible_with_seed(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar1 = ConjugateVAR(lags=1, draws=50, random_seed=123)
        cvar2 = ConjugateVAR(lags=1, draws=50, random_seed=123)
        fitted1 = cvar1.fit(stable_var_data)
        fitted2 = cvar2.fit(stable_var_data)
        np.testing.assert_array_equal(
            fitted1.idata.posterior["B"].values,
            fitted2.idata.posterior["B"].values,
        )

    def test_var_names_correct(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        assert fitted.var_names == ["y1", "y2"]

    def test_n_lags_stored(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=3, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        assert fitted.n_lags == 3

    def test_downstream_forecast_works(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        result = fitted.forecast(steps=4)
        assert result.median().shape == (4, 2)

    def test_downstream_identification_works(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR
        from impulso.identification import Cholesky

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        irfs = identified.impulse_response(horizon=10)
        assert irfs.median().shape[0] == 11

    def test_lag_selection_string(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags="bic", draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        assert fitted.n_lags >= 1

    def test_works_with_dummy_observations(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        augmented = stable_var_data.with_dummy_observations(n_lags=2, mu=5.0, delta=1.0)
        cvar = ConjugateVAR(lags=2, draws=50, random_seed=42)
        fitted = cvar.fit(augmented)
        assert fitted.idata.posterior["B"].values.shape == (1, 50, 2, 4)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_conjugate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'impulso.conjugate'`

**Step 3: Commit test file**

```bash
git add tests/test_conjugate.py
git commit -m "test: add tests for ConjugateVAR"
```

---

## Task 4: ConjugateVAR — Implementation

**Files:**
- Create: `src/impulso/conjugate.py`
- Modify: `src/impulso/__init__.py:1-73` (add exports)

**Step 1: Implement ConjugateVAR**

Create `src/impulso/conjugate.py`:

```python
"""ConjugateVAR — direct Normal-Inverse-Wishart posterior sampling."""

from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from impulso._base import ImpulsoBaseModel
from impulso.data import VARData
from impulso.priors import MinnesotaPrior

if TYPE_CHECKING:
    from impulso.fitted import FittedVAR


class ConjugateVAR(ImpulsoBaseModel):
    """Bayesian VAR with conjugate Normal-Inverse-Wishart estimation.

    Produces iid posterior draws via direct sampling — no MCMC iteration,
    no burn-in, no autocorrelation. Orders of magnitude faster than NUTS
    for models with Minnesota-type priors.

    Attributes:
        lags: Fixed lag order or selection criterion.
        max_lags: Upper bound for automatic lag selection.
        prior: Minnesota prior instance or string shorthand.
        draws: Number of posterior draws.
        random_seed: Seed for reproducibility.
    """

    lags: int | Literal["aic", "bic", "hq"] = Field(...)
    max_lags: int | None = None
    prior: Literal["minnesota", "minnesota_optimized"] | MinnesotaPrior = "minnesota"
    draws: int = Field(2000, ge=1)
    random_seed: int | None = None

    @model_validator(mode="after")
    def _validate_spec(self) -> Self:
        if self.max_lags is not None and isinstance(self.lags, int):
            raise ValueError("max_lags is only valid when lags is a selection criterion")
        if isinstance(self.lags, int) and self.lags < 1:
            raise ValueError(f"lags must be >= 1, got {self.lags}")
        return self

    @property
    def resolved_prior(self) -> MinnesotaPrior:
        """Resolve string shorthand to a MinnesotaPrior instance."""
        if isinstance(self.prior, str):
            return MinnesotaPrior()
        return self.prior

    def fit(self, data: VARData) -> "FittedVAR":
        """Estimate the Bayesian VAR via conjugate NIW posterior sampling.

        Args:
            data: VARData instance.

        Returns:
            FittedVAR with iid posterior draws.
        """
        import arviz as az
        import xarray as xr
        from scipy.stats import invwishart

        from impulso._lag_selection import select_lag_order
        from impulso.fitted import FittedVAR

        # Resolve lags
        if isinstance(self.lags, str):
            max_lags = self.max_lags or 12
            ic = select_lag_order(data, max_lags=max_lags)
            n_lags = getattr(ic, self.lags)
        else:
            n_lags = self.lags

        n_vars = data.endog.shape[1]

        # Resolve prior (optimize if requested)
        if isinstance(self.prior, str) and self.prior == "minnesota_optimized":
            prior = self._optimize_prior_internal(data, n_lags)
        else:
            prior = self.resolved_prior

        prior_params = prior.build_priors(n_vars=n_vars, n_lags=n_lags)

        # Build data matrices: Y = (T-p, n), X = (T-p, n*p + 1) with intercept
        y = data.endog
        Y = y[n_lags:]  # (T-p, n)
        X_parts = [np.ones((Y.shape[0], 1))]  # intercept column
        for lag in range(1, n_lags + 1):
            X_parts.append(y[n_lags - lag : -lag])
        X = np.hstack(X_parts)  # (T-p, 1 + n*p)

        T_eff = Y.shape[0]
        n_coeffs = X.shape[1]  # 1 + n*p

        # Convert Minnesota prior to NIW parameters
        # Prior mean: [intercept_prior | B_mu]
        B_prior = np.zeros((n_coeffs, n_vars))
        B_prior[1:, :] = prior_params["B_mu"].T  # B_mu is (n, n*p), transpose to (n*p, n)

        # Prior precision: diagonal from B_sigma
        # Intercept gets a wide prior (sigma=1 as in PyMC path)
        prior_precision_diag = np.ones(n_coeffs)
        B_sigma_flat = prior_params["B_sigma"].T.ravel()  # (n*p,) per variable -> (n*p,)
        # Use the first variable's sigma as representative for the diagonal
        # Actually: V_prior is (n_coeffs, n_coeffs) diagonal
        intercept_var = 1.0**2
        lag_var = np.mean(prior_params["B_sigma"] ** 2, axis=0)  # average across equations
        prior_var_diag = np.concatenate([[intercept_var], lag_var])
        V_prior = np.diag(prior_var_diag)
        V_prior_inv = np.diag(1.0 / prior_var_diag)

        # OLS estimates for scale matrix initialisation
        B_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid_ols = Y - X @ B_ols
        sigma_ols = (resid_ols.T @ resid_ols) / T_eff

        # NIW prior hyperparameters
        nu_prior = n_vars + 2  # minimally informative
        S_prior = sigma_ols * (nu_prior - n_vars - 1)  # centres IW mode at sigma_ols

        # Posterior parameters
        V_posterior = np.linalg.inv(V_prior_inv + X.T @ X)
        B_posterior = V_posterior @ (V_prior_inv @ B_prior + X.T @ Y)
        nu_posterior = nu_prior + T_eff
        S_posterior = (
            S_prior
            + Y.T @ Y
            + B_prior.T @ V_prior_inv @ B_prior
            - B_posterior.T @ np.linalg.inv(V_posterior) @ B_posterior
        )
        # Symmetrise to avoid numerical issues
        S_posterior = (S_posterior + S_posterior.T) / 2

        # Direct sampling
        rng = np.random.default_rng(self.random_seed)

        B_draws = np.zeros((self.draws, n_coeffs, n_vars))
        Sigma_draws = np.zeros((self.draws, n_vars, n_vars))

        chol_V_posterior = np.linalg.cholesky(V_posterior)

        for i in range(self.draws):
            # Draw Sigma ~ IW(S_posterior, nu_posterior)
            Sigma_draw = invwishart.rvs(df=nu_posterior, scale=S_posterior, random_state=rng)
            Sigma_draws[i] = Sigma_draw

            # Draw B | Sigma ~ MN(B_posterior, Sigma, V_posterior)
            # vec(B) ~ N(vec(B_posterior), Sigma kron V_posterior)
            chol_Sigma = np.linalg.cholesky(Sigma_draw)
            Z = rng.standard_normal((n_coeffs, n_vars))
            B_draw = B_posterior + chol_V_posterior @ Z @ chol_Sigma.T
            B_draws[i] = B_draw

        # Separate intercept and lag coefficients
        intercept_arr = B_draws[:, 0, :]  # (draws, n_vars)
        B_lag_arr = B_draws[:, 1:, :]  # (draws, n*p, n_vars)
        # Transpose to match PyMC convention: B is (n_vars, n_vars*n_lags)
        B_lag_arr = np.swapaxes(B_lag_arr, -2, -1)  # (draws, n_vars, n*p)

        # Add chain dimension (chains=1 for conjugate)
        intercept_arr = intercept_arr[np.newaxis, :]  # (1, draws, n_vars)
        B_lag_arr = B_lag_arr[np.newaxis, :]  # (1, draws, n_vars, n*p)
        Sigma_draws = Sigma_draws[np.newaxis, :]  # (1, draws, n_vars, n_vars)

        # Package as InferenceData
        posterior = xr.Dataset({
            "B": xr.DataArray(B_lag_arr, dims=["chain", "draw", "equations", "coefficients"]),
            "intercept": xr.DataArray(intercept_arr, dims=["chain", "draw", "equations"]),
            "Sigma": xr.DataArray(Sigma_draws, dims=["chain", "draw", "var1", "var2"]),
        })
        idata = az.InferenceData(posterior=posterior)

        return FittedVAR.model_construct(
            idata=idata,
            n_lags=n_lags,
            data=data,
            var_names=data.endog_names,
        )

    def optimize_prior(
        self,
        data: VARData,
        optimize_dummy: bool = False,
    ) -> MinnesotaPrior:
        """Find Minnesota hyperparameters maximising the marginal likelihood.

        Implements Giannone, Lenza & Primiceri (2015) data-driven prior
        selection via closed-form marginal likelihood optimisation.

        Args:
            data: VARData instance (may include dummy observations).
            optimize_dummy: If True, also optimise dummy hyperparameters.

        Returns:
            MinnesotaPrior with optimal tightness and cross_shrinkage.
        """
        from impulso._lag_selection import select_lag_order

        # Resolve lags
        if isinstance(self.lags, str):
            max_lags = self.max_lags or 12
            ic = select_lag_order(data, max_lags=max_lags)
            n_lags = getattr(ic, self.lags)
        else:
            n_lags = self.lags

        return self._optimize_prior_internal(data, n_lags, optimize_dummy)

    def _optimize_prior_internal(
        self,
        data: VARData,
        n_lags: int,
        optimize_dummy: bool = False,
    ) -> MinnesotaPrior:
        """Internal implementation of prior optimisation."""
        from scipy.optimize import minimize

        current_prior = self.resolved_prior

        def neg_log_marginal_likelihood(params: np.ndarray) -> float:
            tightness = params[0]
            cross_shrinkage = params[1]
            prior = MinnesotaPrior(
                tightness=tightness,
                cross_shrinkage=cross_shrinkage,
                decay=current_prior.decay,
            )
            return -self._log_marginal_likelihood(data, n_lags, prior)

        x0 = np.array([current_prior.tightness, current_prior.cross_shrinkage])
        bounds = [(0.001, 10.0), (0.01, 1.0)]

        result = minimize(
            neg_log_marginal_likelihood,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
        )

        return MinnesotaPrior(
            tightness=float(result.x[0]),
            cross_shrinkage=float(result.x[1]),
            decay=current_prior.decay,
        )

    def _log_marginal_likelihood(
        self,
        data: VARData,
        n_lags: int,
        prior: MinnesotaPrior,
    ) -> float:
        """Compute log marginal likelihood p(Y|lambda) for NIW conjugate model.

        Args:
            data: VARData instance.
            n_lags: Number of lags.
            prior: MinnesotaPrior with specific hyperparameters.

        Returns:
            Log marginal likelihood (scalar).
        """
        from scipy.special import gammaln

        n_vars = data.endog.shape[1]
        prior_params = prior.build_priors(n_vars=n_vars, n_lags=n_lags)

        # Build data matrices
        y = data.endog
        Y = y[n_lags:]
        X_parts = [np.ones((Y.shape[0], 1))]
        for lag in range(1, n_lags + 1):
            X_parts.append(y[n_lags - lag : -lag])
        X = np.hstack(X_parts)

        T_eff = Y.shape[0]
        n_coeffs = X.shape[1]

        # Prior parameters (same logic as fit)
        B_prior = np.zeros((n_coeffs, n_vars))
        B_prior[1:, :] = prior_params["B_mu"].T

        intercept_var = 1.0
        lag_var = np.mean(prior_params["B_sigma"] ** 2, axis=0)
        prior_var_diag = np.concatenate([[intercept_var], lag_var])
        V_prior = np.diag(prior_var_diag)
        V_prior_inv = np.diag(1.0 / prior_var_diag)

        B_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid_ols = Y - X @ B_ols
        sigma_ols = (resid_ols.T @ resid_ols) / T_eff

        nu_prior = n_vars + 2
        S_prior = sigma_ols * (nu_prior - n_vars - 1)

        # Posterior parameters
        V_posterior = np.linalg.inv(V_prior_inv + X.T @ X)
        B_posterior = V_posterior @ (V_prior_inv @ B_prior + X.T @ Y)
        nu_posterior = nu_prior + T_eff
        S_posterior = (
            S_prior
            + Y.T @ Y
            + B_prior.T @ V_prior_inv @ B_prior
            - B_posterior.T @ np.linalg.inv(V_posterior) @ B_posterior
        )
        S_posterior = (S_posterior + S_posterior.T) / 2

        # Log marginal likelihood formula
        log_ml = 0.0
        log_ml -= (T_eff * n_vars / 2) * np.log(np.pi)

        # Log-determinant terms
        _, logdet_V_prior = np.linalg.slogdet(V_prior)
        _, logdet_V_posterior = np.linalg.slogdet(V_posterior)
        log_ml += 0.5 * (logdet_V_posterior - logdet_V_prior) * n_vars

        _, logdet_S_prior = np.linalg.slogdet(S_prior)
        _, logdet_S_posterior = np.linalg.slogdet(S_posterior)
        log_ml += (nu_prior / 2) * logdet_S_prior
        log_ml -= (nu_posterior / 2) * logdet_S_posterior

        # Multivariate gamma function terms
        for j in range(n_vars):
            log_ml += gammaln((nu_posterior - j) / 2) - gammaln((nu_prior - j) / 2)

        return log_ml

    def marginal_likelihood(self, data: VARData) -> float:
        """Compute log marginal likelihood for the current prior.

        Args:
            data: VARData instance.

        Returns:
            Log marginal likelihood (scalar).
        """
        from impulso._lag_selection import select_lag_order

        if isinstance(self.lags, str):
            max_lags = self.max_lags or 12
            ic = select_lag_order(data, max_lags=max_lags)
            n_lags = getattr(ic, self.lags)
        else:
            n_lags = self.lags

        return self._log_marginal_likelihood(data, n_lags, self.resolved_prior)
```

**Step 2: Add exports to `__init__.py`**

In `src/impulso/__init__.py`, add `"ConjugateVAR"` to `__all__` and to `_lazy_imports`:

- Add `"ConjugateVAR"` to the `__all__` list
- Add `"ConjugateVAR": "impulso.conjugate"` to `_lazy_imports`

**Step 3: Run tests**

Run: `uv run python -m pytest tests/test_conjugate.py -v`
Expected: All PASS

**Step 4: Run full suite**

Run: `uv run python -m pytest -m "not slow" -v`
Expected: All PASS

**Step 5: Run type checker and linter**

Run: `uv run ruff check src/impulso/conjugate.py && uv run ruff format src/impulso/conjugate.py`
Expected: Clean

**Step 6: Commit**

```bash
git add src/impulso/conjugate.py src/impulso/__init__.py
git commit -m "feat: add ConjugateVAR with direct NIW posterior sampling"
```

---

## Task 5: GLP Hierarchical Prior Selection — Tests

**Files:**
- Create: `tests/test_glp.py`

**Step 1: Write tests for optimize_prior and marginal_likelihood**

```python
"""Tests for GLP hierarchical prior selection on ConjugateVAR."""

import numpy as np
import pandas as pd
import pytest

from impulso.data import VARData
from impulso.priors import MinnesotaPrior


@pytest.fixture
def stable_var_data():
    rng = np.random.default_rng(42)
    T, n = 200, 2
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2"], index=index)


class TestMarginalLikelihood:
    def test_returns_finite_scalar(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1)
        ml = cvar.marginal_likelihood(stable_var_data)
        assert np.isfinite(ml)

    def test_varies_with_prior(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar_tight = ConjugateVAR(lags=1, prior=MinnesotaPrior(tightness=0.01))
        cvar_loose = ConjugateVAR(lags=1, prior=MinnesotaPrior(tightness=1.0))
        ml_tight = cvar_tight.marginal_likelihood(stable_var_data)
        ml_loose = cvar_loose.marginal_likelihood(stable_var_data)
        assert ml_tight != ml_loose

    def test_higher_for_true_lag_order(self, stable_var_data):
        """Marginal likelihood should favour the true DGP lag order (1)."""
        from impulso.conjugate import ConjugateVAR

        ml_1 = ConjugateVAR(lags=1).marginal_likelihood(stable_var_data)
        ml_8 = ConjugateVAR(lags=8).marginal_likelihood(stable_var_data)
        assert ml_1 > ml_8


class TestOptimizePrior:
    def test_returns_minnesota_prior(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1)
        optimal = cvar.optimize_prior(stable_var_data)
        assert isinstance(optimal, MinnesotaPrior)

    def test_optimal_tightness_positive(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1)
        optimal = cvar.optimize_prior(stable_var_data)
        assert optimal.tightness > 0

    def test_optimal_cross_shrinkage_in_bounds(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1)
        optimal = cvar.optimize_prior(stable_var_data)
        assert 0.01 <= optimal.cross_shrinkage <= 1.0

    def test_optimal_has_higher_marginal_likelihood(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar_default = ConjugateVAR(lags=1)
        ml_default = cvar_default.marginal_likelihood(stable_var_data)

        optimal_prior = cvar_default.optimize_prior(stable_var_data)
        cvar_optimal = ConjugateVAR(lags=1, prior=optimal_prior)
        ml_optimal = cvar_optimal.marginal_likelihood(stable_var_data)

        assert ml_optimal >= ml_default - 1e-6  # allow tiny numerical tolerance

    def test_preserves_decay_setting(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, prior=MinnesotaPrior(decay="geometric"))
        optimal = cvar.optimize_prior(stable_var_data)
        assert optimal.decay == "geometric"

    def test_minnesota_optimized_shorthand(self, stable_var_data):
        """prior='minnesota_optimized' should trigger automatic optimisation."""
        from impulso.conjugate import ConjugateVAR
        from impulso.fitted import FittedVAR

        cvar = ConjugateVAR(lags=1, prior="minnesota_optimized", draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        assert isinstance(fitted, FittedVAR)
```

**Step 2: Run tests**

Run: `uv run python -m pytest tests/test_glp.py -v`
Expected: All PASS (GLP is already implemented in ConjugateVAR from Task 4)

**Step 3: Commit**

```bash
git add tests/test_glp.py
git commit -m "test: add tests for GLP hierarchical prior selection"
```

---

## Task 6: Long-Run Restrictions — Tests

**Files:**
- Create: `tests/test_long_run_restriction.py`

**Step 1: Write tests for LongRunRestriction**

```python
"""Tests for Blanchard-Quah long-run identification."""

import arviz as az
import numpy as np
import pytest
import xarray as xr

from impulso.protocols import IdentificationScheme


@pytest.fixture
def stationary_idata_2v():
    """Synthetic InferenceData with stationary VAR(1) coefficients."""
    rng = np.random.default_rng(42)
    n_chains, n_draws, n_vars = 2, 50, 2

    # Stationary coefficients: eigenvalues inside unit circle
    B = np.zeros((n_chains, n_draws, n_vars, n_vars))
    for c in range(n_chains):
        for d in range(n_draws):
            # Diagonal with small values ensures stationarity
            B[c, d] = np.diag(rng.uniform(0.1, 0.4, n_vars))

    intercept = rng.standard_normal((n_chains, n_draws, n_vars)) * 0.01
    sigma = np.zeros((n_chains, n_draws, n_vars, n_vars))
    for c in range(n_chains):
        for d in range(n_draws):
            A = rng.standard_normal((n_vars, n_vars)) * 0.5
            sigma[c, d] = A @ A.T + np.eye(n_vars)

    posterior = xr.Dataset({
        "B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"]),
        "intercept": xr.DataArray(intercept, dims=["chain", "draw", "var"]),
        "Sigma": xr.DataArray(sigma, dims=["chain", "draw", "var1", "var2"]),
    })
    return az.InferenceData(posterior=posterior)


class TestLongRunRestrictionConstruction:
    def test_basic_construction(self):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["output", "prices"])
        assert lr.ordering == ["output", "prices"]

    def test_frozen(self):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["a", "b"])
        with pytest.raises(Exception):
            lr.ordering = ["b", "a"]

    def test_satisfies_protocol(self):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["a", "b"])
        assert isinstance(lr, IdentificationScheme)


class TestLongRunRestrictionIdentify:
    def test_produces_structural_shock_matrix(self, stationary_idata_2v):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y1", "y2"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        assert "structural_shock_matrix" in result.posterior

    def test_output_shape(self, stationary_idata_2v):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y1", "y2"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        P = result.posterior["structural_shock_matrix"].values
        assert P.shape == (2, 50, 2, 2)

    def test_no_nan_values(self, stationary_idata_2v):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y1", "y2"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        P = result.posterior["structural_shock_matrix"].values
        assert not np.any(np.isnan(P))

    def test_long_run_impact_is_lower_triangular(self, stationary_idata_2v):
        """The long-run cumulative impact C(1) @ P should be lower triangular."""
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y1", "y2"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        P = result.posterior["structural_shock_matrix"].values
        B = stationary_idata_2v.posterior["B"].values
        n_vars = 2

        for c in range(2):
            for d in range(50):
                lag_coefficient_sum = B[c, d, :, :n_vars]
                long_run_multiplier = np.linalg.inv(np.eye(n_vars) - lag_coefficient_sum)
                long_run_impact = long_run_multiplier @ P[c, d]
                # Upper triangle (excluding diagonal) should be ~zero
                np.testing.assert_allclose(
                    np.triu(long_run_impact, k=1),
                    0.0,
                    atol=1e-10,
                )

    def test_reordering_works(self, stationary_idata_2v):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y2", "y1"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        assert result.posterior["structural_shock_matrix"].coords["shock"].values.tolist() == ["y2", "y1"]

    def test_coordinates_match_ordering(self, stationary_idata_2v):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y1", "y2"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        assert result.posterior["structural_shock_matrix"].coords["shock"].values.tolist() == ["y1", "y2"]
        assert result.posterior["structural_shock_matrix"].coords["response"].values.tolist() == ["y1", "y2"]

    def test_preserves_other_posterior_variables(self, stationary_idata_2v):
        from impulso.identification import LongRunRestriction

        lr = LongRunRestriction(ordering=["y1", "y2"])
        result = lr.identify(stationary_idata_2v, var_names=["y1", "y2"])
        assert "B" in result.posterior
        assert "Sigma" in result.posterior
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_long_run_restriction.py -v`
Expected: FAIL — `ImportError: cannot import name 'LongRunRestriction'`

**Step 3: Commit test file**

```bash
git add tests/test_long_run_restriction.py
git commit -m "test: add tests for long-run Blanchard-Quah identification"
```

---

## Task 7: Long-Run Restrictions — Implementation

**Files:**
- Modify: `src/impulso/identification.py:1-195` (add LongRunRestriction class)
- Modify: `src/impulso/__init__.py` (add export)

**Step 1: Implement LongRunRestriction**

Add this class to `src/impulso/identification.py`, after the `SignRestriction` class (after line 195):

```python
class LongRunRestriction(ImpulsoModel):
    """Blanchard-Quah long-run identification scheme.

    Identifies structural shocks by their long-run cumulative effects.
    The long-run impact matrix is forced to be lower triangular via
    Cholesky decomposition, so the first shock has no permanent effect
    on the second variable, etc.

    Attributes:
        ordering: Variable ordering (determines which shocks have
            permanent effects on which variables).
    """

    ordering: list[str]

    def identify(self, idata: az.InferenceData, var_names: list[str]) -> az.InferenceData:
        """Apply Blanchard-Quah long-run identification.

        Args:
            idata: InferenceData with 'B' and 'Sigma' in posterior.
            var_names: Variable names from the VAR model.

        Returns:
            InferenceData with 'structural_shock_matrix' added to posterior.
        """
        B_draws = idata.posterior["B"].values  # (C, D, n, n*p)
        sigma_draws = idata.posterior["Sigma"].values  # (C, D, n, n)
        n_chains, n_draws, n_vars, n_total_coeffs = B_draws.shape
        n_lags = n_total_coeffs // n_vars

        # Compute permutation for reordering
        perm = [var_names.index(v) for v in self.ordering]

        P = np.zeros((n_chains, n_draws, n_vars, n_vars))

        for c in range(n_chains):
            for d in range(n_draws):
                B = B_draws[c, d]  # (n, n*p)
                Sigma = sigma_draws[c, d]  # (n, n)

                # Sum of lag coefficient matrices: A_1 + A_2 + ... + A_p
                lag_coefficient_sum = np.zeros((n_vars, n_vars))
                for j in range(n_lags):
                    lag_coefficient_sum += B[:, j * n_vars : (j + 1) * n_vars]

                # Long-run multiplier: (I - A_1 - ... - A_p)^{-1}
                long_run_multiplier = np.linalg.inv(np.eye(n_vars) - lag_coefficient_sum)

                # Reorder for requested ordering
                long_run_multiplier_ordered = long_run_multiplier[np.ix_(perm, perm)]
                Sigma_ordered = Sigma[np.ix_(perm, perm)]

                # Long-run covariance
                long_run_covariance = (
                    long_run_multiplier_ordered @ Sigma_ordered @ long_run_multiplier_ordered.T
                )

                # Cholesky of long-run covariance
                long_run_cholesky = np.linalg.cholesky(long_run_covariance)

                # Structural impact matrix
                structural_impact_matrix = (
                    np.linalg.inv(long_run_multiplier_ordered) @ long_run_cholesky
                )

                P[c, d] = structural_impact_matrix

        P_da = xr.DataArray(
            P,
            dims=["chain", "draw", "shock", "response"],
            coords={"shock": self.ordering, "response": self.ordering},
        )

        new_posterior = idata.posterior.assign(structural_shock_matrix=P_da)
        return az.InferenceData(posterior=new_posterior)
```

**Step 2: Add export to `__init__.py`**

- Add `"LongRunRestriction"` to `__all__`
- Add `"LongRunRestriction": "impulso.identification"` to `_lazy_imports`

**Step 3: Run tests**

Run: `uv run python -m pytest tests/test_long_run_restriction.py -v`
Expected: All PASS

**Step 4: Run full suite**

Run: `uv run python -m pytest -m "not slow" -v`
Expected: All PASS

**Step 5: Lint**

Run: `uv run ruff check src/impulso/identification.py && uv run ruff format src/impulso/identification.py`

**Step 6: Commit**

```bash
git add src/impulso/identification.py src/impulso/__init__.py
git commit -m "feat: add Blanchard-Quah long-run identification"
```

---

## Task 8: Conditional Forecasting — ForecastCondition and Tests

**Files:**
- Create: `src/impulso/conditions.py`
- Create: `tests/test_conditional_forecast.py`

**Step 1: Implement ForecastCondition**

Create `src/impulso/conditions.py`:

```python
"""Forecast condition definitions for conditional forecasting."""

from typing import Literal, Self

from pydantic import Field, model_validator

from impulso._base import ImpulsoModel


class ForecastCondition(ImpulsoModel):
    """A constraint on a variable's future path for conditional forecasting.

    Attributes:
        variable: Name of the variable to constrain.
        periods: Forecast steps to constrain (0-indexed).
        values: Target values at those periods.
        constraint_type: Type of constraint. Only 'hard' is currently supported.
    """

    variable: str
    periods: list[int]
    values: list[float]
    constraint_type: Literal["hard"] = "hard"

    @model_validator(mode="after")
    def _validate_periods_values_match(self) -> Self:
        if len(self.periods) != len(self.values):
            raise ValueError(
                f"periods length ({len(self.periods)}) must equal "
                f"values length ({len(self.values)})"
            )
        if len(self.periods) == 0:
            raise ValueError("periods must be non-empty")
        if any(p < 0 for p in self.periods):
            raise ValueError("All periods must be non-negative")
        return self
```

**Step 2: Write tests for conditional forecasting**

Create `tests/test_conditional_forecast.py`:

```python
"""Tests for conditional forecasting."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from impulso.conditions import ForecastCondition
from impulso.data import VARData


@pytest.fixture
def stable_var_data():
    rng = np.random.default_rng(42)
    T, n = 200, 2
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2"], index=index)


class TestForecastCondition:
    def test_basic_construction(self):
        fc = ForecastCondition(variable="y1", periods=[0, 1, 2], values=[1.0, 1.0, 1.0])
        assert fc.variable == "y1"
        assert fc.periods == [0, 1, 2]
        assert fc.constraint_type == "hard"

    def test_frozen(self):
        fc = ForecastCondition(variable="y1", periods=[0], values=[1.0])
        with pytest.raises(ValidationError):
            fc.variable = "y2"

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValidationError, match="periods length"):
            ForecastCondition(variable="y1", periods=[0, 1], values=[1.0])

    def test_rejects_empty_periods(self):
        with pytest.raises(ValidationError, match="non-empty"):
            ForecastCondition(variable="y1", periods=[], values=[])

    def test_rejects_negative_periods(self):
        with pytest.raises(ValidationError, match="non-negative"):
            ForecastCondition(variable="y1", periods=[-1], values=[1.0])


class TestConditionalForecastOnFittedVAR:
    def test_returns_forecast_result(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        conditions = [
            ForecastCondition(variable="y1", periods=[0, 1, 2, 3], values=[0.5, 0.5, 0.5, 0.5]),
        ]
        result = fitted.conditional_forecast(steps=8, conditions=conditions)
        assert result.median().shape == (8, 2)

    def test_constrained_periods_match_target(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        target = 0.5
        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        conditions = [
            ForecastCondition(variable="y1", periods=[0, 1], values=[target, target]),
        ]
        result = fitted.conditional_forecast(steps=4, conditions=conditions)
        median = result.median()
        # Constrained periods should be close to target (exact for hard constraints)
        np.testing.assert_allclose(median.iloc[0]["y1"], target, atol=1e-6)
        np.testing.assert_allclose(median.iloc[1]["y1"], target, atol=1e-6)

    def test_unconstrained_variable_differs_from_unconditional(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        conditions = [
            ForecastCondition(variable="y1", periods=[0, 1, 2, 3], values=[5.0, 5.0, 5.0, 5.0]),
        ]
        unconditional = fitted.forecast(steps=4).median()
        conditional = fitted.conditional_forecast(steps=4, conditions=conditions).median()
        # y2 should differ because y1 is forced far from its unconditional path
        assert not np.allclose(unconditional["y2"].values, conditional["y2"].values)

    def test_rejects_unknown_variable(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        conditions = [
            ForecastCondition(variable="unknown", periods=[0], values=[1.0]),
        ]
        with pytest.raises(ValueError, match="unknown"):
            fitted.conditional_forecast(steps=4, conditions=conditions)

    def test_rejects_period_out_of_range(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        conditions = [
            ForecastCondition(variable="y1", periods=[10], values=[1.0]),
        ]
        with pytest.raises(ValueError, match="out of range"):
            fitted.conditional_forecast(steps=4, conditions=conditions)


class TestConditionalForecastOnIdentifiedVAR:
    def test_returns_forecast_result(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR
        from impulso.identification import Cholesky

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        conditions = [
            ForecastCondition(variable="y1", periods=[0, 1], values=[0.5, 0.5]),
        ]
        result = identified.conditional_forecast(steps=4, conditions=conditions)
        assert result.median().shape == (4, 2)

    def test_with_shock_conditions(self, stable_var_data):
        from impulso.conjugate import ConjugateVAR
        from impulso.identification import Cholesky

        cvar = ConjugateVAR(lags=1, draws=50, random_seed=42)
        fitted = cvar.fit(stable_var_data)
        identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
        conditions = [
            ForecastCondition(variable="y1", periods=[0, 1], values=[0.5, 0.5]),
        ]
        shock_conditions = [
            ForecastCondition(variable="y1", periods=[0, 1], values=[0.0, 0.0]),
        ]
        result = identified.conditional_forecast(
            steps=4, conditions=conditions, shock_conditions=shock_conditions
        )
        assert result.median().shape == (4, 2)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_conditional_forecast.py -v`
Expected: FAIL — condition tests pass, but `FittedVAR.conditional_forecast` not found

**Step 3: Commit test files**

```bash
git add src/impulso/conditions.py tests/test_conditional_forecast.py
git commit -m "test: add ForecastCondition and conditional forecast tests"
```

---

## Task 9: Conditional Forecasting — Implementation on FittedVAR

**Files:**
- Modify: `src/impulso/fitted.py:1-132` (add conditional_forecast method)
- Modify: `src/impulso/results.py:64-101` (add ConditionalForecastResult)
- Modify: `src/impulso/__init__.py` (add exports)

**Step 1: Add ConditionalForecastResult to results.py**

Add after the `ForecastResult` class (after line 101 in `results.py`):

```python
class ConditionalForecastResult(ForecastResult):
    """Result from conditional VAR forecasting.

    Attributes:
        conditions: List of ForecastConditions applied.
    """

    conditions: list  # list[ForecastCondition], but avoid import for lazy loading
```

**Step 2: Add conditional_forecast to FittedVAR**

Add this method to `FittedVAR` in `fitted.py`, after the `forecast` method (after line 112):

```python
def conditional_forecast(
    self,
    steps: int,
    conditions: list,
    exog_future: np.ndarray | None = None,
) -> "ConditionalForecastResult":
    """Produce conditional forecasts subject to constraints on future paths.

    Implements the Waggoner & Zha (1999) algorithm for hard constraints.
    Computes unconditional forecasts, then solves for the shock paths
    that satisfy the constraints.

    Args:
        steps: Number of forecast steps.
        conditions: List of ForecastCondition instances specifying constraints.
        exog_future: Future exogenous values if model has exog.

    Returns:
        ConditionalForecastResult with constrained posterior forecast draws.
    """
    import xarray as xr

    from impulso.results import ConditionalForecastResult

    # Validate conditions
    for cond in conditions:
        if cond.variable not in self.var_names:
            raise ValueError(
                f"Condition variable '{cond.variable}' not in var_names {self.var_names}"
            )
        for p in cond.periods:
            if p < 0 or p >= steps:
                raise ValueError(
                    f"Condition period {p} out of range for {steps} forecast steps"
                )

    B_draws = self.coefficients  # (C, D, n, n*p)
    intercept_draws = self.intercepts  # (C, D, n)
    sigma_draws = self.sigma  # (C, D, n, n)
    n_chains, n_draws, n_vars, _ = B_draws.shape

    # Compute unconditional forecasts
    y_hist = self.data.endog[-self.n_lags :]
    forecasts = np.zeros((n_chains, n_draws, steps, n_vars))

    for c in range(n_chains):
        for d in range(n_draws):
            B = B_draws[c, d]
            intercept = intercept_draws[c, d]
            Sigma = sigma_draws[c, d]

            # Compute MA coefficients for this draw
            n_lags = self.n_lags
            A_matrices = [B[:, j * n_vars : (j + 1) * n_vars] for j in range(n_lags)]
            ma_coefficients = [np.eye(n_vars)]
            for h in range(1, steps):
                phi_h = np.zeros((n_vars, n_vars))
                for j in range(min(h, n_lags)):
                    phi_h += A_matrices[j] @ ma_coefficients[h - j - 1]
                ma_coefficients.append(phi_h)

            # Unconditional forecast
            y_buffer = y_hist.copy()
            unconditional = np.zeros((steps, n_vars))
            for h in range(steps):
                x_lag = np.concatenate([y_buffer[-(lag + 1)] for lag in range(n_lags)])
                y_new = intercept + B @ x_lag
                if self.has_exog and exog_future is not None:
                    B_exog = self.idata.posterior["B_exog"].values[c, d]
                    y_new = y_new + B_exog @ exog_future[h]
                unconditional[h] = y_new
                y_buffer = np.vstack([y_buffer[1:], y_new.reshape(1, -1)])

            # Build constraint system: R @ shocks = target - unconditional
            constraint_rows = []
            constraint_targets = []
            for cond in conditions:
                var_idx = self.var_names.index(cond.variable)
                for period, value in zip(cond.periods, cond.values):
                    # Row of R: sum of MA coefficients mapping shocks to this variable at this period
                    row = np.zeros(steps * n_vars)
                    for s in range(period + 1):
                        ma = ma_coefficients[period - s]
                        chol_sigma = np.linalg.cholesky(Sigma)
                        response = ma @ chol_sigma
                        row[s * n_vars : (s + 1) * n_vars] = response[var_idx, :]
                    constraint_rows.append(row)
                    constraint_targets.append(value - unconditional[period, var_idx])

            R = np.array(constraint_rows)
            target = np.array(constraint_targets)

            # Solve for constrained shocks (least-squares)
            shocks, _, _, _ = np.linalg.lstsq(R, target, rcond=None)
            shocks = shocks.reshape(steps, n_vars)

            # Compute conditional forecast by adding shock contributions
            chol_sigma = np.linalg.cholesky(Sigma)
            conditional = unconditional.copy()
            for h in range(steps):
                for s in range(h + 1):
                    conditional[h] += ma_coefficients[h - s] @ chol_sigma @ shocks[s]

            forecasts[c, d] = conditional

    forecast_da = xr.DataArray(
        forecasts,
        dims=["chain", "draw", "step", "variable"],
        coords={"variable": self.var_names},
        name="forecast",
    )
    idata = az.InferenceData(posterior_predictive=xr.Dataset({"forecast": forecast_da}))
    return ConditionalForecastResult(
        idata=idata, steps=steps, var_names=self.var_names, conditions=conditions
    )
```

**Step 3: Add exports to `__init__.py`**

- Add `"ForecastCondition"`, `"ConditionalForecastResult"`, to `__all__`
- Add `"ForecastCondition": "impulso.conditions"` and `"ConditionalForecastResult": "impulso.results"` to `_lazy_imports`

**Step 4: Run tests**

Run: `uv run python -m pytest tests/test_conditional_forecast.py::TestForecastCondition tests/test_conditional_forecast.py::TestConditionalForecastOnFittedVAR -v`
Expected: All PASS

**Step 5: Lint**

Run: `uv run ruff check src/impulso/fitted.py src/impulso/conditions.py src/impulso/results.py && uv run ruff format src/impulso/fitted.py src/impulso/conditions.py src/impulso/results.py`

**Step 6: Commit**

```bash
git add src/impulso/fitted.py src/impulso/results.py src/impulso/conditions.py src/impulso/__init__.py
git commit -m "feat: add conditional forecasting on FittedVAR"
```

---

## Task 10: Conditional Forecasting — Implementation on IdentifiedVAR

**Files:**
- Modify: `src/impulso/identified.py:1-182` (add conditional_forecast method)

**Step 1: Add conditional_forecast to IdentifiedVAR**

Add this method to `IdentifiedVAR` in `identified.py`, after the `historical_decomposition` method (after line 181):

```python
def conditional_forecast(
    self,
    steps: int,
    conditions: list,
    shock_conditions: list | None = None,
    exog_future: np.ndarray | None = None,
) -> "ConditionalForecastResult":
    """Produce structural conditional forecasts.

    Extends reduced-form conditional forecasting by allowing constraints
    on structural shock paths in addition to observable variable paths.

    Args:
        steps: Number of forecast steps.
        conditions: List of ForecastCondition instances for observables.
        shock_conditions: Optional list of ForecastCondition instances for
            structural shocks.
        exog_future: Future exogenous values if model has exog.

    Returns:
        ConditionalForecastResult with constrained forecast draws.
    """
    from impulso.fitted import FittedVAR

    # If no shock conditions, delegate to the reduced-form method
    if shock_conditions is None:
        fitted = FittedVAR.model_construct(
            idata=self.idata,
            n_lags=self.n_lags,
            data=self.data,
            var_names=self.var_names,
        )
        return fitted.conditional_forecast(
            steps=steps, conditions=conditions, exog_future=exog_future
        )

    # Structural conditional forecast with shock constraints
    import xarray as xr

    from impulso.results import ConditionalForecastResult

    # Validate conditions
    for cond in conditions:
        if cond.variable not in self.var_names:
            raise ValueError(f"Condition variable '{cond.variable}' not in var_names")
        for p in cond.periods:
            if p < 0 or p >= steps:
                raise ValueError(f"Condition period {p} out of range for {steps} steps")

    shock_names = self.idata.posterior["structural_shock_matrix"].coords["shock"].values.tolist()
    for cond in shock_conditions:
        if cond.variable not in shock_names:
            raise ValueError(f"Shock condition variable '{cond.variable}' not in shock_names {shock_names}")

    B_draws = self.idata.posterior["B"].values
    intercept_draws = self.idata.posterior["intercept"].values
    P_draws = self.idata.posterior["structural_shock_matrix"].values
    n_chains, n_draws, n_vars, _ = B_draws.shape

    y_hist = self.data.endog[-self.n_lags :]
    forecasts = np.zeros((n_chains, n_draws, steps, n_vars))

    for c in range(n_chains):
        for d in range(n_draws):
            B = B_draws[c, d]
            intercept = intercept_draws[c, d]
            P = P_draws[c, d]

            # MA coefficients
            n_lags = self.n_lags
            A_matrices = [B[:, j * n_vars : (j + 1) * n_vars] for j in range(n_lags)]
            ma_coefficients = [np.eye(n_vars)]
            for h in range(1, steps):
                phi_h = np.zeros((n_vars, n_vars))
                for j in range(min(h, n_lags)):
                    phi_h += A_matrices[j] @ ma_coefficients[h - j - 1]
                ma_coefficients.append(phi_h)

            # Unconditional forecast
            y_buffer = y_hist.copy()
            unconditional = np.zeros((steps, n_vars))
            for h in range(steps):
                x_lag = np.concatenate([y_buffer[-(lag + 1)] for lag in range(n_lags)])
                unconditional[h] = intercept + B @ x_lag
                y_buffer = np.vstack([y_buffer[1:], unconditional[h].reshape(1, -1)])

            # Build combined constraint system using structural impact matrix P
            constraint_rows = []
            constraint_targets = []

            # Observable constraints
            for cond in conditions:
                var_idx = self.var_names.index(cond.variable)
                for period, value in zip(cond.periods, cond.values):
                    row = np.zeros(steps * n_vars)
                    for s in range(period + 1):
                        structural_response = ma_coefficients[period - s] @ P
                        row[s * n_vars : (s + 1) * n_vars] = structural_response[var_idx, :]
                    constraint_rows.append(row)
                    constraint_targets.append(value - unconditional[period, var_idx])

            # Shock constraints
            for cond in shock_conditions:
                shock_idx = shock_names.index(cond.variable)
                for period, value in zip(cond.periods, cond.values):
                    row = np.zeros(steps * n_vars)
                    row[period * n_vars + shock_idx] = 1.0
                    constraint_rows.append(row)
                    constraint_targets.append(value)

            R = np.array(constraint_rows)
            target = np.array(constraint_targets)

            structural_shocks, _, _, _ = np.linalg.lstsq(R, target, rcond=None)
            structural_shocks = structural_shocks.reshape(steps, n_vars)

            conditional = unconditional.copy()
            for h in range(steps):
                for s in range(h + 1):
                    conditional[h] += ma_coefficients[h - s] @ P @ structural_shocks[s]

            forecasts[c, d] = conditional

    forecast_da = xr.DataArray(
        forecasts,
        dims=["chain", "draw", "step", "variable"],
        coords={"variable": self.var_names},
        name="forecast",
    )
    idata = az.InferenceData(posterior_predictive=xr.Dataset({"forecast": forecast_da}))
    all_conditions = conditions + (shock_conditions or [])
    return ConditionalForecastResult(
        idata=idata, steps=steps, var_names=self.var_names, conditions=all_conditions
    )
```

**Step 2: Add required import at top of identified.py**

Add `from impulso.results import ..., ConditionalForecastResult` (or use lazy import inside method as shown above).

**Step 3: Run tests**

Run: `uv run python -m pytest tests/test_conditional_forecast.py -v`
Expected: All PASS

**Step 4: Run full test suite**

Run: `uv run python -m pytest -m "not slow" -v`
Expected: All PASS

**Step 5: Lint and type check**

Run: `uv run ruff check . && uv run ruff format .`

**Step 6: Commit**

```bash
git add src/impulso/identified.py
git commit -m "feat: add conditional forecasting on IdentifiedVAR"
```

---

## Task 11: Final Integration — Public API and Full Test Suite

**Files:**
- Modify: `src/impulso/__init__.py` (verify all exports)
- Run: full test suite, type checker, linter

**Step 1: Verify `__init__.py` exports are complete**

Ensure these are all in `__all__` and `_lazy_imports`:
- `ConjugateVAR` -> `impulso.conjugate`
- `LongRunRestriction` -> `impulso.identification`
- `ForecastCondition` -> `impulso.conditions`
- `ConditionalForecastResult` -> `impulso.results`

**Step 2: Run full test suite**

Run: `uv run python -m pytest -m "not slow" -v`
Expected: All PASS

**Step 3: Run linter and type checker**

Run: `make check`
Expected: Clean

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete Tier 1 extensions (conjugate sampler, dummy priors, GLP, long-run ID, conditional forecast)"
```
