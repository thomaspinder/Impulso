"""ConjugateVAR — sibling estimator using the closed-form Normal-Inverse-Wishart path.

`ConjugateVAR` assembles the pure-NumPy/SciPy conjugate engine (`_conjugate`) and its
empirical-Bayes hyperparameter sampler (`_conjugate_sampler`) into the shared
`FittedVAR` container, so forecasting and structural identification work exactly as
they do for the PyMC/NUTS `VAR` path. It never touches PyMC.

The prior must be a :class:`~impulso.priors.NIWPrior` (the conjugate Minnesota prior);
the optional volatility break must be a :class:`~impulso.conjugate_volatility.ConjugateVolatility`.
Cross-paradigm combinations (an independent-Normal prior, or a PyMC volatility process
such as ``Constant``/``StochasticVolatility``) are rejected with a message pointing to `VAR`.

See docs/adr/0004-conjugate-var-is-a-sibling-estimator.md and the build contract.
"""

from __future__ import annotations

import arviz as az
import xarray as xr
from pydantic import Field, field_validator

from impulso._base import ImpulsoBaseModel
from impulso._conjugate import split_intercept
from impulso._conjugate_sampler import select_and_sample
from impulso.conjugate_volatility import ConjugateVolatility
from impulso.data import VARData
from impulso.fitted import FittedVAR
from impulso.priors import NIWPrior
from impulso.volatility import Constant


class ConjugateVAR(ImpulsoBaseModel):
    """Closed-form conjugate (Normal-Inverse-Wishart) Bayesian VAR estimator.

    Attributes:
        lags: Number of lags ``p`` (>= 1).
        prior: The conjugate Minnesota :class:`~impulso.priors.NIWPrior`.
        volatility: Optional deterministic volatility break
            (:class:`~impulso.conjugate_volatility.ConjugateVolatility`), or ``None``
            for a homoscedastic conjugate VAR.
        draws: Number of retained posterior draws.
        tune: Number of Metropolis warm-up iterations (ignored on the fixed-prior fast path).
        seed: Seed for the single RNG driving selection, sampling and coefficient draws.
    """

    lags: int = Field(ge=1)
    prior: NIWPrior
    volatility: ConjugateVolatility | None = None
    draws: int = Field(1000, ge=1)
    tune: int = Field(1000, ge=0)
    seed: int | None = None

    @field_validator("prior", mode="before")
    @classmethod
    def _require_niw_prior(cls, value: object) -> object:
        """Reject non-conjugate priors, pointing at ``VAR`` for the NUTS path."""
        if not isinstance(value, NIWPrior):
            raise ValueError(  # noqa: TRY004
                f"ConjugateVAR requires a conjugate NIWPrior, got {type(value).__name__}. "
                "Independent-Normal priors (e.g. MinnesotaPrior) belong to the PyMC/NUTS "
                "estimator: use `impulso.VAR(prior=...)` instead."
            )
        return value

    @field_validator("volatility", mode="before")
    @classmethod
    def _require_conjugate_volatility(cls, value: object) -> object:
        """Reject PyMC volatility processes, pointing at ``VAR`` for that path."""
        if value is not None and not isinstance(value, ConjugateVolatility):
            raise ValueError(
                f"ConjugateVAR only accepts a ConjugateVolatility break, got {type(value).__name__}. "
                "PyMC volatility processes (Constant, StochasticVolatility) belong to the "
                "PyMC/NUTS estimator: use `impulso.VAR(volatility=...)` instead."
            )
        return value

    def fit(self, data: VARData) -> FittedVAR:
        """Estimate the conjugate VAR and pack the draws into a :class:`FittedVAR`.

        Args:
            data: Endogenous data to fit.

        Returns:
            A ``FittedVAR`` whose posterior holds ``B`` (lag coefficients only),
            ``intercept``, the base Cholesky factor ``L``, and every estimated
            hyperparameter (e.g. ``lambda_``, ``s_march``, ``s_april``, ``s_may``, ``rho``),
            all with a singleton ``chain`` dimension.
        """
        result = select_and_sample(
            data.endog,
            self.lags,
            self.prior,
            self.volatility,
            draws=self.draws,
            tune=self.tune,
            seed=self.seed,
        )

        intercept, b_lags = split_intercept(result["B_full"])  # (draws, n), (draws, n, n*lags)

        # Add a singleton chain dimension (Metropolis / direct draws = single chain).
        posterior_vars: dict[str, tuple[list[str], object]] = {
            "B": (["chain", "draw", "var", "coeff"], b_lags[None]),
            "intercept": (["chain", "draw", "var"], intercept[None]),
            "L": (["chain", "draw", "var1", "var2"], result["L"][None]),
        }
        for name, arr in result["hyperparameters"].items():
            posterior_vars[name] = (["chain", "draw"], arr[None])

        idata = az.InferenceData(posterior=xr.Dataset(posterior_vars))
        volatility = self.volatility if self.volatility is not None else Constant()

        return FittedVAR.model_construct(
            idata=idata,
            n_lags=self.lags,
            data=data,
            var_names=data.endog_names,
            volatility=volatility,
        )
