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
from impulso.evidence import ModelEvidence, _response_digest
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
            for a homoscedastic conjugate VAR. The break must declare at least
            one hyperparameter to estimate; an adapter with none is rejected
            at construction because the closed-form fast path would silently
            ignore it.
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

    @field_validator("volatility", mode="after")
    @classmethod
    def _require_estimable_hyperparameters(cls, value: ConjugateVolatility | None) -> ConjugateVolatility | None:
        """Reject a volatility break with nothing to estimate (issue #161).

        The conjugate engine has no seam for fixed, known scales: it estimates the
        volatility hyperparameters jointly with the Minnesota tightness by Metropolis,
        and with no free hyperparameter at all `select_and_sample` takes the
        closed-form fast path — which fits homoscedastically and never calls the
        adapter's `log_scales`. Such a break would be silently ignored, so refuse it
        at construction rather than return a fit that quietly disregards it.
        """
        if value is not None and not value.hyperparameter_priors():
            raise ValueError(
                f"{type(value).__name__} declares no volatility hyperparameters: "
                "hyperparameter_priors() returned an empty mapping. The conjugate engine "
                "estimates volatility hyperparameters jointly with the Minnesota tightness "
                "by Metropolis, so a break with none to estimate leaves the fit on the "
                "closed-form fast path, which is homoscedastic and never consults the "
                "adapter's log_scales — the break would be silently ignored. Give the "
                "adapter at least one hyperparameter prior, or apply fixed known scales "
                "by pre-scaling the data before building VARData."
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
            all with a singleton ``chain`` dimension. The posterior's attrs
            carry `in_sample_length` (observations after lag trimming) so
            volatility adapters can anchor forecast paths at the true sample
            end, and — only when at least one hyperparameter was estimated —
            `metropolis_acceptance_rate`, the acceptance rate of the
            random-walk Metropolis sampler over the retained draws. On the
            fixed-prior fast path no Metropolis chain runs (draws come
            straight from the closed-form posterior), so the attr is absent
            rather than stamped with a meaningless 1.0.
            `FittedVAR.evidence` carries the closed-form log marginal
            likelihood at the selected hyperparameters together with the
            metadata `impulso.compare_evidence` needs to form Bayes factors.

        Raises:
            ValueError: If `data` carries exogenous regressors — the
                conjugate engine estimates endogenous dynamics only, and
                silently dropping the exog block would corrupt every
                downstream forecast.
        """
        if data.exog is not None:
            raise ValueError(
                "ConjugateVAR does not support exogenous regressors: the conjugate "
                "engine estimates endogenous dynamics only, and silently ignoring "
                "the exog block would corrupt downstream forecasts. Drop exog from "
                "VARData or use the PyMC/NUTS estimator (impulso.VAR), which "
                "consumes it."
            )
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

        posterior = xr.Dataset(posterior_vars)
        # Volatility adapters anchor forecast scale paths at the true sample
        # end (see ConjugateVolatility.forecast_cholesky_path).
        n_obs = data.endog.shape[0] - self.lags
        posterior.attrs["in_sample_length"] = n_obs
        # Hyperparameter-sampler quality signal for convergence reporting. Only
        # meaningful when the Metropolis chain actually ran: with no free
        # hyperparameters `select_and_sample` takes the closed-form fast path and
        # returns a placeholder rate of 1.0, so leave the attr off entirely there.
        if result["hyperparameters"]:
            posterior.attrs["metropolis_acceptance_rate"] = float(result["acceptance_rate"])
        idata = az.InferenceData(posterior=posterior)
        volatility = self.volatility if self.volatility is not None else Constant()

        evidence = ModelEvidence(
            log_marginal_likelihood=result["log_marginal_likelihood"],
            n_obs=n_obs,
            n_vars=len(data.endog_names),
            var_names=list(data.endog_names),
            n_lags=self.lags,
            volatility=(
                None if self.volatility is None else getattr(self.volatility, "name", type(self.volatility).__name__)
            ),
            hyperparameters=dict(result["mode"]),
            sample_start=data.index[self.lags],
            sample_end=data.index[-1],
            sample_digest=_response_digest(data.endog, data.endog_names, self.lags),
        )

        # `error_dist` is deliberately left at its Gaussian default: the
        # Normal-Inverse-Wishart posterior is conjugate *to a Gaussian
        # likelihood*, so the closed form cannot host a Student-t observation
        # law at all (the t is a scale mixture, which breaks conjugacy — it
        # would need a per-observation latent scale and a Gibbs step). Heavy
        # tails are a PyMC/NUTS-path feature; see ADR-0007.
        return FittedVAR.model_construct(
            idata=idata,
            n_lags=self.lags,
            data=data,
            var_names=data.endog_names,
            volatility=volatility,
            evidence=evidence,
        )
