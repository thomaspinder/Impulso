"""StochasticVolatility model specification."""

from typing import TYPE_CHECKING, Literal

import numpy as np

from impulso._base import ImpulsoBaseModel
from impulso.sv.data import SVData
from impulso.sv.dynamics import SV_DYNAMICS_REGISTRY, SVDynamics
from impulso.sv.priors import SVDefaultPrior, SVPrior

if TYPE_CHECKING:
    from impulso.protocols import Sampler
    from impulso.sv.fitted import FittedSV

_SV_PRIOR_REGISTRY: dict[str, type] = {
    "default": SVDefaultPrior,
}


class StochasticVolatility(ImpulsoBaseModel):
    """Univariate stochastic volatility model.

    Attributes:
        dynamics: Log-volatility dynamics. String shorthand (``"random_walk"``
            or ``"ar1"``) or an explicit ``SVDynamics`` instance (e.g.
            ``RandomWalk()``, ``AR1()``).
        prior: Prior shorthand string or SVPrior instance.
    """

    dynamics: Literal["random_walk", "ar1"] | SVDynamics = "random_walk"
    prior: Literal["default"] | SVPrior = "default"

    @property
    def resolved_dynamics(self) -> SVDynamics:
        """Resolve string shorthand to a concrete SVDynamics instance."""
        if isinstance(self.dynamics, str):
            return SV_DYNAMICS_REGISTRY[self.dynamics]()
        return self.dynamics

    @property
    def resolved_prior(self) -> SVPrior:
        """Resolve string shorthand to a concrete SVPrior instance."""
        if isinstance(self.prior, str):
            return _SV_PRIOR_REGISTRY[self.prior]()
        return self.prior

    @staticmethod
    def _default_sampler() -> "Sampler":
        """Default sampler for SV: cores=1 (macOS PyMC segfault), target_accept=0.9."""
        from impulso.samplers import NUTSSampler

        return NUTSSampler(cores=1, chains=4, target_accept=0.9)

    def fit(
        self,
        data: SVData,
        sampler: "Sampler | None" = None,
    ) -> "FittedSV":
        """Fit the SV model via NUTS.

        Args:
            data: SVData container.
            sampler: Sampler instance. Defaults to `_default_sampler()`
                (`cores=1`, `chains=4`, `target_accept=0.9`).

        Returns:
            FittedSV with posterior draws.
        """
        from impulso.sv.fitted import FittedSV

        if sampler is None:
            sampler = self._default_sampler()

        dynamics = self.resolved_dynamics
        prior_params = self.resolved_prior.build_priors(data.y)
        model = self._build_pymc_model(data.y, prior_params, dynamics)
        idata = sampler.sample(model)

        return FittedSV.model_construct(
            idata=idata,
            data=data,
            dynamics=dynamics,
        )

    def _build_pymc_model(self, y: np.ndarray, prior_params: dict, dynamics: SVDynamics):
        """Build the PyMC model with the given log-vol dynamics."""
        import pymc as pm
        import pytensor.tensor as pt

        T = len(y)
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=prior_params["mu_mu"], sigma=prior_params["mu_sigma"])
            sigma_eta = pm.HalfNormal("sigma_eta", sigma=prior_params["sigma_eta_scale"])
            h = dynamics.build_latent_path(prior_params, T, sigma_eta)
            pm.Normal("y", mu=mu, sigma=pm.math.exp(pt.mul(0.5, h)), observed=y)

        return model
