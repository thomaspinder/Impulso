"""StochasticVolatility model specification."""

from typing import TYPE_CHECKING, Literal

from impulso._base import ImpulsoBaseModel
from impulso.sv.data import SVData
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
        dynamics: Log-volatility dynamics. "random_walk" (Primiceri 2005)
            or "ar1" (Kim-Shephard-Chib 1998).
        prior: Prior shorthand string or SVPrior instance.
    """

    dynamics: Literal["random_walk", "ar1"] = "random_walk"
    prior: Literal["default"] | SVPrior = "default"

    @property
    def resolved_prior(self) -> SVPrior:
        """Resolve string shorthand to a concrete SVPrior instance."""
        if isinstance(self.prior, str):
            return _SV_PRIOR_REGISTRY[self.prior]()
        return self.prior

    @staticmethod
    def _default_sampler() -> "Sampler":
        """Return the default sampler for SV fits.

        SV posteriors have heavier tails than linear VARs, so we bump
        `target_accept` to 0.9. We also pin `cores=1` because the
        multiprocessing backend segfaults on macOS for PyMC+PyTensor
        (see `CLAUDE.md`); users who want parallel chains should pass
        a custom `NUTSSampler(cores=n, ...)` explicitly.
        """
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

        prior_params = self.resolved_prior.build_priors(data.y)
        model = self._build_pymc_model(data.y, prior_params)
        idata = sampler.sample(model)

        return FittedSV.model_construct(
            idata=idata,
            data=data,
            dynamics=self.dynamics,
        )

    def _build_pymc_model(self, y, prior_params: dict):
        """Build the PyMC model for the chosen dynamics.

        Args:
            y: 1-D observed array.
            prior_params: Dict from SVPrior.build_priors().

        Returns:
            pm.Model with observed likelihood.
        """
        import pymc as pm
        import pytensor.tensor as pt

        T = len(y)
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=prior_params["mu_mu"], sigma=prior_params["mu_sigma"])
            sigma_eta = pm.HalfNormal("sigma_eta", sigma=prior_params["sigma_eta_scale"])

            init_dist = pm.Normal.dist(
                mu=prior_params["h0_mu"],
                sigma=prior_params["h0_sigma"],
            )

            if self.dynamics == "random_walk":
                h = pm.GaussianRandomWalk(
                    "h",
                    mu=0.0,
                    sigma=sigma_eta,
                    init_dist=init_dist,
                    shape=T,
                )
            else:  # ar1
                phi = pm.Beta("phi", alpha=prior_params["phi_a"], beta=prior_params["phi_b"])
                alpha = pm.Normal(
                    "alpha",
                    mu=prior_params["alpha_mu"],
                    sigma=prior_params["alpha_sigma"],
                )
                # Reparameterise as h_t = alpha + phi*(h_{t-1} - alpha) + sigma_eta*eta_t
                # Equivalently: h_t = (1-phi)*alpha + phi*h_{t-1} + sigma_eta*eta_t
                intercept_term = pt.mul(pt.sub(1.0, phi), alpha)
                rho = pt.stack([intercept_term, phi])
                h = pm.AR(
                    "h",
                    rho=rho,
                    sigma=sigma_eta,
                    constant=True,
                    init_dist=init_dist,
                    shape=T,
                )

            pm.Normal("y", mu=mu, sigma=pm.math.exp(pt.mul(0.5, h)), observed=y)

        return model
