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

            if self.dynamics == "random_walk":
                # Non-centered parameterisation to avoid Neal's funnel between
                # sigma_eta and the latent path. h_1 = h0; h_t = h0 + sigma_eta * sum(z_1..z_{t-1}).
                h0 = pm.Normal("h0", mu=prior_params["h0_mu"], sigma=prior_params["h0_sigma"])
                z = pm.Normal("z", mu=0.0, sigma=1.0, shape=T - 1)
                h_path = pt.concatenate([pt.as_tensor_variable([h0]), h0 + sigma_eta * pt.cumsum(z)])
                h = pm.Deterministic("h", h_path)
            else:  # ar1
                phi = pm.Beta("phi", alpha=prior_params["phi_a"], beta=prior_params["phi_b"])
                alpha = pm.Normal(
                    "alpha",
                    mu=prior_params["alpha_mu"],
                    sigma=prior_params["alpha_sigma"],
                )
                # Non-centered: zero-mean unit-sigma AR(1) g_t with autocorrelation phi,
                # then h_t = alpha + sigma_eta * g_t. Decouples sigma_eta from the latent geometry.
                g_init = pm.Normal.dist(mu=0.0, sigma=1.0)
                g = pm.AR(
                    "g",
                    rho=pt.stack([pt.as_tensor_variable(0.0), phi]),
                    sigma=1.0,
                    constant=True,
                    init_dist=g_init,
                    shape=T,
                )
                h = pm.Deterministic("h", alpha + pt.mul(sigma_eta, g))

            pm.Normal("y", mu=mu, sigma=pm.math.exp(pt.mul(0.5, h)), observed=y)

        return model
