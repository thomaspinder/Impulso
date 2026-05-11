"""StochasticVolatility model specification."""

from typing import TYPE_CHECKING, Literal

import numpy as np

from impulso._base import ImpulsoBaseModel
from impulso.sv.data import SVData
from impulso.sv.dynamics import SV_DYNAMICS_REGISTRY, SVDynamics
from impulso.sv.priors import SVDefaultPrior, SVPrior

if TYPE_CHECKING:
    import pytensor.tensor as pt
    import xarray as xr

    from impulso.protocols import Sampler
    from impulso.sv.fitted import FittedSV

_SV_PRIOR_REGISTRY: dict[str, type] = {
    "default": SVDefaultPrior,
}


class StochasticVolatility(ImpulsoBaseModel):
    """Univariate stochastic volatility model.

    Attributes:
        name: Discriminator key for the volatility-process registry
            (always ``"sv"``).
        dynamics: Log-volatility dynamics. String shorthand (``"random_walk"``
            or ``"ar1"``) or an explicit ``SVDynamics`` instance (e.g.
            ``RandomWalk()``, ``AR1()``).
        prior: Prior shorthand string or SVPrior instance.
    """

    name: Literal["sv"] = "sv"
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

    def build_pymc_latent(self, n_vars: int, T: int) -> "pt.TensorVariable":
        """Register the Clark-style multivariate SV latents.

        For each ``i`` in ``0..n_vars-1``: registers a log-volatility path
        ``h_i,t`` following the configured dynamics, plus a per-variable
        ``sigma_eta_i`` (HalfNormal) and ``mu_i`` (Normal). The shared
        mixing factor ``R_chol`` (a unit-diagonal lower-triangular
        ``n_vars x n_vars`` matrix) is registered once via the manual LKJ
        workaround. Note: pinning the diagonal of a Cholesky factor to 1
        does **not** make ``R_chol @ R_chol.T`` a correlation matrix; the
        Gram-matrix diagonal is ``1 + sum_j off[i,j]^2``. The diagonal
        pin is an *identifiability* device: all volatility scaling lives
        in ``h``, so ``R_chol`` is identified only up to its
        off-diagonal mixing entries. See CLAUDE.md for the broader
        LKJCholeskyCov workaround.

        Convention divergence from the standalone univariate fit:

        - ``_build_pymc_model`` (standalone): likelihood is
          ``N(mu, exp(h/2))`` — ``mu`` is the conditional mean of ``y``
          and the latent ``h`` is mean-zero log-vol of innovations.
        - This adapter (multivariate, Clark-style): ``mu_i`` is folded
          into the log-vol *level* (``h_i + mu_i``); the conditional
          mean of ``y`` lives in the VAR's intercept, not here.

        Args:
            n_vars: Number of structural shocks / endogenous variables.
            T: Number of in-sample observations.

        Returns:
            ``L_t`` of shape ``(T, n_vars, n_vars)`` where
            ``L_t[t] = diag(exp(h_t / 2)) @ R_chol``.
        """
        import pymc as pm
        import pytensor.tensor as pt

        dynamics = self.resolved_dynamics
        # The SV prior depends on the observed series only via the data
        # mean / variance used to set mu_mu, h0_mu, etc. The multivariate
        # adapter cannot peek at the structural-shock series (it does not
        # exist at this point in the pipeline), so we use a synthetic
        # zero-array. This produces a near-pinpoint prior on mu_i
        # (mu_sigma == 1e-6 floor) and an h0_mu of log(1e-8) ~= -18.4.
        # These are weakly-influential given the data; a P4 issue tracks
        # tightening this (see plans/2026-05-10-volatility-process-seam-p3.md).
        prior_params = self.resolved_prior.build_priors(np.zeros(T))

        # Per-variable log-vol paths.
        h_paths = []
        for i in range(n_vars):
            prefix = f"v{i}_"
            mu_i = pm.Normal(f"{prefix}mu", mu=prior_params["mu_mu"], sigma=prior_params["mu_sigma"])
            sigma_eta_i = pm.HalfNormal(f"{prefix}sigma_eta", sigma=prior_params["sigma_eta_scale"])
            h_i = dynamics.build_latent_path(prior_params, T, sigma_eta_i, name_prefix=prefix)
            # Shift by mu_i so the log-vol *level* is mu_i; the VAR intercept
            # handles the mean of y, so h here is innovation log-vol.
            h_paths.append(h_i + mu_i)

        # h: (T, n_vars) — stacked per-variable log-vol levels.
        h = pt.stack(h_paths, axis=1)
        pm.Deterministic("h", h)

        # Unit-diagonal lower-triangular mixing factor R_chol (n_vars x n_vars).
        # Manual parameterisation per the LKJ workaround documented in CLAUDE.md.
        # Diagonal pinned to 1 for identification (vol scale lives in h, so the
        # mixing factor is identified only by its off-diagonals); off-diagonals
        # from Normal(0, 0.5). This does NOT make R_chol @ R_chol.T a correlation
        # matrix — its diagonal is 1 + sum_j off[i,j]^2.
        n_tril = n_vars * (n_vars - 1) // 2
        R_chol = pt.eye(n_vars)
        if n_tril > 0:
            offdiag = pm.Normal("R_chol_offdiag", mu=0.0, sigma=0.5, shape=n_tril)
            idx = 0
            for i in range(1, n_vars):
                for j in range(i):
                    R_chol = pt.set_subtensor(R_chol[i, j], offdiag[idx])
                    idx += 1
        pm.Deterministic("R_chol", R_chol)

        # L_t = diag(exp(h_t / 2)) @ R_chol for each t.
        # Broadcasting: L[t, i, j] = sigma_t[t, i] * R_chol[i, j].
        sigma_t = pt.exp(h / 2)  # (T, n_vars)
        L = sigma_t[:, :, None] * R_chol[None, :, :]  # (T, n_vars, n_vars)
        # NOTE: SVDynamics.forecast_log_vol reads bare posterior keys ("h",
        # "sigma_eta", "phi", "alpha"). After build_pymc_latent fits, the
        # posterior has prefixed names (v0_h, v0_sigma_eta, ...). Task 6's
        # forecast_cholesky_path must build a per-variable slice posterior
        # with renamed keys before calling forecast_log_vol.
        return L

    def cholesky_at(self, posterior: "xr.Dataset", t: int | None) -> np.ndarray:
        """Posterior draws of the Cholesky factor at time ``t``.

        Stub; body lands in Task 5. See ADR-0003 and the
        ``VolatilityProcess`` Protocol in ``impulso.protocols``.
        """
        raise NotImplementedError  # Task 5

    def cholesky_path(self, posterior: "xr.Dataset", T: int) -> np.ndarray:
        """Posterior draws of the Cholesky factor path across in-sample ``t``.

        Stub; body lands in Task 5. See ADR-0003 and the
        ``VolatilityProcess`` Protocol in ``impulso.protocols``.
        """
        raise NotImplementedError  # Task 5

    def forecast_cholesky_path(
        self,
        posterior: "xr.Dataset",
        steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Posterior-predictive Cholesky factors for ``steps`` ahead.

        Stub; body lands in Task 6. See ADR-0003 and the
        ``VolatilityProcess`` Protocol in ``impulso.protocols``.
        """
        raise NotImplementedError  # Task 6
