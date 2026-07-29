"""Observation error distributions for the VAR pipeline.

Defines concrete adapters of the `ErrorDistribution` Protocol declared in
protocols.py. The seam owns *how the observation error enters the model*:
which likelihood is registered inside PyMC, and how standardised
innovations are drawn on the forecast side. It deliberately does not own
the error's *scale* — that stays with the volatility process, which
supplies the Cholesky factor `L` of the scale matrix Ω = L Lᵀ.

Two adapters ship today:

* `Gaussian` — the default. `y_t ~ N(μ_t, Ω)`, so Ω is both the scale
  matrix and the covariance.
* `StudentT` — heavy-tailed. `y_t ~ MvT_nu(μ_t, Ω)` with Ω the **scale**
  matrix, not the covariance: `Cov[y_t] = nu/(nu-2) · Ω` for nu > 2. See
  docs/adr/0007-student-t-errors-use-the-scale-matrix-convention.md.

The multivariate t is a scale mixture of normals with a single mixing
variable shared across the whole observation vector:

    y_t = μ_t + L ξ_t,    ξ_t = z_t / sqrt(g_t / nu),
    z_t ~ N(0, I_n),      g_t ~ χ²_nu,    g_t independent across t.

`draw_standardised_innovations` reproduces exactly that construction, so
`FittedVAR.forecast` matches the law PyMC's own `MvStudentT` random
variable would sample from.
"""

from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from impulso._base import ImpulsoModel

if TYPE_CHECKING:
    import xarray as xr


def _nu_draws(posterior: "xr.Dataset") -> np.ndarray:
    """Read posterior draws of the degrees of freedom nu.

    nu is registered as a PyMC `Deterministic` under **both** the fixed and
    the inferred parameterisation, so the posterior is the single source of
    truth in either mode and no caller has to branch on how the model was
    specified.

    Args:
        posterior: An xarray Dataset (typically `idata.posterior`).

    Returns:
        Array of shape `(chains, draws)`.

    Raises:
        ValueError: If the posterior carries no `nu` variable.
    """
    if "nu" not in posterior:
        raise ValueError(
            "This posterior carries no 'nu' variable, so the Student-t "
            "degrees of freedom are unknown. A VAR fitted with "
            "error_dist='student_t' registers 'nu' as a deterministic under "
            "both fixed and inferred parameterisations; a hand-built "
            "posterior must supply it as a (chain, draw) variable."
        )
    return np.asarray(posterior["nu"].values, dtype=float)


class Gaussian(ImpulsoModel):
    """Gaussian observation errors — `y_t ~ N(μ_t, Ω)`.

    The default and the historical behaviour of the library. Under this
    adapter the volatility process's Ω = L Lᵀ is simultaneously the scale
    matrix and the innovation covariance, so `FittedVAR.sigma()` and
    `FittedVAR.innovation_covariance()` agree exactly.

    Attributes:
        name: Discriminator key for the registry (always `"gaussian"`).
        is_heavy_tailed: Always `False`.
    """

    name: Literal["gaussian"] = "gaussian"
    is_heavy_tailed: bool = False

    def build_likelihood(
        self,
        name: str,
        mu: Any,
        chol: Any,
        observed: np.ndarray,
        dims: tuple[str, ...] | None = None,
    ) -> Any:
        """Register the multivariate normal likelihood in the active PyMC model.

        PyMC handles batched `chol` natively: a 2-D factor is shared by every
        observation, a 3-D `(T, n, n)` factor gives observation `t` its own.

        Args:
            name: Name for the observed random variable (the pipeline uses
                `"obs"`).
            mu: Conditional mean tensor, shape `(T, n_vars)`.
            chol: Lower-triangular Cholesky factor of Ω, shape
                `(n_vars, n_vars)` or `(T, n_vars, n_vars)`.
            observed: Observed endogenous matrix, shape `(T, n_vars)`.
            dims: PyMC dims for the observed variable.

        Returns:
            The registered PyMC random variable.
        """
        import pymc as pm

        return pm.MvNormal(name, mu=mu, chol=chol, observed=observed, dims=dims)

    def draw_standardised_innovations(
        self,
        shape: tuple[int, ...],
        rng: np.random.Generator,
        posterior: "xr.Dataset",
    ) -> np.ndarray:
        """Draw standardised innovations ξ with `Cov[ξ] = I`.

        A single `rng.standard_normal(shape)` call and nothing else — this
        is the RNG contract that keeps seeded Gaussian forecasts
        bit-identical to every release before the error-distribution seam
        existed.

        Args:
            shape: Draw shape, `(chains, draws, n_vars)`.
            rng: Generator to consume.
            posterior: Unused; accepted for Protocol parity.

        Returns:
            Standard normal draws of shape `shape`.
        """
        return rng.standard_normal(shape)

    def variance_inflation(self, posterior: "xr.Dataset") -> float:
        """Factor converting the scale matrix Ω into the innovation covariance.

        Always `1.0` under Gaussian errors: Ω *is* the covariance.

        Args:
            posterior: Unused; accepted for Protocol parity.

        Returns:
            The scalar `1.0`.
        """
        return 1.0


class StudentT(ImpulsoModel):
    """Student-t observation errors — `y_t ~ MvT_nu(μ_t, Ω)`.

    Heavy-tailed innovations for samples with outliers (crises, pandemic
    quarters, data revisions). Estimation stays fully Bayesian: the t
    likelihood downweights extreme observations automatically rather than
    requiring them to be dummied out by hand, because the t score with
    respect to the location is bounded and redescending —
    `∂ log p/∂μ = w · Ω⁻¹(y - μ)` with `w = (nu + n)/(nu + q)` and `q` the
    squared Mahalanobis distance, so `w → 0` as an observation moves far
    into the tail.

    Ω = L Lᵀ is the **scale** matrix, not the covariance:
    `E[y_t] = μ_t` (nu > 1) and `Cov[y_t] = nu/(nu-2) · Ω` (nu > 2). The
    consequences of that convention — which downstream quantities change
    and which are exactly invariant — are set out in
    docs/adr/0007-student-t-errors-use-the-scale-matrix-convention.md.

    Attributes:
        name: Discriminator key for the registry (always `"student_t"`).
        is_heavy_tailed: Always `True`.
        nu: Degrees of freedom. A float strictly greater than 2 fixes nu;
            the default `"infer"` estimates it from the data. Small values
            mean fat tails (nu ≈ 4 is aggressively robust); large values
            approach the Gaussian.
        prior_alpha: Shape of the Gamma prior on the excess degrees of
            freedom `nu - 2`. Must be strictly greater than 1 so the prior
            density vanishes at the origin — see `PRIOR_ALPHA_LOWER`. Only
            used when `nu="infer"`.
        prior_beta: Rate of that Gamma prior. Only used when `nu="infer"`.
    """

    name: Literal["student_t"] = "student_t"
    is_heavy_tailed: bool = True

    nu: float | Literal["infer"] = "infer"
    prior_alpha: float = 2.0
    prior_beta: float = Field(0.1, gt=0)

    NU_LOWER: ClassVar[float] = 2.0
    """Hard lower bound on nu. Not configurable: below it the t has infinite
    variance, which would make `innovation_covariance`, density-forecast
    bands, and variance decompositions meaningless. Users who genuinely want
    infinite-variance innovations should build the PyMC model directly."""

    PRIOR_ALPHA_LOWER: ClassVar[float] = 1.0
    """Hard lower bound on `prior_alpha`. `Gamma(alpha, ·)` has zero density at
    the origin only for alpha > 1: at alpha = 1 the density there is the rate
    itself and for alpha < 1 it is unbounded. Since the prior on nu is that
    Gamma shifted by `NU_LOWER`, any alpha <= 1 piles mass against the nu = 2
    infinite-variance boundary the shift exists to avoid."""

    @model_validator(mode="after")
    def _validate_nu(self) -> Self:
        if not isinstance(self.nu, str) and self.nu <= self.NU_LOWER:
            raise ValueError(
                f"nu must be > {self.NU_LOWER}, got {self.nu}. At or below "
                f"{self.NU_LOWER} the multivariate t has infinite variance, so "
                "the innovation covariance nu/(nu-2)*Omega diverges and density "
                "forecasts, FEVD shares, and forecast bands stop being defined. "
                "Use a larger nu, or nu='infer' to let the data choose."
            )
        return self

    @model_validator(mode="after")
    def _validate_prior_alpha(self) -> Self:
        if self.prior_alpha <= self.PRIOR_ALPHA_LOWER:
            raise ValueError(
                f"prior_alpha must be > {self.PRIOR_ALPHA_LOWER}, got {self.prior_alpha}. "
                "The prior on nu is a Gamma on the excess degrees of freedom shifted by "
                f"{self.NU_LOWER}, chosen because its density vanishes at the origin — "
                "exactly where nu/(nu-2) diverges. That property needs alpha > "
                f"{self.PRIOR_ALPHA_LOWER}: at alpha = 1 the Gamma density at 0 equals the "
                "rate and below 1 it is unbounded, so the prior would concentrate against "
                f"the nu = {self.NU_LOWER} infinite-variance boundary instead of avoiding "
                "it. Use prior_alpha > 1 (the default 2.0 is the reference choice)."
            )
        return self

    def build_likelihood(
        self,
        name: str,
        mu: Any,
        chol: Any,
        observed: np.ndarray,
        dims: tuple[str, ...] | None = None,
    ) -> Any:
        """Register the multivariate Student-t likelihood in the active PyMC model.

        Registers nu as a `Deterministic` named `"nu"` in both modes, so the
        posterior always carries the degrees of freedom and downstream code
        never has to know whether they were fixed or inferred. When
        `nu="infer"`, the free random variable is the *excess* degrees of
        freedom `nu_excess ~ Gamma(prior_alpha, prior_beta)` and
        `nu = 2 + nu_excess`. The shift (rather than a truncation) is
        deliberate: `Gamma(alpha, ·)` has zero density at the origin for any
        `alpha > 1` — which the validator enforces — so the prior vanishes
        exactly where `nu/(nu-2)` blows up, and the unconstrained NUTS
        transform is the standard positive-support one.

        `chol` is passed straight through, so the manual Cholesky
        parameterisation the volatility seam builds carries over verbatim
        (`pm.MvStudentT` shares `quaddist_matrix` with `pm.MvNormal`).

        Args:
            name: Name for the observed random variable (the pipeline uses
                `"obs"`).
            mu: Conditional mean tensor, shape `(T, n_vars)`.
            chol: Lower-triangular Cholesky factor of the scale matrix Ω.
            observed: Observed endogenous matrix, shape `(T, n_vars)`.
            dims: PyMC dims for the observed variable.

        Returns:
            The registered PyMC random variable.
        """
        import pymc as pm
        import pytensor.tensor as pt

        if isinstance(self.nu, str):
            nu_excess = pm.Gamma("nu_excess", alpha=self.prior_alpha, beta=self.prior_beta)
            nu = pm.Deterministic("nu", pt.add(nu_excess, self.NU_LOWER))
        else:
            nu = pm.Deterministic("nu", pt.as_tensor(float(self.nu)))

        return pm.MvStudentT(name, nu=nu, mu=mu, chol=chol, observed=observed, dims=dims)

    def draw_standardised_innovations(
        self,
        shape: tuple[int, ...],
        rng: np.random.Generator,
        posterior: "xr.Dataset",
    ) -> np.ndarray:
        """Draw standardised multivariate-t innovations ξ.

        Reproduces PyMC's own `MvStudentT` construction:
        `ξ = z / sqrt(g / nu)` with `z ~ N(0, I)` and `g ~ χ²_nu` drawn **once
        per observation vector**, not once per component. The shared mixing
        variable is what makes the components dependent (though uncorrelated)
        and the joint law genuinely multivariate-t; drawing one χ² per
        component would give a product of independent t marginals instead.

        `rng.standard_normal` is consumed first, before the χ² draw, so the
        Gaussian adapter's stream stays a strict prefix of this one and
        seeded Gaussian forecasts are unaffected by the seam.

        nu is read per posterior draw, so a model with inferred degrees of
        freedom yields innovations whose tail weight varies across draws.

        Args:
            shape: Draw shape, `(chains, draws, n_vars)`.
            rng: Generator to consume.
            posterior: Posterior Dataset carrying `nu` with shape
                `(chains, draws)`.

        Returns:
            Standardised t innovations of shape `shape`, with `E[ξ] = 0`
            and `Cov[ξ] = nu/(nu-2) · I`. Each marginal is exactly a standard
            `t_nu`.

        Raises:
            ValueError: If the posterior carries no `nu`, or if its shape
                does not match the leading `(chains, draws)` of `shape`.
        """
        z = rng.standard_normal(shape)
        nu = _nu_draws(posterior)
        if nu.shape != tuple(shape[:-1]):
            raise ValueError(
                f"posterior['nu'] has shape {nu.shape} but innovations of shape "
                f"{tuple(shape)} need nu of shape {tuple(shape[:-1])} (chains, draws)."
            )
        g = rng.chisquare(nu)
        return z / np.sqrt(g / nu)[..., np.newaxis]

    def variance_inflation(self, posterior: "xr.Dataset") -> np.ndarray:
        """Factor converting the scale matrix Ω into the innovation covariance.

        `nu/(nu-2)`, evaluated per posterior draw. Finite by construction: nu is
        bounded below by 2 both by the validator (fixed mode) and by the
        shifted Gamma prior (inferred mode).

        Args:
            posterior: Posterior Dataset carrying `nu` with shape
                `(chains, draws)`.

        Returns:
            Array of shape `(chains, draws)`.

        Raises:
            ValueError: If the posterior carries no `nu`.
        """
        nu = _nu_draws(posterior)
        return nu / (nu - self.NU_LOWER)
