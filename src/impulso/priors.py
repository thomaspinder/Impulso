"""Prior specifications for VAR models."""

from typing import Literal

import numpy as np
from pydantic import Field

from impulso._base import ImpulsoModel
from impulso._conjugate import ar1_residual_sd, minnesota_dummies


class MinnesotaPrior(ImpulsoModel):
    """Minnesota prior for VAR coefficient shrinkage.

    Attributes:
        tightness: Overall shrinkage toward prior mean. Must be > 0.
        decay: How coefficients shrink on longer lags.
        cross_shrinkage: Shrinkage on other variables' lags vs own. 0 = only own lags, 1 = equal.
    """

    tightness: float = Field(0.1, gt=0)
    decay: Literal["harmonic", "geometric"] = "harmonic"
    cross_shrinkage: float = Field(0.5, ge=0, le=1)

    def build_priors(self, n_vars: int, n_lags: int) -> dict[str, np.ndarray]:
        """Build prior mean and standard deviation arrays for VAR coefficients.

        Args:
            n_vars: Number of endogenous variables.
            n_lags: Number of lags.

        Returns:
            Dictionary with keys 'B_mu' and 'B_sigma' as numpy arrays.
        """
        n_coeffs = n_vars * n_lags

        # B_mu: identity on the first lag block, zero elsewhere
        B_mu = np.zeros((n_vars, n_coeffs))
        B_mu[np.arange(n_vars), np.arange(n_vars)] = 1.0

        # Lag decay per column: each lag's decay repeated n_vars times
        lags = np.arange(1, n_lags + 1)
        lag_decay = 1.0 / lags if self.decay == "harmonic" else 1.0 / lags**2
        decay_per_col = np.repeat(lag_decay, n_vars)  # (n_coeffs,)

        # Own vs cross mask: 1.0 on own-variable columns, cross_shrinkage elsewhere
        col_var = np.arange(n_coeffs) % n_vars
        is_own = col_var[np.newaxis, :] == np.arange(n_vars)[:, np.newaxis]
        cross_mask = np.where(is_own, 1.0, self.cross_shrinkage)

        B_sigma = self.tightness * decay_per_col[np.newaxis, :] * cross_mask

        return {"B_mu": B_mu, "B_sigma": B_sigma}


class NIWPrior(ImpulsoModel):
    """Natural-conjugate Normal-Inverse-Wishart Minnesota prior (Giannone-Lenza-Primiceri, 2015).

    Distinct from :class:`MinnesotaPrior`: that prior uses an independent-Normal
    coefficient prior with a separate covariance and is sampled with MCMC, whereas this
    prior is conjugate, so the posterior and marginal likelihood are closed-form. The
    conjugate (Kronecker) structure is what buys the closed form; its cost is that
    per-equation own/cross shrinkage asymmetry is not identified (use ``MinnesotaPrior``
    for that).

    Attributes:
        tightness: Overall Minnesota shrinkage ``lambda`` (prior standard deviation).
            Must be > 0. When ``select`` is set this is only the starting value; the
            tightness is estimated by marginal likelihood.
        select: Estimate the tightness from the data (empirical / hierarchical Bayes)
            rather than fixing it.
        decay: Lag-decay exponent on the prior *variance* (GLP ``alpha``); ``2`` gives a
            harmonic decay of the prior standard deviation.
        cross_shrinkage: Shared lag-variance scale; ``1.0`` reproduces GLP (2015). In the
            conjugate prior this is not separately identified from ``tightness``.
        sum_of_coefficients: Sum-of-coefficients prior scale, or ``None`` to disable.
        single_unit_root: Single-unit-root (dummy-initial-observation) prior scale, or
            ``None`` to disable.
        lambda_mode: Mode of the Gamma hyperprior on the tightness (used when ``select``).
        lambda_sd: Standard deviation of the Gamma hyperprior on the tightness.
    """

    tightness: float = Field(0.2, gt=0)
    select: bool = False
    decay: float = Field(2.0, ge=0)
    cross_shrinkage: float = Field(1.0, gt=0)
    sum_of_coefficients: float | None = Field(None, gt=0)
    single_unit_root: float | None = Field(None, gt=0)
    lambda_mode: float = Field(0.2, gt=0)
    lambda_sd: float = Field(0.4, gt=0)

    def build_dummies(
        self,
        y: np.ndarray,
        n_lags: int,
        sigma: np.ndarray | None = None,
        *,
        tightness: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the Minnesota dummy observations this prior implies.

        Args:
            y: Raw data of shape ``(T_full, n_vars)``.
            n_lags: Number of lags.
            sigma: Per-variable scale (AR(1) residual sd). Computed from ``y`` via
                :func:`impulso._conjugate.ar1_residual_sd` when ``None``.
            tightness: Override for ``lambda`` (used when sweeping the marginal
                likelihood during selection); defaults to :attr:`tightness`.

        Returns:
            Tuple ``(Yd, Xd)`` as returned by :func:`impulso._conjugate.minnesota_dummies`.
        """
        if sigma is None:
            sigma = ar1_residual_sd(y)
        lam = self.tightness if tightness is None else tightness
        return minnesota_dummies(
            y,
            n_lags,
            lam=lam,
            decay=self.decay,
            cross=self.cross_shrinkage,
            sigma=sigma,
            mu_sur=self.single_unit_root,
            mu_soc=self.sum_of_coefficients,
        )
