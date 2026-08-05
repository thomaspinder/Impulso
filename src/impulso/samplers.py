"""Sampler specifications for posterior inference."""

import os
from typing import Literal

import numpy as np
import pymc as pm
from pydantic import Field

from impulso._arviz_compat import InferenceDataLike, get_group_dataset
from impulso._base import ImpulsoModel


def _raise_on_incomplete_draws(idata: InferenceDataLike, expected_draws: int) -> None:
    """Reject a posterior with draws the sampler never delivered.

    An interrupted run (Ctrl-C, a notebook cell timeout sending SIGINT
    through the kernel) or a dead chain does not raise: nutpie's Python
    wrapper catches the interrupt, aborts, and returns the partial trace,
    and PyMC's own backend likewise returns whatever it has. The damage
    takes two shapes:

    - **Truncated:** every chain stopped early, so the `draw` dimension is
      shorter than requested (zero-length when the interrupt lands during
      tuning).
    - **NaN-padded:** chains stopped at different points; nutpie pre-fills
      its trace buffers with NaN and zero-fills the boolean `diverging`
      stat, so slower chains come back as silently NaN-padded draws with
      zero divergences. Every downstream statistic then degrades to NaN
      without a single error (the proxy-svar docs deploy shipped exactly
      that).

    A draw counts as missing only when *every* value of *every* float
    posterior variable is NaN at that (chain, draw) — the padding
    signature. Legitimate NaN in an individual variable never trips this.

    Args:
        idata: The container returned by `pm.sample`.
        expected_draws: Post-tuning draws requested per chain.

    Raises:
        RuntimeError: If the posterior is truncated or NaN-padded,
            reporting the delivered draw count per affected chain.
    """
    posterior = get_group_dataset(idata, "posterior")
    n_draws = int(posterior.sizes.get("draw", 0))

    per_chain = []
    if n_draws < expected_draws:
        per_chain.append(f"every chain delivered at most {n_draws} of {expected_draws} draws")
    else:
        missing = None
        for var in posterior.data_vars.values():
            if not np.issubdtype(var.dtype, np.floating):
                continue
            reduce_dims = [d for d in var.dims if d not in ("chain", "draw")]
            var_all_nan = var.isnull().all(dim=reduce_dims) if reduce_dims else var.isnull()
            missing = var_all_nan if missing is None else missing & var_all_nan
        if missing is None or not bool(missing.any()):
            return
        for chain in missing.coords.get("chain", range(missing.sizes["chain"])).values:
            n_missing = int(missing.sel(chain=chain).sum())
            if n_missing:
                per_chain.append(f"chain {chain} delivered {n_draws - n_missing} of {expected_draws} draws")

    raise RuntimeError(
        "Sampling returned an incomplete posterior: "
        + "; ".join(per_chain)
        + ". Missing draws come back NaN-padded (or absent), so every "
        "downstream statistic (ESS, r_hat, IRFs, forecasts) would silently "
        "degrade to NaN. This happens when the sampler is interrupted "
        "mid-run (Ctrl-C, a notebook cell timeout) or a chain fails; the "
        "backend returns the partial trace instead of raising. Re-run the "
        "fit to completion, or reduce its cost (fewer draws/tune steps or "
        "chains) so it fits the available time budget."
    )


def _default_nuts_sampler() -> Literal["pymc", "nutpie"]:
    """Return 'nutpie' if installed, otherwise 'pymc'."""
    try:
        import nutpie  # noqa: F401
    except ImportError:
        return "pymc"
    else:
        return "nutpie"


def _default_progressbar() -> bool:
    """Show the sampler progress bar, except during documentation builds.

    Sphinx sets ``IMPULSO_DOCS_BUILD=1`` so rendered notebooks do not embed the
    live progress widget. Normal usage is unaffected.
    """
    return os.environ.get("IMPULSO_DOCS_BUILD") != "1"


class NUTSSampler(ImpulsoModel):
    """NUTS sampler configuration for PyMC.

    Attributes:
        draws: Number of posterior draws per chain.
        tune: Number of tuning steps per chain.
        chains: Number of independent chains.
        cores: Number of CPU cores. None = auto-detect.
        target_accept: Target acceptance rate for NUTS.
        random_seed: Random seed for reproducibility.
        nuts_sampler: NUTS backend. Auto-detects nutpie if installed.
        progressbar: Show the sampler progress bar. Defaults to True, but off
            during documentation builds (``IMPULSO_DOCS_BUILD=1``).
        nuts_sampler_kwargs: Extra keyword arguments forwarded verbatim to
            the NUTS backend (`pm.sample(nuts_sampler_kwargs=...)`). Useful
            for backend-specific adaptation options — e.g. nutpie's
            `low_rank_modified_mass_matrix=True`, which handles the
            ill-conditioned posteriors that arise in large VARs with many
            near-collinear lag regressors, where diagonal mass-matrix
            adaptation mixes poorly.
    """

    draws: int = Field(1000, ge=1)
    tune: int = Field(1000, ge=0)
    chains: int = Field(4, ge=1)
    cores: int | None = Field(None, ge=1)
    target_accept: float = Field(0.8, gt=0, lt=1)
    random_seed: int | None = None
    nuts_sampler: Literal["pymc", "nutpie"] = Field(default_factory=_default_nuts_sampler)
    progressbar: bool = Field(default_factory=_default_progressbar)
    nuts_sampler_kwargs: dict | None = None

    def sample(self, model: pm.Model) -> InferenceDataLike:
        """Run NUTS sampling on the given PyMC model.

        Args:
            model: A fully specified PyMC model.

        Returns:
            ArviZ InferenceData with posterior and log_likelihood groups.

        Raises:
            RuntimeError: If the returned posterior is incomplete — an
                interrupted or failed nutpie run comes back as silently
                NaN-padded draws (see `_raise_on_incomplete_draws`).
        """
        with model:
            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                nuts_sampler=self.nuts_sampler,
                progressbar=self.progressbar,
                nuts_sampler_kwargs=self.nuts_sampler_kwargs or {},
                idata_kwargs={"log_likelihood": True},
            )
        _raise_on_incomplete_draws(idata, self.draws)
        return idata
