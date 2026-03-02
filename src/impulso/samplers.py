"""Sampler specifications for posterior inference."""

import arviz as az
import pymc as pm
from pydantic import BaseModel, ConfigDict, Field


class NUTSSampler(BaseModel):
    """NUTS sampler configuration for PyMC.

    Attributes:
        draws: Number of posterior draws per chain.
        tune: Number of tuning steps per chain.
        chains: Number of independent chains.
        cores: Number of CPU cores. None = auto-detect.
        target_accept: Target acceptance rate for NUTS.
        random_seed: Random seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    draws: int = Field(1000, ge=1)
    tune: int = Field(1000, ge=0)
    chains: int = Field(4, ge=1)
    cores: int | None = Field(None, ge=1)
    target_accept: float = Field(0.8, gt=0, lt=1)
    random_seed: int | None = None

    def sample(self, model: pm.Model) -> az.InferenceData:
        """Run NUTS sampling on the given PyMC model.

        Args:
            model: A fully specified PyMC model.

        Returns:
            ArviZ InferenceData with posterior and log_likelihood groups.
        """
        with model:
            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                idata_kwargs={"log_likelihood": True},
            )
        return idata
