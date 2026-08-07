"""Base model classes for Impulso.

Every domain model inherits from one of the two bases here, so the conventions
that bind adapters across every seam are recorded alongside them.

Adapter discriminator convention:
    A concrete adapter declares its registry key as `name: Literal["x"] = "x"`,
    never `name: ClassVar[str] = "x"`. The Literal form is the Pydantic v2
    idiom: it makes `name` a real instance attribute, so a mismatch raises
    `ValidationError` at construction (`Constant(name="other")`) and again on
    post-construction mutation under `frozen=True`, and the key participates in
    `model_dump` / `model_validate` round-trips. The sharp edge is that
    class-level access (`Constant.name`) does *not* work with this pattern — a
    registry needing the key value without an instance must hardcode the
    literal string.

    Only seams whose Protocol declares `name` carry a discriminator:
    `VolatilityProcess`, `ErrorDistribution` and `SVDynamics` (and the
    `ConjugateVolatility` base, for its break adapters). `Prior`, `Sampler` and
    `IdentificationScheme` deliberately declare none, so their adapters carry
    no discriminator and are selected by type or by an estimator-local registry
    instead.
"""

from pydantic import BaseModel, ConfigDict


class ImpulsoBaseModel(BaseModel):
    """Base model with frozen + arbitrary_types_allowed config.

    Use for models that hold numpy arrays, InferenceData, or
    other non-standard types.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ImpulsoModel(BaseModel):
    """Base model with frozen config only.

    Use for models that only hold standard Python types.
    """

    model_config = ConfigDict(frozen=True)
