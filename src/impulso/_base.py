"""Base model classes for Impulso."""

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
