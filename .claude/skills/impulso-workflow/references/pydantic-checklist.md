# Pydantic Usage Checklist

Use this checklist when auditing Pydantic `BaseModel` usage. Impulso uses frozen Pydantic models as the backbone of its entire pipeline (`VARData` -> `VAR` -> `FittedVAR` -> `IdentifiedVAR` and all result types).

## Model Configuration

- [ ] **Shared base class**: If multiple models repeat the same `model_config` (e.g., `frozen=True, arbitrary_types_allowed=True`), consider a shared `ImpulsoBaseModel` to reduce duplication
- [ ] **`frozen=True` applied consistently**: All pipeline and result models should be frozen for immutability guarantees
- [ ] **`arbitrary_types_allowed` justified**: Each model using this should need it (e.g., holds numpy arrays or InferenceData). Models with only primitive/Pydantic-native fields should NOT set it
- [ ] **`ConfigDict` vs `model_config`**: All models should use `model_config = ConfigDict(...)` (Pydantic v2 style), not the v1 `class Config:` pattern
- [ ] **`repr` control**: Large array fields should use `Field(repr=False)` to keep repr output readable
- [ ] **`str_strip_whitespace`**: String fields (e.g., variable names) should consider `str_strip_whitespace=True` if user input is expected

## Field Definitions

- [ ] **`Field()` constraints on numerics**: All numeric parameters should have appropriate bounds (`ge`, `gt`, `le`, `lt`) — e.g., lag counts >= 1, probabilities in (0, 1)
- [ ] **`Field()` descriptions**: Public-facing fields should have `description=` for auto-generated docs and schema export
- [ ] **Default values**: Defaults should be sensible and documented. Required fields should use `Field(...)` (no default) to make this explicit
- [ ] **`Literal` types**: Where a field accepts a fixed set of string values, use `Literal["a", "b"]` rather than `str` with runtime validation
- [ ] **Annotated types**: Consider `Annotated[float, Field(gt=0)]` pattern for reusable constrained types (e.g., `PositiveFloat = Annotated[float, Field(gt=0)]`)

## Validators

- [ ] **`model_validator` vs `field_validator`**: `model_validator(mode="after")` is correct for cross-field validation. `field_validator` should be used for single-field transforms
- [ ] **Validator mode**: `mode="before"` validators should only be used when transforming raw input (e.g., coercing a DataFrame). `mode="after"` is the default and usually correct
- [ ] **No side effects in validators**: Validators should validate/transform, not trigger I/O, logging, or computation
- [ ] **`object.__setattr__` in frozen models**: If used (e.g., to make arrays read-only), it should be limited to `model_validator(mode="after")` and clearly commented. Prefer `model_post_init` if available

## Serialization and Round-Tripping

- [ ] **`model_dump()` works**: Can all models produce a dictionary via `model_dump()`? Models holding numpy arrays or ArviZ InferenceData may need custom serializers
- [ ] **`model_validate()` works**: Can models be reconstructed from `model_dump()` output? This matters for caching, checkpointing, and testing
- [ ] **Custom serializers**: If round-tripping is needed for non-native types, use `@field_serializer` / `@field_validator(mode="before")` pairs or `PlainSerializer` / `PlainValidator`
- [ ] **JSON schema**: If API/CLI exposure is planned, verify `model_json_schema()` produces clean output

## Type Safety with Non-Pydantic Types

- [ ] **`arbitrary_types_allowed` minimised**: Each use should be necessary. Where possible, use `BeforeValidator` to coerce input into a validated form
- [ ] **Array validation**: Numpy arrays accepted as fields should have validators checking shape, dtype, and dimensionality where relevant
- [ ] **InferenceData validation**: Fields holding ArviZ `InferenceData` should validate expected groups exist (e.g., `posterior`, `observed_data`)
- [ ] **DatetimeIndex validation**: Fields holding pandas `DatetimeIndex` should validate frequency, monotonicity, or other domain constraints

## Pydantic v2 Feature Adoption

- [ ] **`@computed_field`**: Read-only derived properties that should appear in serialization/repr — use `@computed_field` instead of manual `@property` + custom serializer
- [ ] **`TypeAdapter`**: For validating/serializing types outside a model context (e.g., standalone array validation)
- [ ] **`@validate_call`**: Consider for public functions that accept complex arguments — gives automatic argument validation without a wrapping model
- [ ] **Discriminated unions**: If a field can hold one of several model types (e.g., different prior types), use `Discriminator` for efficient parsing
- [ ] **`model_post_init`**: Prefer over `model_validator(mode="after")` for post-construction setup that doesn't need to return `self`

## Performance

- [ ] **Construction overhead**: If models are created in hot loops (e.g., per-sample or per-iteration), consider whether the validation overhead is acceptable. Use `model_construct()` for trusted internal data
- [ ] **`model_construct()` for internal use**: When building models from already-validated data (e.g., inside `.fit()` returning a `FittedVAR`), `model_construct()` skips validation and is significantly faster
- [ ] **Frozen model copying**: If a frozen model needs modification (e.g., adding a field), use `model_copy(update={...})` rather than reconstructing from scratch
