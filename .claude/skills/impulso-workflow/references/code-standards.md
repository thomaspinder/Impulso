# Modern Python Code Standards Checklist

Use this checklist when auditing `.py` files. Not every item applies to every file — use judgement.

## Language Features (Python 3.10+)

- [ ] **Modern type unions**: `X | Y` instead of `Union[X, Y]` and `X | None` instead of `Optional[X]`
- [ ] **Modern collection types**: `list[X]`, `dict[K, V]`, `tuple[X, ...]` instead of `List`, `Dict`, `Tuple` from typing
- [ ] **`match` statements**: Where long if/elif chains on type or value exist, consider `match`
- [ ] **`from __future__ import annotations`**: Should be present if supporting <3.10 OR removed if min Python ≥3.10
- [ ] **Walrus operator**: Used where appropriate to avoid redundant computation
- [ ] **f-strings**: No `%` formatting or `.format()` unless there's a specific reason (e.g., logging)
- [ ] **`@dataclass(slots=True)`**: Data classes should use slots where possible for memory efficiency
- [ ] **`type` keyword** (3.12+): For simple type aliases if min Python ≥3.12

## Code Quality

- [ ] **No bare `except:`**: All except clauses should catch specific exceptions
- [ ] **No mutable default arguments**: `def f(x=[])` → `def f(x=None)`
- [ ] **Context managers**: Files, locks, connections use `with` statements
- [ ] **Pathlib over os.path**: `Path` objects preferred for file path manipulation
- [ ] **Enum for constants**: Named constants grouped into `Enum` or `StrEnum` classes where appropriate
- [ ] **No wildcard imports**: `from module import *` should not appear in library code
- [ ] **`__all__` defined**: Public modules should define `__all__` for explicit API surface
- [ ] **Consistent naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants

## Structure and Design

- [ ] **Single responsibility**: Functions/classes do one thing
- [ ] **No god objects**: Classes should be focused, not catch-all containers
- [ ] **Minimal inheritance depth**: Prefer composition; inheritance chains >3 levels are suspect
- [ ] **Abstract base classes**: ABCs used where there's a genuine interface contract, not just for code reuse
- [ ] **Module size**: Files >500 lines should be considered for splitting
- [ ] **Circular imports**: None present (check for deferred imports that suggest circularity)
- [ ] **Dependency inversion**: Core logic doesn't depend on I/O, plotting, or optional deps

## Probabilistic Programming / Domain-Specific

- [ ] **Numerical stability**: Computations in log-space where appropriate (log-likelihoods, log-determinants). No bare `np.exp()` on unbounded values
- [ ] **Cholesky over inverse**: Use Cholesky decomposition for covariance operations rather than direct matrix inversion
- [ ] **Array immutability**: Numpy arrays stored in frozen Pydantic models should be made read-only (`arr.flags.writeable = False`) in validators
- [ ] **Lazy imports**: PyMC, matplotlib, and heavy optional dependencies imported inside methods/functions, not at module top level. Verify this is applied consistently across all modules
- [ ] **Deterministic seeds**: Any code that uses random state should accept a `random_seed` parameter or use PyMC's built-in seeding, not global `np.random` state
- [ ] **Shape assertions**: Array shapes checked at boundaries (function entry/exit) with informative error messages, not silently broadcast

## Protocols and Extensibility

- [ ] **Protocol completeness**: All `Protocol` classes in `protocols.py` define complete method signatures with full type annotations
- [ ] **Protocol compliance**: Every concrete implementation satisfies its Protocol (verify with `ty check` or `isinstance` checks in tests)
- [ ] **Protocol documentation**: Protocols are documented as the public extension API — users should know they can implement `Prior`, `Sampler`, `IdentificationScheme`
- [ ] **No leaky abstractions**: Protocol method signatures don't expose internal types (e.g., shouldn't require PyMC-specific types in the Protocol interface if the Protocol is meant to be backend-agnostic)
