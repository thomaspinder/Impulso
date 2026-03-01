"""Impulso: Bayesian Vector Autoregression in Python."""

from impulso._lag_selection import select_lag_order
from impulso.data import VARData
from impulso.spec import VAR

__all__ = [
    "VAR",
    "VARData",
    "enable_runtime_checks",
    "select_lag_order",
]


def enable_runtime_checks() -> None:
    """Enable beartype runtime type checking on public API.

    Intended for use in test suites. Wraps public functions and methods
    with beartype decorators for runtime validation.
    """
    import contextlib

    from beartype import beartype
    from beartype.roar import BeartypeDecorHintPep484585Exception

    import impulso.data
    import impulso.fitted
    import impulso.identified
    import impulso.spec

    for mod in [impulso.data, impulso.spec, impulso.fitted, impulso.identified]:
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type):
                with contextlib.suppress(BeartypeDecorHintPep484585Exception):
                    setattr(mod, name, beartype(obj))
