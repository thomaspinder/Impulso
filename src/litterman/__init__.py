"""Litterman: Bayesian Vector Autoregression in Python."""

from litterman._lag_selection import select_lag_order
from litterman.data import VARData
from litterman.spec import VAR

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

    import litterman.data
    import litterman.fitted
    import litterman.identified
    import litterman.spec

    for mod in [litterman.data, litterman.spec, litterman.fitted, litterman.identified]:
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type):
                with contextlib.suppress(BeartypeDecorHintPep484585Exception):
                    setattr(mod, name, beartype(obj))
