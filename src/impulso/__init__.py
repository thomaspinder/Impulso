"""Impulso: Bayesian Vector Autoregression in Python."""

from impulso._lag_selection import select_lag_order
from impulso.data import VARData
from impulso.spec import VAR

__all__ = [
    "VAR",
    "Cholesky",
    "FEVDResult",
    "FittedSV",
    "FittedVAR",
    "ForecastResult",
    "HDIResult",
    "HistoricalDecompositionResult",
    "IRFResult",
    "IdentifiedVAR",
    "LagOrderResult",
    "MinnesotaPrior",
    "NUTSSampler",
    "SVData",
    "SVDefaultPrior",
    "SVForecastResult",
    "SignRestriction",
    "StochasticVolatility",
    "VARData",
    "VolatilityResult",
    "enable_runtime_checks",
    "select_lag_order",
]


def __getattr__(name: str):
    """Lazy imports for types not needed at import time."""
    _lazy_imports = {
        "FittedVAR": "impulso.fitted",
        "IdentifiedVAR": "impulso.identified",
        "Cholesky": "impulso.identification",
        "SignRestriction": "impulso.identification",
        "MinnesotaPrior": "impulso.priors",
        "NUTSSampler": "impulso.samplers",
        "ForecastResult": "impulso.results",
        "IRFResult": "impulso.results",
        "FEVDResult": "impulso.results",
        "HistoricalDecompositionResult": "impulso.results",
        "HDIResult": "impulso.results",
        "LagOrderResult": "impulso.results",
        "SVData": "impulso.sv.data",
        "StochasticVolatility": "impulso.sv.spec",
        "FittedSV": "impulso.sv.fitted",
        "SVDefaultPrior": "impulso.sv.priors",
        "VolatilityResult": "impulso.results",
        "SVForecastResult": "impulso.results",
    }
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module 'impulso' has no attribute {name!r}")


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
    import impulso.sv.data
    import impulso.sv.fitted
    import impulso.sv.priors
    import impulso.sv.spec

    for mod in [
        impulso.data,
        impulso.spec,
        impulso.fitted,
        impulso.identified,
        impulso.sv.data,
        impulso.sv.spec,
        impulso.sv.fitted,
        impulso.sv.priors,
    ]:
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type):
                with contextlib.suppress(BeartypeDecorHintPep484585Exception):
                    setattr(mod, name, beartype(obj))
