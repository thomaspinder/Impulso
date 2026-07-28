"""Impulso: Bayesian Vector Autoregression in Python."""

from typing import TYPE_CHECKING

from impulso._lag_selection import select_lag_order
from impulso.data import VARData
from impulso.spec import VAR

if TYPE_CHECKING:
    from impulso._linalg import lag_matrices
    from impulso._ma import compute_ma_phi
    from impulso.conjugate import ConjugateVAR
    from impulso.conjugate_volatility import ConjugateVolatility, PandemicBreak
    from impulso.fitted import FittedVAR
    from impulso.identification import Cholesky, ProxySVAR, SignRestriction
    from impulso.identified import IdentifiedVAR
    from impulso.priors import MinnesotaPrior, NIWPrior
    from impulso.protocols import VolatilityProcess
    from impulso.results import (
        ConditionalForecastResult,
        CounterfactualResult,
        DynamicMultiplierResult,
        FEVDResult,
        ForecastResult,
        HDIResult,
        HistoricalDecompositionResult,
        IRFResult,
        LagOrderResult,
        ScenarioResult,
        SVForecastResult,
        VolatilityResult,
    )
    from impulso.samplers import NUTSSampler
    from impulso.scenario import ShockPath, VariablePath
    from impulso.sv.data import SVData
    from impulso.sv.fitted import FittedSV
    from impulso.sv.priors import SVDefaultPrior
    from impulso.sv.spec import StochasticVolatility
    from impulso.volatility import Constant

__all__ = [
    "VAR",
    "Cholesky",
    "ConditionalForecastResult",
    "ConjugateVAR",
    "ConjugateVolatility",
    "Constant",
    "CounterfactualResult",
    "DynamicMultiplierResult",
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
    "NIWPrior",
    "NUTSSampler",
    "PandemicBreak",
    "ProxySVAR",
    "SVData",
    "SVDefaultPrior",
    "SVForecastResult",
    "ScenarioResult",
    "ShockPath",
    "SignRestriction",
    "StochasticVolatility",
    "VARData",
    "VariablePath",
    "VolatilityProcess",
    "VolatilityResult",
    "compute_ma_phi",
    "enable_runtime_checks",
    "lag_matrices",
    "select_lag_order",
]


_LAZY_IMPORTS: dict[str, str] = {
    "FittedVAR": "impulso.fitted",
    "IdentifiedVAR": "impulso.identified",
    "Cholesky": "impulso.identification",
    "ProxySVAR": "impulso.identification",
    "SignRestriction": "impulso.identification",
    "MinnesotaPrior": "impulso.priors",
    "NIWPrior": "impulso.priors",
    "ConjugateVAR": "impulso.conjugate",
    "ConjugateVolatility": "impulso.conjugate_volatility",
    "PandemicBreak": "impulso.conjugate_volatility",
    "NUTSSampler": "impulso.samplers",
    "ForecastResult": "impulso.results",
    "ConditionalForecastResult": "impulso.results",
    "CounterfactualResult": "impulso.results",
    "ScenarioResult": "impulso.results",
    "DynamicMultiplierResult": "impulso.results",
    "ShockPath": "impulso.scenario",
    "VariablePath": "impulso.scenario",
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
    "Constant": "impulso.volatility",
    "VolatilityProcess": "impulso.protocols",
    "compute_ma_phi": "impulso._ma",
    "lag_matrices": "impulso._linalg",
}
"""Map of lazily-exported name to the module that defines it.

Built once at import time so that `__getattr__` does not rebuild it on every
lazy attribute access.
"""


def __getattr__(name: str):
    """Lazy imports for types not needed at import time."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'impulso' has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the module's attributes, including the lazily-imported ones.

    Module-level `__getattr__` hides lazy names from the default `dir()`,
    which degrades REPL completion and IDE discovery. Returning the union of
    the real globals and `__all__` restores it.

    Returns:
        Sorted attribute names.
    """
    return sorted(set(globals()) | set(__all__))


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
