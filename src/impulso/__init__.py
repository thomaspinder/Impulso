"""Impulso: Bayesian Vector Autoregression in Python."""

from typing import TYPE_CHECKING

from impulso._lag_selection import select_lag_order
from impulso._stationarity import adf_test, integration_order, johansen_test, kpss_test
from impulso.data import VARData
from impulso.spec import VAR

if TYPE_CHECKING:
    from types import ModuleType

    from impulso._granger import toda_yamamoto
    from impulso._linalg import lag_matrices
    from impulso._ma import compute_ma_phi
    from impulso.conjugate import ConjugateVAR
    from impulso.conjugate_volatility import ConjugateVolatility, PandemicBreak
    from impulso.evidence import EvidenceComparison, ModelEvidence, compare_evidence
    from impulso.fitted import FittedVAR
    from impulso.identification import Cholesky, LongRunRestriction, ProxySVAR, SignRestriction, ZeroSignRestriction
    from impulso.identified import IdentifiedVAR
    from impulso.observation import Gaussian, StudentT
    from impulso.priors import MinnesotaPrior, NIWPrior
    from impulso.protocols import ErrorDistribution, VolatilityProcess
    from impulso.results import (
        CointegrationTestResult,
        ConditionalForecastResult,
        CounterfactualResult,
        DynamicMultiplierResult,
        FEVDResult,
        ForecastResult,
        GrangerCausalityResult,
        HDIResult,
        HistoricalDecompositionResult,
        IntegrationOrderResult,
        IRFResult,
        LagOrderResult,
        ScenarioResult,
        StationarityTestResult,
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
    "CointegrationTestResult",
    "ConditionalForecastResult",
    "ConjugateVAR",
    "ConjugateVolatility",
    "Constant",
    "CounterfactualResult",
    "DynamicMultiplierResult",
    "ErrorDistribution",
    "EvidenceComparison",
    "FEVDResult",
    "FittedSV",
    "FittedVAR",
    "ForecastResult",
    "Gaussian",
    "GrangerCausalityResult",
    "HDIResult",
    "HistoricalDecompositionResult",
    "IRFResult",
    "IdentifiedVAR",
    "IntegrationOrderResult",
    "LagOrderResult",
    "LongRunRestriction",
    "MinnesotaPrior",
    "ModelEvidence",
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
    "StationarityTestResult",
    "StochasticVolatility",
    "StudentT",
    "VARData",
    "VariablePath",
    "VolatilityProcess",
    "VolatilityResult",
    "ZeroSignRestriction",
    "adf_test",
    "compare_evidence",
    "compute_ma_phi",
    "enable_runtime_checks",
    "integration_order",
    "johansen_test",
    "kpss_test",
    "lag_matrices",
    "select_lag_order",
    "toda_yamamoto",
]


_LAZY_IMPORTS: dict[str, str] = {
    "FittedVAR": "impulso.fitted",
    "IdentifiedVAR": "impulso.identified",
    "Cholesky": "impulso.identification",
    "LongRunRestriction": "impulso.identification",
    "ProxySVAR": "impulso.identification",
    "SignRestriction": "impulso.identification",
    "ZeroSignRestriction": "impulso.identification",
    "MinnesotaPrior": "impulso.priors",
    "NIWPrior": "impulso.priors",
    "ConjugateVAR": "impulso.conjugate",
    "ConjugateVolatility": "impulso.conjugate_volatility",
    "PandemicBreak": "impulso.conjugate_volatility",
    "ModelEvidence": "impulso.evidence",
    "EvidenceComparison": "impulso.evidence",
    "compare_evidence": "impulso.evidence",
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
    "StationarityTestResult": "impulso.results",
    "CointegrationTestResult": "impulso.results",
    "IntegrationOrderResult": "impulso.results",
    "SVData": "impulso.sv.data",
    "StochasticVolatility": "impulso.sv.spec",
    "FittedSV": "impulso.sv.fitted",
    "SVDefaultPrior": "impulso.sv.priors",
    "VolatilityResult": "impulso.results",
    "SVForecastResult": "impulso.results",
    "Constant": "impulso.volatility",
    "Gaussian": "impulso.observation",
    "StudentT": "impulso.observation",
    "ErrorDistribution": "impulso.protocols",
    "VolatilityProcess": "impulso.protocols",
    "GrangerCausalityResult": "impulso.results",
    "toda_yamamoto": "impulso._granger",
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


def _bind_deferred_imports(module: "ModuleType") -> None:
    """Materialise a module's `if TYPE_CHECKING:` imports for real.

    Beartype resolves stringified annotations (e.g. `-> "ForecastResult"`,
    `posterior: "xr.Dataset"`) against the defining module's namespace when
    the wrapped method is *called*. Names imported only under `TYPE_CHECKING`
    are absent from that namespace, so every such call would raise
    `BeartypeCallHintForwardRefException`. Binding them for real keeps those
    annotations resolvable; the extra import cost is acceptable because
    runtime checking is opt-in and intended for test suites.

    Args:
        module: Module whose deferred imports should be bound.
    """
    import ast
    import contextlib
    import importlib
    import inspect

    def _is_type_checking(test: ast.expr) -> bool:
        return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )

    with contextlib.suppress(OSError, TypeError, SyntaxError):
        tree = ast.parse(inspect.getsource(module))
        for node in tree.body:
            if not (isinstance(node, ast.If) and _is_type_checking(node.test)):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.Import):
                    for alias in stmt.names:
                        # `import a.b as ab` binds the submodule; `import a.b` binds `a`.
                        imported = alias.name if alias.asname else alias.name.split(".")[0]
                        setattr(module, alias.asname or imported, importlib.import_module(imported))
                elif isinstance(stmt, ast.ImportFrom) and stmt.module and not stmt.level:
                    source = importlib.import_module(stmt.module)
                    for alias in stmt.names:
                        setattr(module, alias.asname or alias.name, getattr(source, alias.name))


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
        _bind_deferred_imports(mod)
        for name in dir(mod):
            obj = getattr(mod, name)
            # Only classes *defined* here: `dir(mod)` also surfaces imported
            # names (typing.Protocol, impulso.volatility.Constant, ...), and
            # wrapping those would mutate classes the module does not own.
            if isinstance(obj, type) and getattr(obj, "__module__", None) == mod.__name__:
                with contextlib.suppress(BeartypeDecorHintPep484585Exception):
                    setattr(mod, name, beartype(obj))
