"""Helpers for importing optional third-party dependencies.

Impulso keeps its core install lean. Features that need a heavier library
(for example the stationarity diagnostics, which need `statsmodels`) declare
an extra in `pyproject.toml` and import the dependency lazily through
`require`, so that a core-only install never pays the import cost and never
fails at `import impulso` time.
"""

import importlib
from types import ModuleType


def require(module: str, *, extra: str) -> ModuleType:
    """Import an optional dependency or raise an actionable error.

    Args:
        module: Importable module name, e.g. `"statsmodels"`.
        extra: Name of the Impulso extra that provides it, e.g.
            `"diagnostics"`.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not installed. The message names the
            extra and gives copy-pasteable install commands.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} is required for this feature. "
            f'Install it with: pip install "impulso[{extra}]" or: uv add "impulso[{extra}]"'
        ) from exc
