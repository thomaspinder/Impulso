"""Tests for impulso package-level lazy imports."""

import pytest

import impulso


def test_lazy_import_happy_path():
    """Lazy __getattr__ resolves known names."""
    assert impulso.FittedVAR is not None
    assert impulso.MinnesotaPrior is not None
    assert impulso.NUTSSampler is not None


def test_lazy_import_unknown_raises():
    """Lazy __getattr__ raises AttributeError for unknown names."""
    with pytest.raises(AttributeError, match="does_not_exist"):
        _ = impulso.does_not_exist


def test_lazy_import_unknown_message_shape():
    """Unknown names report the module and the repr'd attribute name."""
    with pytest.raises(AttributeError, match=r"module 'impulso' has no attribute 'does_not_exist'"):
        _ = impulso.does_not_exist


@pytest.mark.parametrize("name", impulso.__all__)
def test_all_names_resolve(name):
    """Every name advertised in __all__ is resolvable via getattr."""
    assert getattr(impulso, name) is not None


def test_lazy_imports_is_module_level_constant():
    """The lazy-import table is built once at import time, not per access."""
    table = impulso._LAZY_IMPORTS
    assert isinstance(table, dict)
    # Same object on repeated access -- not rebuilt.
    assert impulso._LAZY_IMPORTS is table


def test_lazy_imports_covers_all_lazy_names():
    """Names in __all__ are either real globals or present in the lazy table."""
    eager = set(vars(impulso))
    missing = [name for name in impulso.__all__ if name not in eager and name not in impulso._LAZY_IMPORTS]
    assert missing == []


def test_lazy_imports_targets_define_their_names():
    """Each lazy entry points at a module that actually defines the name."""
    import importlib

    for name, module_path in impulso._LAZY_IMPORTS.items():
        module = importlib.import_module(module_path)
        assert hasattr(module, name), f"{module_path} does not define {name}"


def test_dir_contains_all_public_names():
    """dir(impulso) exposes every name in __all__."""
    listing = dir(impulso)
    assert set(impulso.__all__) <= set(listing)


def test_dir_is_sorted_and_unique():
    """dir(impulso) returns a sorted, duplicate-free listing."""
    listing = dir(impulso)
    assert listing == sorted(listing)
    assert len(listing) == len(set(listing))


def test_dir_includes_real_globals():
    """dir(impulso) still includes the eagerly-imported globals."""
    listing = set(dir(impulso))
    assert {"VAR", "VARData", "select_lag_order", "enable_runtime_checks"} <= listing
