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
