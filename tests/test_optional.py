"""Tests for the optional-dependency import helper."""

import pytest

from impulso._optional import require


def test_require_returns_installed_module():
    mod = require("math", extra="diagnostics")
    assert mod.sqrt(4) == 2


def test_require_missing_module_raises_actionable_importerror():
    with pytest.raises(ImportError) as exc_info:
        require("nonexistent_pkg_xyz", extra="diagnostics")

    message = str(exc_info.value)
    assert "nonexistent_pkg_xyz" in message
    assert "impulso[diagnostics]" in message
    assert "pip install" in message
    assert "uv add" in message


def test_require_names_the_extra_it_is_given():
    with pytest.raises(ImportError, match=r"impulso\[somethingelse\]"):
        require("nonexistent_pkg_xyz", extra="somethingelse")
