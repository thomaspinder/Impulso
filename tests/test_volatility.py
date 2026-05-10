"""Tests for the volatility-process seam and adapters."""

import pytest

from impulso.protocols import VolatilityProcess
from impulso.volatility import Constant


class TestConstantAdapter:
    def test_constant_satisfies_protocol(self):
        adapter = Constant()
        assert isinstance(adapter, VolatilityProcess)

    def test_constant_name(self):
        assert Constant().name == "constant"

    def test_constant_is_frozen(self):
        from pydantic import ValidationError

        adapter = Constant()
        with pytest.raises(ValidationError):
            adapter.name = "other"
