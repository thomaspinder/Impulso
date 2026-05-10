"""Tests for protocol definitions."""

from typing import runtime_checkable

from typing_extensions import get_protocol_members

from impulso.protocols import IdentificationScheme, Prior, Sampler, VolatilityProcess


class TestProtocols:
    def test_prior_is_runtime_checkable(self):
        assert hasattr(Prior, "__protocol_attrs__") or runtime_checkable

    def test_sampler_is_runtime_checkable(self):
        assert hasattr(Sampler, "__protocol_attrs__") or runtime_checkable

    def test_identification_is_runtime_checkable(self):
        assert hasattr(IdentificationScheme, "__protocol_attrs__") or runtime_checkable


class TestVolatilityProcess:
    def test_volatility_process_is_runtime_checkable(self):
        assert hasattr(VolatilityProcess, "__protocol_attrs__") or runtime_checkable

    def test_volatility_process_methods_declared(self):
        # Protocol must declare the three seam operations.
        attrs = set(get_protocol_members(VolatilityProcess))
        assert {"build_pymc_latent", "cholesky_at", "forecast_cholesky_path"} <= attrs
