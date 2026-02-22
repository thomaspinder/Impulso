"""Tests for protocol definitions."""

from typing import runtime_checkable

from litterman.protocols import IdentificationScheme, Prior, Sampler


class TestProtocols:
    def test_prior_is_runtime_checkable(self):
        assert hasattr(Prior, "__protocol_attrs__") or runtime_checkable

    def test_sampler_is_runtime_checkable(self):
        assert hasattr(Sampler, "__protocol_attrs__") or runtime_checkable

    def test_identification_is_runtime_checkable(self):
        assert hasattr(IdentificationScheme, "__protocol_attrs__") or runtime_checkable
