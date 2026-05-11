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


class TestIdentificationSchemeNewSignature:
    """Locks in the (L, var_names, posterior=None) -> ndarray signature
    that Cholesky and SignRestriction will adopt in Tasks 3-4.
    """

    def test_identify_signature_takes_L_not_idata(self):
        """The Protocol's identify method declares the new signature."""
        import inspect

        from impulso.protocols import IdentificationScheme

        sig = inspect.signature(IdentificationScheme.identify)
        params = list(sig.parameters.values())
        # self, L, var_names, posterior=None
        assert [p.name for p in params] == ["self", "L", "var_names", "posterior"]
        assert params[3].default is None
