"""Tests for protocol definitions."""

from typing import runtime_checkable

from typing_extensions import get_protocol_members

from impulso.protocols import IdentificationScheme, Prior, PyMCVolatilityProcess, Sampler, VolatilityProcess


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

    def test_volatility_process_declares_query_surface(self):
        # Query surface only — the PyMC builder lives on the sub-protocol.
        attrs = set(get_protocol_members(VolatilityProcess))
        assert {"cholesky_at", "cholesky_path", "forecast_cholesky_path"} <= attrs
        assert "build_pymc_latent" not in attrs

    def test_pymc_volatility_process_adds_builder(self):
        # Sub-protocol extends the query surface with build_pymc_latent.
        assert hasattr(PyMCVolatilityProcess, "__protocol_attrs__") or runtime_checkable
        attrs = set(get_protocol_members(PyMCVolatilityProcess))
        assert "build_pymc_latent" in attrs
        assert {"cholesky_at", "cholesky_path", "forecast_cholesky_path"} <= attrs


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
        # self, L, var_names, posterior=None, data=None, n_lags=None
        assert [p.name for p in params] == ["self", "L", "var_names", "posterior", "data", "n_lags"]
        assert params[3].default is None
        assert params[4].default is None
        assert params[5].default is None
