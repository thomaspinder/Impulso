"""Tests for protocol definitions.

Two properties are pinned for every Protocol: the exact set of members it
declares (`get_protocol_members`, which works on 3.11 where the stdlib
`__protocol_attrs__` attribute does not yet exist), and that `isinstance`
discriminates a conforming stub from a non-conforming one. The `isinstance`
calls double as the runtime-checkability assertion — `isinstance` against a
Protocol raises `TypeError` unless it is decorated `@runtime_checkable`.
"""

from typing_extensions import get_protocol_members, is_protocol

from impulso.protocols import IdentificationScheme, Prior, PyMCVolatilityProcess, Sampler, VolatilityProcess


class _ConformingPrior:
    def build_priors(self, n_vars, n_lags):
        return {}


class _ConformingSampler:
    def sample(self, model):
        return None


class _ConformingScheme:
    def identify(self, L, var_names, posterior=None, data=None, n_lags=None):
        return L

    def shock_coords(self, n_vars):
        return []


class _ConformingVolatility:
    name = "stub"
    is_time_varying = False

    def cholesky_at(self, posterior, t):
        return None

    def cholesky_path(self, posterior, T):
        return None

    def forecast_cholesky_path(self, posterior, steps, rng):
        return None


class _ConformingPyMCVolatility(_ConformingVolatility):
    def build_pymc_latent(self, n_vars, T, data=None):
        return None


class _Empty:
    """Implements nothing — the negative case for every protocol."""


class TestProtocols:
    def test_prior_declares_build_priors(self):
        assert is_protocol(Prior)
        assert get_protocol_members(Prior) == frozenset({"build_priors"})

    def test_prior_isinstance_discriminates(self):
        assert isinstance(_ConformingPrior(), Prior)
        assert not isinstance(_Empty(), Prior)

    def test_sampler_declares_sample(self):
        assert is_protocol(Sampler)
        assert get_protocol_members(Sampler) == frozenset({"sample"})

    def test_sampler_isinstance_discriminates(self):
        assert isinstance(_ConformingSampler(), Sampler)
        assert not isinstance(_Empty(), Sampler)

    def test_identification_declares_identify_and_shock_coords(self):
        assert is_protocol(IdentificationScheme)
        assert get_protocol_members(IdentificationScheme) == frozenset({"identify", "shock_coords"})

    def test_identification_isinstance_discriminates(self):
        assert isinstance(_ConformingScheme(), IdentificationScheme)
        assert not isinstance(_Empty(), IdentificationScheme)
        assert not isinstance(_ConformingPrior(), IdentificationScheme)


class TestVolatilityProcess:
    def test_volatility_process_declares_query_surface(self):
        # Query surface only — the PyMC builder lives on the sub-protocol.
        assert is_protocol(VolatilityProcess)
        assert get_protocol_members(VolatilityProcess) == frozenset({
            "name",
            "is_time_varying",
            "cholesky_at",
            "cholesky_path",
            "forecast_cholesky_path",
        })

    def test_volatility_process_isinstance_discriminates(self):
        assert isinstance(_ConformingVolatility(), VolatilityProcess)
        assert not isinstance(_Empty(), VolatilityProcess)

    def test_docstring_reserves_the_sv_variable_prefix(self):
        # Adapter authors read the protocol, not ADR-0008; the reserved
        # prefix has to be visible where a custom adapter is written.
        doc = VolatilityProcess.__doc__ or ""
        assert "v{i}_" in doc
        assert "posterior_var_names()" in doc

    def test_pymc_volatility_process_adds_builder(self):
        # Sub-protocol extends the query surface with build_pymc_latent.
        assert is_protocol(PyMCVolatilityProcess)
        assert get_protocol_members(PyMCVolatilityProcess) == frozenset({
            "name",
            "is_time_varying",
            "cholesky_at",
            "cholesky_path",
            "forecast_cholesky_path",
            "build_pymc_latent",
        })

    def test_pymc_volatility_process_isinstance_discriminates(self):
        assert isinstance(_ConformingPyMCVolatility(), PyMCVolatilityProcess)
        # Query-only adapters satisfy the parent protocol but not the sub-protocol.
        assert isinstance(_ConformingVolatility(), VolatilityProcess)
        assert not isinstance(_ConformingVolatility(), PyMCVolatilityProcess)


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


class TestErrorDistribution:
    """The observation-error seam's Protocol (issue #152)."""

    def test_is_runtime_checkable(self):
        from impulso.protocols import ErrorDistribution

        assert is_protocol(ErrorDistribution)

    def test_declares_the_full_surface(self):
        from impulso.protocols import ErrorDistribution

        attrs = set(get_protocol_members(ErrorDistribution))
        assert {
            "name",
            "is_heavy_tailed",
            "build_likelihood",
            "draw_standardised_innovations",
            "variance_inflation",
        } <= attrs

    def test_both_adapters_are_instances(self):
        from impulso.observation import Gaussian, StudentT
        from impulso.protocols import ErrorDistribution

        assert isinstance(Gaussian(), ErrorDistribution)
        assert isinstance(StudentT(), ErrorDistribution)

    def test_volatility_adapter_is_not_an_error_distribution(self):
        from impulso.protocols import ErrorDistribution
        from impulso.volatility import Constant

        assert not isinstance(Constant(), ErrorDistribution)

    def test_error_distribution_adapter_is_not_a_volatility_process(self):
        from impulso.observation import StudentT
        from impulso.protocols import VolatilityProcess

        assert not isinstance(StudentT(), VolatilityProcess)
