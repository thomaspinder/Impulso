"""Tests for the volatility-process seam and adapters."""

import numpy as np
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


class TestConstantBuildPymcLatent:
    def test_returns_lower_triangular_for_n_vars_3(self):
        import pymc as pm

        adapter = Constant()
        with pm.Model():
            L_tensor = adapter.build_pymc_latent(n_vars=3, T=100)
            L_value = L_tensor.eval()  # uses prior values; deterministic shape

        assert L_value.shape == (3, 3)
        # Strictly lower-triangular plus positive diagonal: upper triangle is zero.
        upper = np.triu(L_value, k=1)
        assert np.allclose(upper, 0.0)

    def test_handles_n_vars_1(self):
        import pymc as pm

        adapter = Constant()
        with pm.Model():
            L_tensor = adapter.build_pymc_latent(n_vars=1, T=50)
            assert L_tensor.eval().shape == (1, 1)

    def test_registers_expected_pymc_vars(self):
        import pymc as pm

        adapter = Constant()
        with pm.Model() as model:
            adapter.build_pymc_latent(n_vars=3, T=100)

        var_names = {v.name for v in model.unobserved_RVs}
        assert "sigma_sd" in var_names
        assert "tril_offdiag" in var_names

    def test_n_vars_1_skips_tril_offdiag(self):
        import pymc as pm

        adapter = Constant()
        with pm.Model() as model:
            adapter.build_pymc_latent(n_vars=1, T=50)

        var_names = {v.name for v in model.unobserved_RVs}
        assert "sigma_sd" in var_names
        assert "tril_offdiag" not in var_names
