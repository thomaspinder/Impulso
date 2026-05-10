"""Tests for public API re-exports."""


class TestPublicAPI:
    def test_var_importable(self):
        from impulso import VAR

        assert VAR is not None

    def test_var_data_importable(self):
        from impulso import VARData

        assert VARData is not None

    def test_select_lag_order_importable(self):
        from impulso import select_lag_order

        assert select_lag_order is not None

    def test_enable_runtime_checks_importable(self):
        from impulso import enable_runtime_checks

        assert callable(enable_runtime_checks)


class TestVolatilityPublicAPI:
    def test_constant_importable_from_impulso(self):
        from impulso import Constant
        from impulso.volatility import Constant as DirectConstant

        assert Constant is DirectConstant

    def test_volatility_process_importable_from_impulso(self):
        from impulso import VolatilityProcess
        from impulso.protocols import VolatilityProcess as DirectVolatilityProcess

        assert VolatilityProcess is DirectVolatilityProcess

    def test_volatility_names_in_all(self):
        import impulso

        assert "Constant" in impulso.__all__
        assert "VolatilityProcess" in impulso.__all__
