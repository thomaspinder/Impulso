"""Tests for public API re-exports."""


class TestPublicAPI:
    def test_var_importable(self):
        from litterman import VAR

        assert VAR is not None

    def test_var_data_importable(self):
        from litterman import VARData

        assert VARData is not None

    def test_select_lag_order_importable(self):
        from litterman import select_lag_order

        assert select_lag_order is not None

    def test_enable_runtime_checks_importable(self):
        from litterman import enable_runtime_checks

        assert callable(enable_runtime_checks)
