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

    def test_dir_includes_public_api(self):
        import impulso

        names = dir(impulso)
        for expected in impulso.__all__:
            assert expected in names
