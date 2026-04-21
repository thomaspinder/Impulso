"""Tests for public SV API exports."""


def test_top_level_imports():
    from impulso import (
        FittedSV,
        StochasticVolatility,
        SVData,
        SVDefaultPrior,
        SVForecastResult,
        VolatilityResult,
    )

    assert SVData is not None
    assert StochasticVolatility is not None
    assert FittedSV is not None
    assert SVDefaultPrior is not None
    assert VolatilityResult is not None
    assert SVForecastResult is not None


def test_subpackage_imports():
    from impulso.sv import (
        FittedSV,
        StochasticVolatility,
        SVData,
        SVDefaultPrior,
        SVPrior,
    )

    assert SVData is not None
    assert StochasticVolatility is not None
    assert FittedSV is not None
    assert SVDefaultPrior is not None
    assert SVPrior is not None
