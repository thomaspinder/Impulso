"""Tests for public API re-exports."""

import subprocess
import sys

# Driven in a fresh interpreter: enable_runtime_checks() mutates the library's
# classes in place, so the beartype wrapping must not leak into the rest of the
# suite. No MCMC — the posterior is synthetic.
_RUNTIME_CHECKS_SCRIPT = """
import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

import impulso

impulso.enable_runtime_checks()

import impulso.fitted
import impulso.results
import impulso.sv.spec
from beartype.roar import BeartypeCallHintViolation
from impulso import VAR, VARData
from impulso.fitted import FittedVAR
from impulso.identification import Cholesky
from impulso.volatility import Constant

# TYPE_CHECKING-only annotations must be resolvable once checks are on:
# beartype resolves them against the defining module at call time.
assert impulso.fitted.ForecastResult is impulso.results.ForecastResult
assert impulso.sv.spec.xr is xr

rng = np.random.default_rng(0)
index = pd.date_range("2000-01-01", periods=30, freq="QS")
frame = pd.DataFrame(rng.standard_normal((30, 2)) * 0.1, columns=["y1", "y2"], index=index)

data = VARData.from_df(frame, endog=["y1", "y2"])
assert data.endog.shape == (30, 2)
assert VAR(lags=1).lags == 1

n_chains, n_draws, n_vars = 2, 5, 2
B = rng.standard_normal((n_chains, n_draws, n_vars, n_vars)) * 0.2
intercept = rng.standard_normal((n_chains, n_draws, n_vars)) * 0.01
L = np.zeros((n_chains, n_draws, n_vars, n_vars))
for c in range(n_chains):
    for d in range(n_draws):
        A = rng.standard_normal((n_vars, n_vars)) * 0.3
        L[c, d] = np.linalg.cholesky(A @ A.T + np.eye(n_vars))
posterior = xr.Dataset({
    "B": xr.DataArray(B, dims=["chain", "draw", "var", "coeff"]),
    "intercept": xr.DataArray(intercept, dims=["chain", "draw", "var"]),
    "L": xr.DataArray(L, dims=["chain", "draw", "var1", "var2"]),
})

fitted = FittedVAR(
    idata=az.InferenceData(posterior=posterior),
    n_lags=1,
    data=data,
    var_names=["y1", "y2"],
    volatility=Constant(),
)
assert fitted.sigma().shape == (n_chains, n_draws, n_vars, n_vars)
# Return annotations below are TYPE_CHECKING-only forward refs
# ("ForecastResult", "IdentifiedVAR"), which beartype resolves on call.
assert type(fitted.forecast(steps=2, seed=0)).__name__ == "ForecastResult"
identified = fitted.set_identification_strategy(Cholesky(ordering=["y1", "y2"]))
assert type(identified).__name__ == "IdentifiedVAR"
assert type(identified.impulse_response(horizon=2)).__name__ == "IRFResult"
print("HAPPY_PATH_OK")

try:
    fitted.forecast(steps="two")
except BeartypeCallHintViolation:
    print("VIOLATION_CAUGHT")
else:
    raise AssertionError("beartype did not flag steps='two'")
"""


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


class TestRuntimeChecks:
    def test_enable_runtime_checks_drives_pipeline(self):
        """Checks-on run of a synthetic pipeline, in a throwaway interpreter."""
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _RUNTIME_CHECKS_SCRIPT],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert "HAPPY_PATH_OK" in result.stdout, result.stdout
        assert "VIOLATION_CAUGHT" in result.stdout, result.stdout


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


class TestEvidencePublicAPI:
    def test_model_evidence_importable_from_impulso(self):
        from impulso import ModelEvidence
        from impulso.evidence import ModelEvidence as DirectModelEvidence

        assert ModelEvidence is DirectModelEvidence

    def test_evidence_comparison_importable_from_impulso(self):
        from impulso import EvidenceComparison
        from impulso.evidence import EvidenceComparison as DirectEvidenceComparison

        assert EvidenceComparison is DirectEvidenceComparison

    def test_compare_evidence_importable_from_impulso(self):
        from impulso import compare_evidence
        from impulso.evidence import compare_evidence as direct_compare_evidence

        assert compare_evidence is direct_compare_evidence

    def test_evidence_names_in_all(self):
        import impulso

        assert "ModelEvidence" in impulso.__all__
        assert "EvidenceComparison" in impulso.__all__
        assert "compare_evidence" in impulso.__all__


class TestErrorDistributionPublicAPI:
    def test_gaussian_importable_from_impulso(self):
        from impulso import Gaussian
        from impulso.observation import Gaussian as DirectGaussian

        assert Gaussian is DirectGaussian

    def test_student_t_importable_from_impulso(self):
        from impulso import StudentT
        from impulso.observation import StudentT as DirectStudentT

        assert StudentT is DirectStudentT

    def test_error_distribution_protocol_importable_from_impulso(self):
        from impulso import ErrorDistribution
        from impulso.protocols import ErrorDistribution as DirectErrorDistribution

        assert ErrorDistribution is DirectErrorDistribution

    def test_names_in_all(self):
        import impulso

        assert {"Gaussian", "StudentT", "ErrorDistribution"} <= set(impulso.__all__)


class TestIdentificationPublicAPI:
    def test_long_run_restriction_importable_from_impulso(self):
        from impulso import LongRunRestriction
        from impulso.identification import LongRunRestriction as Direct

        assert LongRunRestriction is Direct

    def test_long_run_restriction_in_all(self):
        import impulso

        assert "LongRunRestriction" in impulso.__all__

    def test_max_share_importable_from_impulso(self):
        from impulso import MaxShare
        from impulso.identification import MaxShare as Direct

        assert MaxShare is Direct

    def test_max_share_in_all(self):
        import impulso

        assert "MaxShare" in impulso.__all__
