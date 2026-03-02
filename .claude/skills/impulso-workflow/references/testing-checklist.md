# Test Quality Checklist

## Coverage

- [ ] **Module coverage**: Every public module in the package has a corresponding test file
- [ ] **Function coverage**: Core public functions/methods all have at least one test
- [ ] **Edge cases**: Tests cover boundary conditions (empty inputs, single-element, large inputs)
- [ ] **Error paths**: Tests verify that expected exceptions are raised with `pytest.raises`
- [ ] **Coverage config**: `pyproject.toml` or `.coveragerc` defines coverage settings
- [ ] **Coverage threshold**: A minimum coverage % is enforced in CI (ideally ≥80%)

## Test Quality

- [ ] **Descriptive names**: Test names describe the scenario and expected outcome, e.g., `test_matern32_kernel_returns_positive_definite_matrix`
- [ ] **Single assertion focus**: Each test ideally tests one behaviour (multiple asserts OK if testing one logical thing)
- [ ] **No test interdependence**: Tests don't rely on execution order or shared mutable state
- [ ] **Deterministic**: Tests use fixed random seeds (`np.random.RandomState(123)`) for reproducibility
- [ ] **Fast**: Individual tests complete in <5s. Slow tests marked with `@pytest.mark.slow`
- [ ] **No hardcoded paths**: Tests don't reference absolute file paths

## Pytest Best Practices

- [ ] **`@pytest.mark.parametrize`**: Used to test multiple inputs/configs instead of copy-paste tests
- [ ] **Fixtures**: Shared setup uses `@pytest.fixture`, not repeated code in each test
- [ ] **Fixture scope**: Expensive fixtures (model compilation, data loading) use `scope="module"` or `scope="session"`
- [ ] **`conftest.py`**: Common fixtures defined in `conftest.py` at appropriate directory levels
- [ ] **Markers**: Custom markers defined in `pyproject.toml` (e.g., `slow`, `integration`)
- [ ] **`tmp_path`**: Used for temporary file operations instead of manual cleanup
- [ ] **`approx` / `allclose`**: Floating-point comparisons use `pytest.approx` or `jnp.allclose` with explicit tolerances

## Test Organisation

- [ ] **Mirror source layout**: `tests/` directory structure mirrors `impulso/` structure
- [ ] **No test utilities in source**: Test helpers live in `tests/`, not in the main package
- [ ] **Integration tests separated**: Integration/slow tests clearly separated from unit tests
