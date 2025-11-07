# llm-apple Tests

This directory contains the test suite for llm-apple, including unit tests and integration tests.

## Test Structure

### Unit Tests (Using Mocks)
These tests use mocks to test the code without requiring Apple Intelligence to be available:

- **`test_apple_model.py`** - Tests for the AppleModel class initialization and session management
- **`test_availability.py`** - Tests for Apple Intelligence availability checking
- **`test_execute.py`** - Tests for the execute method and response generation
- **`test_parameter_passing.py`** - Tests for parameter passing to the Apple Foundation Models API
- **`test_registration.py`** - Tests for plugin registration with llm
- **`test_tools.py`** - Tests for tool calling functionality (mocked)

### Integration Tests (Requires Apple Intelligence)
These tests run against the actual Apple Foundation Models API and require Apple Intelligence to be available on your system:

- **`test_integration_tools.py`** - End-to-end tests for tool calling with real API calls

Integration tests will be automatically skipped if Apple Intelligence is not available.

## Running Tests

### Run All Unit Tests
```bash
uv run pytest tests/ -v
```

### Run Specific Test File
```bash
uv run pytest tests/test_tools.py -v
```

### Run Integration Tests
Integration tests will only run if Apple Intelligence is available on your system:

```bash
# Run integration tests (will skip if Apple Intelligence not available)
uv run pytest tests/test_integration_tools.py -v -s

# Show skip reasons
uv run pytest tests/test_integration_tools.py -v -s -rs
```

### Run Tests with Coverage
```bash
uv run pytest tests/ --cov=llm_apple --cov-report=html
```

### Run Specific Test
```bash
uv run pytest tests/test_tools.py::test_apple_model_supports_tools -v
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

- `mock_applefoundationmodels` - Mock of the applefoundationmodels module
- `mock_client_class` - Mock of the Client class
- `mock_availability` - Mock Availability enum
- `mock_prompt` - Mock llm.Prompt object
- `mock_response` - Mock llm.Response object
- `mock_conversation` - Mock llm.Conversation object

## Integration Test Requirements

To run integration tests, you need:

1. **A Mac with Apple Silicon** (M1, M2, M3, or later)
2. **macOS 15.1 or later**
3. **Apple Intelligence enabled** in System Settings
4. **Model downloaded** (happens automatically on first use)

If these requirements are not met, integration tests will be skipped with a clear message.

## Writing New Tests

### Unit Tests
Use the existing mocks in `conftest.py` to write unit tests that don't require Apple Intelligence:

```python
def test_my_feature(mock_applefoundationmodels):
    model = llm_apple.AppleModel()
    # Test your feature with mocks
```

### Integration Tests
Add new integration tests to `test_integration_tools.py` or create new integration test files. Make sure to use the skip decorator:

```python
pytestmark = pytest.mark.skipif(
    not is_apple_intelligence_available(),
    reason="Apple Intelligence not available on this system"
)

def test_my_integration():
    # Test with real API
    model = llm.get_model("apple")
    response = model.prompt("Test prompt")
    assert response.text()
```

## Continuous Integration

In CI environments where Apple Intelligence is not available, integration tests will be automatically skipped. Only unit tests will run.
