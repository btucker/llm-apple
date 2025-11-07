"""Pytest configuration and fixtures for llm-apple tests."""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
import sys
import llm


@pytest.fixture
def mock_availability():
    """Mock Availability enum."""
    mock = Mock()
    mock.AVAILABLE = 1
    mock.UNAVAILABLE = 0
    return mock


@pytest.fixture
def mock_client_class(mock_availability):
    """Mock Client class from applefoundationmodels."""
    mock = Mock()
    mock.check_availability = Mock(return_value=mock_availability.AVAILABLE)
    mock.get_availability_reason = Mock(return_value=None)

    # Create a mock client instance
    client_instance = Mock()

    # Create a mock session
    session_mock = Mock()
    session_mock.generate = Mock(return_value="Generated response")

    # Create async mock for streaming
    async def mock_stream():
        """Mock async stream generator."""
        for chunk in ["chunk1", "chunk2", "chunk3"]:
            yield chunk

    session_mock.generate_stream = Mock(return_value=mock_stream())

    client_instance.create_session = Mock(return_value=session_mock)
    mock.return_value = client_instance

    return mock


@pytest.fixture
def mock_applefoundationmodels(monkeypatch, mock_client_class, mock_availability):
    """Mock the applefoundationmodels module."""
    # Create a mock module
    mock_module = MagicMock()
    mock_module.Client = mock_client_class
    mock_module.Availability = mock_availability

    # Add to sys.modules before importing llm_apple
    sys.modules["applefoundationmodels"] = mock_module

    yield mock_module

    # Cleanup
    if "applefoundationmodels" in sys.modules:
        del sys.modules["applefoundationmodels"]


@pytest.fixture
def mock_prompt():
    """Mock llm.Prompt object."""
    prompt = Mock()
    prompt.prompt = "Test prompt"
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024
    prompt.options.instructions = None
    return prompt


@pytest.fixture
def mock_response():
    """Mock llm.Response object."""
    return Mock()


@pytest.fixture
def mock_conversation():
    """Mock llm.Conversation object."""
    conversation = Mock()
    conversation.id = "test-conversation-id"
    return conversation


# ============================================================================
# DRY Test Helpers - Phase 1
# ============================================================================


def create_tool(name, description, properties=None, required=None, implementation=None):
    """
    Factory function to create llm.Tool objects with less boilerplate.

    Args:
        name: Tool name
        description: Tool description
        properties: Dict of property definitions (default: {})
        required: List of required properties (default: [])
        implementation: Tool implementation function

    Returns:
        llm.Tool object
    """
    return llm.Tool(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties or {},
            "required": required or [],
        },
        implementation=implementation,
    )


@pytest.fixture
def tool_factory():
    """Fixture that provides tool creation factory."""
    return create_tool


class CallTracker:
    """Helper class to track tool calls with parameters."""

    def __init__(self):
        self.calls = []

    def track(self, **kwargs):
        """Track a call with the given parameters."""
        self.calls.append(kwargs)

    def was_called(self):
        """Check if any calls were tracked."""
        return len(self.calls) > 0

    def call_count(self):
        """Return number of calls tracked."""
        return len(self.calls)

    def get_call(self, index=0):
        """Get a specific call by index (default: first call)."""
        if index < len(self.calls):
            return self.calls[index]
        return None

    def assert_called_with(self, **expected_kwargs):
        """Assert first call was made with expected parameters."""
        assert self.was_called(), "Tool was not called"
        actual = self.get_call(0)
        for key, expected_value in expected_kwargs.items():
            assert key in actual, f"Parameter '{key}' not found in call"
            actual_value = actual[key]
            # Case-insensitive comparison for strings
            if isinstance(expected_value, str) and isinstance(actual_value, str):
                assert (
                    actual_value.lower() == expected_value.lower()
                ), f"Expected {key}={expected_value}, got {actual_value}"
            else:
                assert (
                    actual_value == expected_value
                ), f"Expected {key}={expected_value}, got {actual_value}"


@pytest.fixture
def call_tracker():
    """Fixture providing a call tracker for tool testing."""
    return CallTracker()


def create_mock_session(
    generate_return="Generated response", transcript=None, include_tool_support=True
):
    """
    Factory function to create mock sessions with configurable behavior.

    Args:
        generate_return: Return value for session.generate()
        transcript: Session transcript (optional)
        include_tool_support: Whether to include tool-related mocks

    Returns:
        Mock session object
    """
    session = Mock()
    session.generate = Mock(return_value=generate_return)
    session.add_message = Mock()

    if include_tool_support:
        session._tools = {}
        session._register_tools = Mock()

    if transcript is not None:
        session.transcript = transcript
    else:
        session.transcript = []

    return session


@pytest.fixture
def session_factory():
    """Fixture providing session creation factory."""
    return create_mock_session


def assert_response_contains(response, *expected_strings, case_sensitive=True):
    """
    Assert that response text contains all expected strings.

    Args:
        response: LLM response object
        *expected_strings: Strings that should appear in response
        case_sensitive: Whether to do case-sensitive matching (default: True)

    Returns:
        str: The response text
    """
    assert response.text(), "Response has no text"
    response_text = response.text()

    for expected in expected_strings:
        search_text = response_text if case_sensitive else response_text.lower()
        search_expected = expected if case_sensitive else expected.lower()
        assert (
            search_expected in search_text
        ), f"Expected '{expected}' in response: {response_text}"

    return response_text


@pytest.fixture
def assert_response():
    """Fixture providing response assertion helper."""
    return assert_response_contains
