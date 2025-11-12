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
def mock_session_class(mock_availability):
    """Mock Session class from applefoundationmodels."""
    from dataclasses import dataclass

    # Create GenerationResponse for the mock
    @dataclass
    class MockGenerationResponse:
        content: str
        is_structured: bool = False
        tool_calls: list = None
        finish_reason: str = None

        @property
        def text(self):
            if self.is_structured:
                raise ValueError("Response is structured")
            return self.content

        @property
        def parsed(self):
            if not self.is_structured:
                raise ValueError("Response is not structured")
            return self.content

    # Create StreamChunk for the mock
    @dataclass
    class MockStreamChunk:
        content: str
        finish_reason: str = None
        index: int = 0

    # Create a mock Session class
    mock = Mock()
    mock.check_availability = Mock(return_value=mock_availability.AVAILABLE)
    mock.get_availability_reason = Mock(return_value=None)

    # Create a mock session instance (what Session() returns)
    session_mock = Mock()
    # Return GenerationResponse object instead of string (0.2.0+ API)
    session_mock.generate = Mock(
        return_value=MockGenerationResponse(content="Generated response")
    )

    # For streaming, return an iterator of StreamChunk objects
    def mock_stream_gen():
        """Mock sync stream generator (0.2.0+ API)."""
        for chunk_text in ["chunk1", "chunk2", "chunk3"]:
            yield MockStreamChunk(content=chunk_text)

    # Store the generator factory
    def create_stream(*args, **kwargs):
        return mock_stream_gen()

    session_mock.generate.side_effect = lambda *args, **kwargs: (
        create_stream()
        if kwargs.get("stream")
        else MockGenerationResponse(content="Generated response")
    )

    # Session() constructor returns session_mock
    mock.return_value = session_mock

    return mock


@pytest.fixture
def mock_async_session_class(mock_availability):
    """Mock AsyncSession class from applefoundationmodels."""
    from dataclasses import dataclass

    # Create GenerationResponse for the mock
    @dataclass
    class MockGenerationResponse:
        content: str
        is_structured: bool = False
        tool_calls: list = None
        finish_reason: str = None

        @property
        def text(self):
            if self.is_structured:
                raise ValueError("Response is structured")
            return self.content

        @property
        def parsed(self):
            if not self.is_structured:
                raise ValueError("Response is not structured")
            return self.content

    # Create StreamChunk for the mock
    @dataclass
    class MockStreamChunk:
        content: str
        finish_reason: str = None
        index: int = 0

    # Create a mock AsyncSession class
    mock = Mock()
    mock.check_availability = Mock(return_value=mock_availability.AVAILABLE)
    mock.get_availability_reason = Mock(return_value=None)

    # Create a mock async session instance (what AsyncSession() returns)
    session_mock = AsyncMock()
    # Return GenerationResponse object instead of string (0.2.0+ API)
    session_mock.generate = AsyncMock(
        return_value=MockGenerationResponse(content="Generated response")
    )

    # For streaming, return an async iterator of StreamChunk objects
    async def mock_async_stream_gen():
        """Mock async stream generator (0.2.0+ API)."""
        for chunk_text in ["chunk1", "chunk2", "chunk3"]:
            yield MockStreamChunk(content=chunk_text)

    # Store the generator factory
    def create_async_stream(*args, **kwargs):
        return mock_async_stream_gen()

    # For async, we need to handle both streaming and non-streaming differently
    # We'll use the default return_value for non-streaming

    # AsyncSession() constructor returns session_mock
    mock.return_value = session_mock

    return mock


@pytest.fixture
def mock_applefoundationmodels(
    monkeypatch, mock_session_class, mock_async_session_class, mock_availability
):
    """Mock the applefoundationmodels module."""
    # Create a mock module
    mock_module = MagicMock()
    mock_module.Session = mock_session_class
    mock_module.AsyncSession = mock_async_session_class
    mock_module.Availability = mock_availability

    # Mock the types submodule for 0.2.0+ API
    mock_types = MagicMock()

    # Create real ToolCall and Function classes for testing
    from dataclasses import dataclass

    @dataclass
    class Function:
        name: str
        arguments: str

    @dataclass
    class ToolCall:
        id: str
        type: str
        function: Function

    @dataclass
    class GenerationResponse:
        content: str
        is_structured: bool
        tool_calls: list = None
        finish_reason: str = None

        @property
        def text(self):
            if self.is_structured:
                raise ValueError("Response is structured")
            return self.content

        @property
        def parsed(self):
            if not self.is_structured:
                raise ValueError("Response is not structured")
            return self.content

    mock_types.ToolCall = ToolCall
    mock_types.Function = Function
    mock_types.GenerationResponse = GenerationResponse
    mock_module.types = mock_types

    # Add to sys.modules before importing llm_apple
    sys.modules["applefoundationmodels"] = mock_module
    sys.modules["applefoundationmodels.types"] = mock_types

    yield mock_module

    # Cleanup
    if "applefoundationmodels" in sys.modules:
        del sys.modules["applefoundationmodels"]
    if "applefoundationmodels.types" in sys.modules:
        del sys.modules["applefoundationmodels.types"]


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
        generate_return: Return value for session.generate() (will be wrapped in GenerationResponse)
        transcript: Session transcript (optional)
        include_tool_support: Whether to include tool-related mocks

    Returns:
        Mock session object
    """
    from dataclasses import dataclass

    # Create GenerationResponse for the mock (0.2.0+ API)
    @dataclass
    class MockGenerationResponse:
        content: str
        is_structured: bool = False
        tool_calls: list = None
        finish_reason: str = None

        @property
        def text(self):
            if self.is_structured:
                raise ValueError("Response is structured")
            return self.content

        @property
        def parsed(self):
            if not self.is_structured:
                raise ValueError("Response is not structured")
            return self.content

    session = Mock()
    # Wrap return value in GenerationResponse for 0.2.0+ API
    session.generate = Mock(
        return_value=MockGenerationResponse(content=generate_return)
    )
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
