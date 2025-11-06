"""Pytest configuration and fixtures for llm-apple tests."""
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
import sys


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
    sys.modules['applefoundationmodels'] = mock_module

    yield mock_module

    # Cleanup
    if 'applefoundationmodels' in sys.modules:
        del sys.modules['applefoundationmodels']


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
