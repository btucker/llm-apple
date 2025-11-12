"""Tests for AppleModel.execute method."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
import llm_apple


def test_execute_non_streaming(mock_applefoundationmodels, mock_prompt, mock_response):
    """Test execute method without streaming."""
    model = llm_apple.AppleModel()

    result = model.execute(
        prompt=mock_prompt, stream=False, response=mock_response, conversation=None
    )

    assert result == "Generated response"


def test_execute_streaming(mock_applefoundationmodels, mock_prompt, mock_response):
    """Test execute method with streaming."""
    model = llm_apple.AppleModel()

    result = model.execute(
        prompt=mock_prompt, stream=True, response=mock_response, conversation=None
    )

    # Result should be a generator
    chunks = list(result)
    assert len(chunks) == 3
    assert chunks == ["chunk1", "chunk2", "chunk3"]


def test_execute_with_conversation(
    mock_applefoundationmodels, mock_prompt, mock_response, mock_conversation
):
    """Test execute method with conversation context."""
    model = llm_apple.AppleModel()

    result = model.execute(
        prompt=mock_prompt,
        stream=False,
        response=mock_response,
        conversation=mock_conversation,
    )

    # Should reuse session for conversation
    assert mock_conversation.id in model._sessions
    assert result == "Generated response"


def test_execute_with_custom_temperature(mock_applefoundationmodels, mock_response):
    """Test execute with custom temperature."""
    model = llm_apple.AppleModel()

    prompt = Mock()
    prompt.prompt = "Test"
    prompt.options = Mock()
    prompt.options.temperature = 0.5
    prompt.options.max_tokens = 1024
    prompt.options.instructions = None

    result = model.execute(
        prompt=prompt, stream=False, response=mock_response, conversation=None
    )

    # Verify session.generate was called with temperature
    session = model._get_session(None, None)
    session.generate.assert_called_with("Test", temperature=0.5, max_tokens=1024)


def test_execute_with_custom_max_tokens(mock_applefoundationmodels, mock_response):
    """Test execute with custom max_tokens."""
    model = llm_apple.AppleModel()

    prompt = Mock()
    prompt.prompt = "Test"
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 500
    prompt.options.instructions = None

    result = model.execute(
        prompt=prompt, stream=False, response=mock_response, conversation=None
    )

    session = model._get_session(None, None)
    session.generate.assert_called_with("Test", temperature=1.0, max_tokens=500)


def test_execute_with_instructions(mock_applefoundationmodels, mock_response):
    """Test execute with system prompt."""
    model = llm_apple.AppleModel()

    prompt = Mock()
    prompt.prompt = "Test"
    prompt.system = "You are helpful"
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024

    result = model.execute(
        prompt=prompt, stream=False, response=mock_response, conversation=None
    )

    # Verify Session was created with system prompt
    # Should have called Session() with instructions
    called_with_instructions = False
    for call in mock_applefoundationmodels.Session.call_args_list:
        if call[1].get("instructions") == "You are helpful":
            called_with_instructions = True
            break

    assert called_with_instructions


def test_execute_default_options(mock_applefoundationmodels, mock_response):
    """Test execute uses default options when not specified."""
    model = llm_apple.AppleModel()

    prompt = Mock()
    prompt.prompt = "Test"
    prompt.options = Mock()
    prompt.options.temperature = None
    prompt.options.max_tokens = None

    result = model.execute(
        prompt=prompt, stream=False, response=mock_response, conversation=None
    )

    # Should use defaults: temperature=1.0, max_tokens=1024
    session = model._get_session(None, None)
    session.generate.assert_called_with("Test", temperature=1.0, max_tokens=1024)


def test_stream_response_yields_chunks(mock_applefoundationmodels):
    """Test _stream_response yields chunks from generate(stream=True)."""
    from dataclasses import dataclass

    @dataclass
    class MockStreamChunk:
        content: str

    model = llm_apple.AppleModel()

    # Create a mock session with generate that returns chunks when stream=True
    session = Mock()

    def mock_stream_gen(*args, **kwargs):
        for chunk_text in ["a", "b", "c"]:
            yield MockStreamChunk(content=chunk_text)

    session.generate = Mock(return_value=mock_stream_gen())

    # Get streaming results
    chunks = list(
        model._stream_response(
            session=session, prompt_text="Test", temperature=1.0, max_tokens=100
        )
    )

    assert chunks == ["a", "b", "c"]
    session.generate.assert_called_once_with(
        "Test", stream=True, temperature=1.0, max_tokens=100
    )


def test_execute_multiple_conversations(
    mock_applefoundationmodels, mock_prompt, mock_response
):
    """Test execute maintains separate sessions for different conversations."""
    model = llm_apple.AppleModel()

    conv1 = Mock()
    conv1.id = "conv-1"

    conv2 = Mock()
    conv2.id = "conv-2"

    # Execute with first conversation
    model.execute(
        prompt=mock_prompt, stream=False, response=mock_response, conversation=conv1
    )

    # Execute with second conversation
    model.execute(
        prompt=mock_prompt, stream=False, response=mock_response, conversation=conv2
    )

    # Should have two separate sessions
    assert "conv-1" in model._sessions
    assert "conv-2" in model._sessions
    assert len(model._sessions) == 2
