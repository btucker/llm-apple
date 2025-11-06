"""Integration tests to verify parameters are correctly passed through."""
import pytest
from unittest.mock import Mock, call
import llm_apple


def test_all_parameters_passed_through_non_streaming(mock_applefoundationmodels):
    """Verify all parameters flow correctly from prompt to session.generate()."""
    model = llm_apple.AppleModel()

    # Create prompt with all custom options
    prompt = Mock()
    prompt.prompt = "Test prompt text"
    prompt.system = "You are a helpful assistant"
    prompt.options = Mock()
    prompt.options.temperature = 0.7
    prompt.options.max_tokens = 500

    response = Mock()
    conversation = None

    # Execute
    model.execute(
        prompt=prompt,
        stream=False,
        response=response,
        conversation=conversation
    )

    # Get the session that was created
    client = model._get_client()

    # Verify create_session was called with system prompt as instructions
    client.create_session.assert_called_with(instructions="You are a helpful assistant")

    # Get the session instance
    session = client.create_session.return_value

    # Verify session.generate was called with correct parameters
    session.generate.assert_called_once_with(
        "Test prompt text",
        temperature=0.7,
        max_tokens=500
    )


def test_all_parameters_passed_through_streaming(mock_applefoundationmodels):
    """Verify all parameters flow correctly from prompt to session.generate_stream()."""
    model = llm_apple.AppleModel()

    # Create prompt with all custom options
    prompt = Mock()
    prompt.prompt = "Streaming test"
    prompt.system = "You are creative"
    prompt.options = Mock()
    prompt.options.temperature = 1.5
    prompt.options.max_tokens = 2048

    response = Mock()
    conversation = None

    # Execute streaming
    result = model.execute(
        prompt=prompt,
        stream=True,
        response=response,
        conversation=conversation
    )

    # Consume the generator
    chunks = list(result)

    # Get the session that was created
    client = model._get_client()

    # Verify create_session was called with system prompt as instructions
    client.create_session.assert_called_with(instructions="You are creative")

    # Get the session instance
    session = client.create_session.return_value

    # Verify session.generate_stream was called with correct parameters
    session.generate_stream.assert_called_once_with(
        "Streaming test",
        temperature=1.5,
        max_tokens=2048
    )


def test_default_values_when_options_none(mock_applefoundationmodels):
    """Verify default values are used when prompt.options values are None."""
    model = llm_apple.AppleModel()

    # Create prompt with None values
    prompt = Mock()
    prompt.prompt = "Test"
    prompt.system = None
    prompt.options = Mock()
    prompt.options.temperature = None
    prompt.options.max_tokens = None

    response = Mock()

    # Execute
    model.execute(
        prompt=prompt,
        stream=False,
        response=response,
        conversation=None
    )

    # Get session
    client = model._get_client()
    session = client.create_session.return_value

    # Verify defaults were used (1.0 for temperature, 1024 for max_tokens)
    session.generate.assert_called_once_with(
        "Test",
        temperature=1.0,
        max_tokens=1024
    )


def test_edge_case_temperatures(mock_applefoundationmodels):
    """Test edge case temperature values are passed through correctly."""
    model = llm_apple.AppleModel()

    # Test minimum temperature (0.0)
    prompt = Mock()
    prompt.prompt = "Test"
    prompt.options = llm_apple.AppleModel.Options(temperature=0.0, max_tokens=100)

    model.execute(prompt=prompt, stream=False, response=Mock(), conversation=None)

    client = model._get_client()
    session = client.create_session.return_value

    # Should pass through 0.0, not the default
    calls = [c for c in session.generate.call_args_list]
    assert any(c == call("Test", temperature=0.0, max_tokens=100) for c in calls)

    # Test maximum temperature (2.0)
    model2 = llm_apple.AppleModel()
    prompt2 = Mock()
    prompt2.prompt = "Test2"
    prompt2.options = llm_apple.AppleModel.Options(temperature=2.0, max_tokens=100)

    model2.execute(prompt=prompt2, stream=False, response=Mock(), conversation=None)

    client2 = model2._get_client()
    session2 = client2.create_session.return_value

    session2.generate.assert_called_with("Test2", temperature=2.0, max_tokens=100)


def test_conversation_session_reuse_preserves_parameters(mock_applefoundationmodels):
    """Verify that conversation sessions still receive correct parameters on each call."""
    model = llm_apple.AppleModel()

    conversation = Mock()
    conversation.id = "conv-123"

    # First call with one set of parameters
    prompt1 = Mock()
    prompt1.prompt = "First"
    prompt1.system = "Be concise"
    prompt1.options = Mock()
    prompt1.options.temperature = 0.5
    prompt1.options.max_tokens = 100

    model.execute(prompt=prompt1, stream=False, response=Mock(), conversation=conversation)

    # Second call to same conversation with different parameters
    prompt2 = Mock()
    prompt2.prompt = "Second"
    prompt2.system = "Be creative"
    prompt2.options = Mock()
    prompt2.options.temperature = 1.2
    prompt2.options.max_tokens = 500

    model.execute(prompt=prompt2, stream=False, response=Mock(), conversation=conversation)

    # Get the shared session
    session = model._sessions["conv-123"]

    # Verify both calls used their respective parameters
    assert session.generate.call_count == 2

    call1, call2 = session.generate.call_args_list

    # First call
    assert call1[0][0] == "First"
    assert call1[1]['temperature'] == 0.5
    assert call1[1]['max_tokens'] == 100

    # Second call
    assert call2[0][0] == "Second"
    assert call2[1]['temperature'] == 1.2
    assert call2[1]['max_tokens'] == 500
