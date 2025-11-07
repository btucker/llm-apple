"""Tests for tool calling functionality in llm-apple."""
import pytest
from unittest.mock import Mock, MagicMock
import llm
import llm_apple


@pytest.fixture
def weather_tool(tool_factory):
    """Create a sample weather tool for testing."""
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"Weather in {location}: sunny, 72°F"

    return tool_factory(
        name="get_weather",
        description="Get weather for a location",
        properties={"location": {"type": "string"}},
        required=["location"],
        implementation=get_weather
    )


@pytest.fixture
def session_with_tool_transcript(session_factory):
    """Create a mock session with tool call transcript."""
    transcript = [
        {'type': 'prompt', 'content': 'What is the weather in Paris?'},
        {
            'type': 'tool_calls',
            'tool_calls': [
                {
                    'name': 'get_weather',
                    'id': 'call_123',
                    'arguments': '{"location": "Paris"}'
                }
            ]
        },
        {'type': 'tool_output', 'content': 'Weather in Paris: sunny, 72°F'},
        {'type': 'response', 'content': 'The weather is sunny'}
    ]

    return session_factory(
        generate_return="The weather is sunny",
        transcript=transcript,
        include_tool_support=True
    )


def test_apple_model_supports_tools():
    """Test that AppleModel declares tool support."""
    model = llm_apple.AppleModel()
    assert model.supports_tools is True


def test_register_tools_with_session(mock_applefoundationmodels, weather_tool, session_factory):
    """Test that tools are properly registered with a session."""
    model = llm_apple.AppleModel()
    session = session_factory(include_tool_support=True)

    model._register_tools_with_session(session, [weather_tool])

    # Verify tool was added to session
    assert 'get_weather' in session._tools
    assert session._register_tools.called

    # Verify tool metadata
    registered_tool = session._tools['get_weather']
    assert registered_tool._tool_name == 'get_weather'
    assert registered_tool._tool_description == 'Get weather for a location'
    assert registered_tool._tool_parameters == weather_tool.input_schema


def test_register_tools_with_empty_list(mock_applefoundationmodels, session_factory):
    """Test that registering empty tool list is a no-op."""
    model = llm_apple.AppleModel()
    session = session_factory(include_tool_support=True)

    model._register_tools_with_session(session, [])

    # Should not modify session
    assert len(session._tools) == 0


def test_extract_tool_calls_from_transcript(mock_applefoundationmodels):
    """Test extracting tool calls from session transcript."""
    model = llm_apple.AppleModel()

    transcript = [
        {'type': 'prompt', 'content': 'Test'},
        {
            'type': 'tool_calls',
            'tool_calls': [
                {
                    'name': 'get_weather',
                    'id': 'call_1',
                    'arguments': '{"location": "Paris"}'
                },
                {
                    'name': 'get_time',
                    'id': 'call_2',
                    'arguments': '{}'
                }
            ]
        }
    ]

    tool_calls = model._extract_tool_calls_from_transcript(transcript)

    assert len(tool_calls) == 2
    assert tool_calls[0].name == 'get_weather'
    assert tool_calls[0].arguments == {'location': 'Paris'}
    assert tool_calls[0].tool_call_id == 'call_1'
    assert tool_calls[1].name == 'get_time'
    assert tool_calls[1].arguments == {}
    assert tool_calls[1].tool_call_id == 'call_2'


def test_extract_tool_calls_with_no_tool_calls(mock_applefoundationmodels):
    """Test extracting tool calls from transcript with no tool calls."""
    model = llm_apple.AppleModel()

    transcript = [
        {'type': 'prompt', 'content': 'Test'},
        {'type': 'response', 'content': 'Response'}
    ]

    tool_calls = model._extract_tool_calls_from_transcript(transcript)

    assert len(tool_calls) == 0


def test_add_tool_results_to_session(mock_applefoundationmodels, session_factory):
    """Test adding tool results to session."""
    model = llm_apple.AppleModel()
    session = session_factory()

    tool_results = [
        llm.ToolResult(
            name='get_weather',
            output='Weather in Paris: sunny, 72°F',
            tool_call_id='call_1'
        ),
        llm.ToolResult(
            name='get_time',
            output='2:30 PM',
            tool_call_id='call_2'
        )
    ]

    model._add_tool_results_to_session(session, tool_results)

    # Verify add_message was called for each result
    assert session.add_message.call_count == 2
    session.add_message.assert_any_call("user", "Tool get_weather returned: Weather in Paris: sunny, 72°F")
    session.add_message.assert_any_call("user", "Tool get_time returned: 2:30 PM")


def test_add_tool_results_with_empty_list(mock_applefoundationmodels, session_factory):
    """Test adding empty tool results list is a no-op."""
    model = llm_apple.AppleModel()
    session = session_factory()

    model._add_tool_results_to_session(session, [])

    assert session.add_message.call_count == 0


def test_execute_with_tools(mock_applefoundationmodels, weather_tool, session_with_tool_transcript):
    """Test execute method with tools."""
    model = llm_apple.AppleModel()

    # Mock the session creation to return our mock session
    model._sessions = {}
    client = model._get_client()
    client.create_session = Mock(return_value=session_with_tool_transcript)

    # Create prompt with tools
    prompt = Mock()
    prompt.prompt = "What is the weather in Paris?"
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024
    prompt.system = None
    prompt.tools = [weather_tool]
    prompt.tool_results = []

    response = Mock()
    response.add_tool_call = Mock()

    result = model.execute(prompt, stream=False, response=response, conversation=None)

    # Verify tool was registered
    assert 'get_weather' in session_with_tool_transcript._tools

    # Verify tool calls were added to response
    assert response.add_tool_call.called

    # Verify result was returned
    assert result == "The weather is sunny"


def test_execute_with_tool_results(mock_applefoundationmodels, session_factory):
    """Test execute method with tool results."""
    model = llm_apple.AppleModel()

    session = session_factory(
        generate_return="Based on the weather, I recommend...",
        transcript=[],
        include_tool_support=True
    )

    client = model._get_client()
    client.create_session = Mock(return_value=session)

    # Create prompt with tool results
    prompt = Mock()
    prompt.prompt = "Based on the weather, what should I do?"
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024
    prompt.system = None
    prompt.tools = []
    prompt.tool_results = [
        llm.ToolResult(
            name='get_weather',
            output='Weather in Paris: sunny, 72°F',
            tool_call_id='call_1'
        )
    ]

    response = Mock()
    response.add_tool_call = Mock()

    result = model.execute(prompt, stream=False, response=response, conversation=None)

    # Verify tool result was added to session
    assert session.add_message.called
    session.add_message.assert_called_with("user", "Tool get_weather returned: Weather in Paris: sunny, 72°F")

    # Verify result was returned
    assert result == "Based on the weather, I recommend..."


def test_execute_without_prompt_text_but_with_tool_results(mock_applefoundationmodels, session_factory):
    """Test execute method when prompt.prompt is None but tool_results are present."""
    model = llm_apple.AppleModel()

    session = session_factory(
        generate_return="Continuation response",
        transcript=[],
        include_tool_support=True
    )

    client = model._get_client()
    client.create_session = Mock(return_value=session)

    # Create prompt without prompt text but with tool results
    prompt = Mock()
    prompt.prompt = None  # No prompt text
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024
    prompt.system = None
    prompt.tools = []
    prompt.tool_results = [
        llm.ToolResult(
            name='get_weather',
            output='Weather in Paris: sunny, 72°F',
            tool_call_id='call_1'
        )
    ]

    response = Mock()
    response.add_tool_call = Mock()

    result = model.execute(prompt, stream=False, response=response, conversation=None)

    # Verify a continuation prompt was created
    session.generate.assert_called_once()
    call_args = session.generate.call_args
    assert "Please continue based on the tool results above" in call_args[0][0]

    # Verify result was returned
    assert result == "Continuation response"


def test_tool_wrapper_closure(mock_applefoundationmodels, tool_factory, session_factory):
    """Test that tool wrappers have proper closure and don't share state."""
    model = llm_apple.AppleModel()

    # Create multiple tools
    def tool1_impl(x: str) -> str:
        return f"tool1: {x}"

    def tool2_impl(y: str) -> str:
        return f"tool2: {y}"

    tools = [
        tool_factory(
            name="tool1",
            description="First tool",
            properties={"x": {"type": "string"}},
            implementation=tool1_impl
        ),
        tool_factory(
            name="tool2",
            description="Second tool",
            properties={"y": {"type": "string"}},
            implementation=tool2_impl
        )
    ]

    session = session_factory(include_tool_support=True)

    model._register_tools_with_session(session, tools)

    # Verify both tools were registered
    assert 'tool1' in session._tools
    assert 'tool2' in session._tools

    # Verify they call the correct implementations
    result1 = session._tools['tool1'](x="test1")
    result2 = session._tools['tool2'](y="test2")

    assert result1 == "tool1: test1"
    assert result2 == "tool2: test2"
