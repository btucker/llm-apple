"""Tests for tool calling functionality in llm-apple."""

import pytest
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
        implementation=get_weather,
    )


@pytest.fixture
def session_with_tool_transcript(session_factory):
    """Create a mock session with tool call transcript."""
    transcript = [
        {"type": "prompt", "content": "What is the weather in Paris?"},
        {
            "type": "tool_calls",
            "tool_calls": [
                {
                    "name": "get_weather",
                    "id": "call_123",
                    "arguments": '{"location": "Paris"}',
                }
            ],
        },
        {"type": "tool_output", "content": "Weather in Paris: sunny, 72°F"},
        {"type": "response", "content": "The weather is sunny"},
    ]

    return session_factory(
        generate_return="The weather is sunny",
        transcript=transcript,
        include_tool_support=True,
    )


def test_apple_model_supports_tools():
    """Test that AppleModel declares tool support."""
    model = llm_apple.AppleModel()
    assert model.supports_tools is True


def test_create_session_with_tools(mock_applefoundationmodels, weather_tool):
    """Test that tools are properly passed to session creation."""
    model = llm_apple.AppleModel()

    # Create session with tools (in 0.2.0, tools are passed at creation)
    session = model._create_session(instructions="Test", tools=[weather_tool])

    # Verify session is the mocked Session instance
    assert session is mock_applefoundationmodels.Session.return_value

    # Verify Session() was called with tools
    assert mock_applefoundationmodels.Session.called
    call_kwargs = mock_applefoundationmodels.Session.call_args[1]
    assert "tools" in call_kwargs
    assert len(call_kwargs["tools"]) == 1
    # The tool function should be the implementation
    assert call_kwargs["tools"][0] == weather_tool.implementation


def test_create_session_without_tools(mock_applefoundationmodels):
    """Test that session creation works without tools."""
    model = llm_apple.AppleModel()

    # Create session without tools
    session = model._create_session(instructions="Test", tools=None)

    # Verify session is the mocked Session instance
    assert session is mock_applefoundationmodels.Session.return_value

    # Verify Session() was called without tools
    assert mock_applefoundationmodels.Session.called
    call_kwargs = mock_applefoundationmodels.Session.call_args[1]
    assert "tools" not in call_kwargs or call_kwargs.get("tools") is None


def test_extract_tool_calls_from_response(mock_applefoundationmodels):
    """Test extracting tool calls from GenerationResponse."""
    from unittest.mock import Mock
    from applefoundationmodels.types import ToolCall, Function

    model = llm_apple.AppleModel()

    # Create a mock response with tool calls (0.2.0+ API)
    response = Mock()
    response.tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function=Function(name="get_weather", arguments='{"location": "Paris"}'),
        ),
        ToolCall(
            id="call_2",
            type="function",
            function=Function(name="get_time", arguments="{}"),
        ),
    ]

    tool_calls = model._extract_tool_calls_from_response(response)

    assert len(tool_calls) == 2
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == {"location": "Paris"}
    assert tool_calls[0].tool_call_id == "call_1"
    assert tool_calls[1].name == "get_time"
    assert tool_calls[1].arguments == {}
    assert tool_calls[1].tool_call_id == "call_2"


def test_extract_tool_calls_with_no_tool_calls(mock_applefoundationmodels):
    """Test extracting tool calls from response with no tool calls."""
    from unittest.mock import Mock

    model = llm_apple.AppleModel()

    # Create a mock response without tool calls
    response = Mock()
    response.tool_calls = None

    tool_calls = model._extract_tool_calls_from_response(response)

    assert len(tool_calls) == 0


def test_format_tool_results_as_prompt(mock_applefoundationmodels):
    """Test formatting tool results as prompt text."""
    model = llm_apple.AppleModel()

    tool_results = [
        llm.ToolResult(
            name="get_weather",
            output="Weather in Paris: sunny, 72°F",
            tool_call_id="call_1",
        ),
        llm.ToolResult(name="get_time", output="2:30 PM", tool_call_id="call_2"),
    ]

    result = model._format_tool_results_as_prompt(tool_results)

    # Verify the formatted string contains both results
    assert "get_weather() returned: Weather in Paris: sunny, 72°F" in result
    assert "get_time() returned: 2:30 PM" in result


def test_format_tool_results_with_empty_list(mock_applefoundationmodels):
    """Test formatting empty tool results list returns empty string."""
    model = llm_apple.AppleModel()

    result = model._format_tool_results_as_prompt([])

    assert result == ""


def test_execute_with_tools(mock_applefoundationmodels, weather_tool):
    """Test execute method with tools."""
    from unittest.mock import Mock
    from applefoundationmodels.types import GenerationResponse, ToolCall, Function

    model = llm_apple.AppleModel()

    # Create a mock session with generate method that returns a response with tool calls
    mock_session = Mock()
    mock_response = GenerationResponse(
        content="The weather is sunny", is_structured=False
    )
    mock_response.tool_calls = [
        ToolCall(
            id="call_123",
            type="function",
            function=Function(name="get_weather", arguments='{"location": "Paris"}'),
        )
    ]
    mock_session.generate.return_value = mock_response

    # Mock the session creation
    model._sessions = {}
    mock_applefoundationmodels.Session.return_value = mock_session

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

    # Verify tool calls were added to response
    assert response.add_tool_call.called

    # Verify result was returned
    assert result == "The weather is sunny"


def test_execute_with_tool_results(mock_applefoundationmodels):
    """Test execute method with tool results."""
    from unittest.mock import Mock
    from applefoundationmodels.types import GenerationResponse

    model = llm_apple.AppleModel()

    # Create a mock session with generate method
    mock_session = Mock()
    mock_response = GenerationResponse(
        content="Based on the weather, I recommend...", is_structured=False
    )
    mock_session.generate.return_value = mock_response

    mock_applefoundationmodels.Session.return_value = mock_session

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
            name="get_weather",
            output="Weather in Paris: sunny, 72°F",
            tool_call_id="call_1",
        )
    ]

    response = Mock()
    response.add_tool_call = Mock()

    result = model.execute(prompt, stream=False, response=response, conversation=None)

    # Verify generate was called with tool results in the prompt
    assert mock_session.generate.called
    call_args = mock_session.generate.call_args[0]
    prompt_text = call_args[0]
    # Tool results should be formatted into the prompt
    assert "get_weather() returned: Weather in Paris: sunny, 72°F" in prompt_text

    # Verify result was returned
    assert result == "Based on the weather, I recommend..."


def test_execute_without_prompt_text_but_with_tool_results(mock_applefoundationmodels):
    """Test execute method when prompt.prompt is None but tool_results are present."""
    from unittest.mock import Mock
    from applefoundationmodels.types import GenerationResponse

    model = llm_apple.AppleModel()

    # Create a mock session with generate method
    mock_session = Mock()
    mock_response = GenerationResponse(
        content="Continuation response", is_structured=False
    )
    mock_session.generate.return_value = mock_response

    mock_applefoundationmodels.Session.return_value = mock_session

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
            name="get_weather",
            output="Weather in Paris: sunny, 72°F",
            tool_call_id="call_1",
        )
    ]

    response = Mock()
    response.add_tool_call = Mock()

    result = model.execute(prompt, stream=False, response=response, conversation=None)

    # Verify a continuation prompt was created
    mock_session.generate.assert_called_once()
    call_args = mock_session.generate.call_args[0]
    prompt_text = call_args[0]
    assert "get_weather() returned: Weather in Paris: sunny, 72°F" in prompt_text
    assert "Please continue based on these results" in prompt_text

    # Verify result was returned
    assert result == "Continuation response"


def test_create_session_passes_tool_implementations(
    mock_applefoundationmodels, tool_factory
):
    """Test that tool implementations are passed correctly to create_session."""
    from unittest.mock import Mock

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
            implementation=tool1_impl,
        ),
        tool_factory(
            name="tool2",
            description="Second tool",
            properties={"y": {"type": "string"}},
            implementation=tool2_impl,
        ),
    ]

    # Create session with tools
    model._create_session(instructions="Test", tools=tools)

    # Verify Session() was called with both tool implementations
    assert mock_applefoundationmodels.Session.called
    call_kwargs = mock_applefoundationmodels.Session.call_args[1]
    assert "tools" in call_kwargs
    assert len(call_kwargs["tools"]) == 2
    assert call_kwargs["tools"][0] == tool1_impl
    assert call_kwargs["tools"][1] == tool2_impl
