"""Tests for AppleAsyncModel class."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import llm_apple
import llm


@pytest.mark.asyncio
async def test_async_model_initialization():
    """Test that AppleAsyncModel initializes with correct state."""
    model = llm_apple.AppleAsyncModel()

    assert model._sessions == {}
    assert model._availability_checked is False
    assert model.model_id == "apple"
    assert model.can_stream is True
    assert model.supports_tools is True


@pytest.mark.asyncio
async def test_async_model_non_streaming(mock_applefoundationmodels):
    """Test async execute method without streaming."""
    model = llm_apple.AppleAsyncModel()

    prompt = Mock()
    prompt.prompt = "Test prompt"
    prompt.system = None
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024
    prompt.tools = []
    prompt.tool_results = []

    response = Mock()
    response.add_tool_call = Mock()

    # Create async mock for session.generate
    from dataclasses import dataclass

    @dataclass
    class MockGenerationResponse:
        content: str
        is_structured: bool = False
        tool_calls: list = None
        finish_reason: str = None

        @property
        def text(self):
            return self.content

    mock_session = AsyncMock()
    mock_session.generate = AsyncMock(
        return_value=MockGenerationResponse(content="Async response")
    )

    # Replace just the return_value, not the whole AsyncSession mock
    mock_applefoundationmodels.AsyncSession.return_value = mock_session

    # Execute and collect results
    result = []
    async for chunk in model.execute(
        prompt=prompt, stream=False, response=response, conversation=None
    ):
        result.append(chunk)

    assert len(result) == 1
    assert result[0] == "Async response"
    mock_session.generate.assert_called_once()


@pytest.mark.asyncio
async def test_async_model_streaming(mock_applefoundationmodels):
    """Test async execute method with streaming."""
    model = llm_apple.AppleAsyncModel()

    prompt = Mock()
    prompt.prompt = "Stream test"
    prompt.system = None
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024
    prompt.tools = []
    prompt.tool_results = []

    response = Mock()

    # Create async generator for streaming
    from dataclasses import dataclass

    @dataclass
    class MockStreamChunk:
        content: str
        finish_reason: str = None

    async def async_stream_gen():
        """Mock async stream generator."""
        for chunk_text in ["chunk1", "chunk2", "chunk3"]:
            yield MockStreamChunk(content=chunk_text)

    mock_session = AsyncMock()
    mock_session.generate = Mock(return_value=async_stream_gen())
    mock_applefoundationmodels.AsyncSession.return_value = mock_session

    # Execute and collect streaming results
    result = []
    async for chunk in model.execute(
        prompt=prompt, stream=True, response=response, conversation=None
    ):
        result.append(chunk)

    assert result == ["chunk1", "chunk2", "chunk3"]


@pytest.mark.asyncio
async def test_async_model_with_tools(mock_applefoundationmodels):
    """Test async execute method with tools."""
    from dataclasses import dataclass

    model = llm_apple.AppleAsyncModel()

    # Create a tool
    def get_weather(location: str) -> str:
        """Get weather for location."""
        return f"Weather in {location}: sunny"

    weather_tool = llm.Tool(
        name="get_weather",
        description="Get current weather",
        input_schema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
        implementation=get_weather,
    )

    # Create mock response with tool calls
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
    class MockGenerationResponse:
        content: str
        is_structured: bool = False
        tool_calls: list = None
        finish_reason: str = "tool_calls"

        @property
        def text(self):
            return self.content

    mock_tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function=Function(name="get_weather", arguments='{"location": "Paris"}'),
        )
    ]

    mock_session = AsyncMock()
    mock_session.generate = AsyncMock(
        return_value=MockGenerationResponse(
            content="The weather is sunny", tool_calls=mock_tool_calls
        )
    )
    mock_applefoundationmodels.AsyncSession.return_value = mock_session

    prompt = Mock()
    prompt.prompt = "What's the weather in Paris?"
    prompt.system = None
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024
    prompt.tools = [weather_tool]
    prompt.tool_results = []

    response = Mock()
    response.add_tool_call = Mock()

    # Execute
    result = []
    async for chunk in model.execute(
        prompt=prompt, stream=False, response=response, conversation=None
    ):
        result.append(chunk)

    # Verify tool call was added to response
    assert response.add_tool_call.called
    call_args = response.add_tool_call.call_args[0][0]
    assert call_args.name == "get_weather"
    assert call_args.arguments == {"location": "Paris"}
    assert call_args.tool_call_id == "call_1"

    # Verify result was returned
    assert result[0] == "The weather is sunny"


@pytest.mark.asyncio
async def test_async_model_with_tool_results(mock_applefoundationmodels):
    """Test async execute method with tool results."""
    from dataclasses import dataclass

    model = llm_apple.AppleAsyncModel()

    @dataclass
    class MockGenerationResponse:
        content: str
        is_structured: bool = False
        tool_calls: list = None
        finish_reason: str = "stop"

        @property
        def text(self):
            return self.content

    mock_session = AsyncMock()
    mock_session.generate = AsyncMock(
        return_value=MockGenerationResponse(
            content="Based on the weather, I recommend a picnic!"
        )
    )
    mock_applefoundationmodels.AsyncSession.return_value = mock_session

    # Create prompt with tool results
    prompt = Mock()
    prompt.prompt = "What should I do?"
    prompt.system = None
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024
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

    # Execute
    result = []
    async for chunk in model.execute(
        prompt=prompt, stream=False, response=response, conversation=None
    ):
        result.append(chunk)

    # Verify tool results were included in the prompt
    assert mock_session.generate.called
    call_args = mock_session.generate.call_args[0]
    assert "get_weather() returned: Weather in Paris: sunny, 72°F" in call_args[0]
    assert "What should I do?" in call_args[0]

    # Verify result
    assert result[0] == "Based on the weather, I recommend a picnic!"


@pytest.mark.asyncio
async def test_async_model_with_conversation(mock_applefoundationmodels):
    """Test async execute method reuses session for same conversation."""
    from dataclasses import dataclass

    model = llm_apple.AppleAsyncModel()

    @dataclass
    class MockGenerationResponse:
        content: str
        is_structured: bool = False
        tool_calls: list = None
        finish_reason: str = "stop"

        @property
        def text(self):
            return self.content

    mock_session = AsyncMock()
    mock_session.generate = AsyncMock(
        return_value=MockGenerationResponse(content="Response")
    )
    mock_applefoundationmodels.AsyncSession.return_value = mock_session

    conversation = Mock()
    conversation.id = "test-conv-async"

    prompt = Mock()
    prompt.prompt = "First message"
    prompt.system = None
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024
    prompt.tools = []
    prompt.tool_results = []

    response = Mock()

    # Execute first message
    result1 = []
    async for chunk in model.execute(
        prompt=prompt, stream=False, response=response, conversation=conversation
    ):
        result1.append(chunk)

    # AsyncSession should be created once
    first_call_count = mock_applefoundationmodels.AsyncSession.call_count

    # Execute second message with same conversation
    prompt.prompt = "Second message"
    result2 = []
    async for chunk in model.execute(
        prompt=prompt, stream=False, response=response, conversation=conversation
    ):
        result2.append(chunk)

    # AsyncSession should not be created again (reused)
    assert mock_applefoundationmodels.AsyncSession.call_count == first_call_count

    # Verify conversation was cached
    assert conversation.id in model._sessions


@pytest.mark.asyncio
async def test_async_model_without_prompt_text_but_with_tool_results(
    mock_applefoundationmodels,
):
    """Test async execute without prompt text but with tool results."""
    from dataclasses import dataclass

    model = llm_apple.AppleAsyncModel()

    @dataclass
    class MockGenerationResponse:
        content: str
        is_structured: bool = False
        tool_calls: list = None
        finish_reason: str = "stop"

        @property
        def text(self):
            return self.content

    mock_session = AsyncMock()
    mock_session.generate = AsyncMock(
        return_value=MockGenerationResponse(content="Continuation response")
    )
    mock_applefoundationmodels.AsyncSession.return_value = mock_session

    # Create prompt without prompt text but with tool results
    prompt = Mock()
    prompt.prompt = None  # No prompt text
    prompt.system = None
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024
    prompt.tools = []
    prompt.tool_results = [
        llm.ToolResult(
            name="get_weather",
            output="Weather in Paris: sunny, 72°F",
            tool_call_id="call_1",
        )
    ]

    response = Mock()

    # Execute
    result = []
    async for chunk in model.execute(
        prompt=prompt, stream=False, response=response, conversation=None
    ):
        result.append(chunk)

    # Verify tool results were formatted with continuation prompt
    assert mock_session.generate.called
    call_args = mock_session.generate.call_args[0]
    assert "get_weather() returned: Weather in Paris: sunny, 72°F" in call_args[0]
    assert "Please continue based on these results" in call_args[0]

    # Verify result
    assert result[0] == "Continuation response"


@pytest.mark.asyncio
async def test_async_model_create_session_with_tools(mock_applefoundationmodels):
    """Test that async create_session passes tools correctly."""
    model = llm_apple.AppleAsyncModel()

    def weather_func(location: str) -> str:
        """Get weather."""
        return f"Weather in {location}"

    weather_tool = llm.Tool(
        name="get_weather",
        description="Get weather",
        input_schema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
        implementation=weather_func,
    )

    # Create session with tools
    session = model._create_session(instructions="Test", tools=[weather_tool])

    # Verify AsyncSession was called with tools
    assert mock_applefoundationmodels.AsyncSession.called
    call_kwargs = mock_applefoundationmodels.AsyncSession.call_args[1]
    assert "tools" in call_kwargs
    assert len(call_kwargs["tools"]) == 1
    assert call_kwargs["tools"][0] == weather_func


@pytest.mark.asyncio
async def test_async_model_parameters_passed_through(mock_applefoundationmodels):
    """Test that temperature and max_tokens are passed through correctly."""
    from dataclasses import dataclass

    model = llm_apple.AppleAsyncModel()

    @dataclass
    class MockGenerationResponse:
        content: str
        is_structured: bool = False
        tool_calls: list = None
        finish_reason: str = "stop"

        @property
        def text(self):
            return self.content

    mock_session = AsyncMock()
    mock_session.generate = AsyncMock(
        return_value=MockGenerationResponse(content="Response")
    )
    mock_applefoundationmodels.AsyncSession.return_value = mock_session

    prompt = Mock()
    prompt.prompt = "Test"
    prompt.system = "You are helpful"
    prompt.options = Mock()
    prompt.options.temperature = 0.5
    prompt.options.max_tokens = 2048
    prompt.tools = []
    prompt.tool_results = []

    response = Mock()

    # Execute
    result = []
    async for chunk in model.execute(
        prompt=prompt, stream=False, response=response, conversation=None
    ):
        result.append(chunk)

    # Verify AsyncSession was called with system prompt
    mock_applefoundationmodels.AsyncSession.assert_called_with(
        instructions="You are helpful"
    )

    # Verify session.generate was called with correct parameters
    mock_session.generate.assert_called_once_with(
        "Test", temperature=0.5, max_tokens=2048
    )
