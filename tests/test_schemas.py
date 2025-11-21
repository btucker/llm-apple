"""Tests for schema (structured output) support."""

import pytest
from unittest.mock import Mock
import json
import llm_apple


def test_apple_model_supports_schemas():
    """Test that AppleModel declares schema support."""
    model = llm_apple.AppleModel()
    assert hasattr(model, "supports_schema")
    assert model.supports_schema is True


def test_async_model_supports_schemas():
    """Test that AppleAsyncModel declares schema support."""
    model = llm_apple.AppleAsyncModel()
    assert hasattr(model, "supports_schema")
    assert model.supports_schema is True


def test_execute_with_schema(mock_applefoundationmodels, mock_response):
    """Test execute method with schema returns parsed JSON."""
    from dataclasses import dataclass

    @dataclass
    class MockGenerationResponse:
        content: str = "Generated response"
        is_structured: bool = True
        tool_calls: list = None

        @property
        def text(self):
            if self.is_structured:
                raise ValueError("Response is structured")
            return self.content

        @property
        def parsed(self):
            if not self.is_structured:
                raise ValueError("Response is not structured")
            return {"name": "Alice", "age": 28, "city": "Paris"}

    # Update the session mock to return structured response
    session = Mock()
    session.generate = Mock(return_value=MockGenerationResponse())
    mock_applefoundationmodels.Session.return_value = session

    model = llm_apple.AppleModel()

    # Create prompt with schema
    prompt = Mock()
    prompt.prompt = "Extract person info: Alice is 28 and lives in Paris"
    prompt.schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
        },
        "required": ["name", "age", "city"],
    }
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024

    result = model.execute(
        prompt=prompt, stream=False, response=mock_response, conversation=None
    )

    # Verify schema was passed to generate
    session.generate.assert_called_once()
    call_kwargs = session.generate.call_args[1]
    assert call_kwargs["schema"] == prompt.schema

    # Result should be JSON-serialized parsed response
    assert result == json.dumps({"name": "Alice", "age": 28, "city": "Paris"})


def test_execute_without_schema_returns_text(mock_applefoundationmodels, mock_response):
    """Test execute method without schema returns text as usual."""
    from dataclasses import dataclass

    @dataclass
    class MockGenerationResponse:
        content: str = "Generated response"
        is_structured: bool = False
        tool_calls: list = None

        @property
        def text(self):
            return self.content

        @property
        def parsed(self):
            if not self.is_structured:
                raise ValueError("Response is not structured")
            return None

    session = Mock()
    session.generate = Mock(return_value=MockGenerationResponse())
    mock_applefoundationmodels.Session.return_value = session

    model = llm_apple.AppleModel()

    prompt = Mock()
    prompt.prompt = "Tell me a story"
    prompt.schema = None  # No schema
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024

    result = model.execute(
        prompt=prompt, stream=False, response=mock_response, conversation=None
    )

    # Verify schema parameter was not passed when None
    session.generate.assert_called_once()
    call_kwargs = session.generate.call_args[1]
    assert "schema" not in call_kwargs  # Schema should not be in kwargs when None

    # Result should be plain text
    assert result == "Generated response"


def test_schema_automatically_disables_streaming(
    mock_applefoundationmodels, mock_response
):
    """Test that schema automatically disables streaming."""
    from dataclasses import dataclass

    @dataclass
    class MockGenerationResponse:
        content: str = "Generated response"
        is_structured: bool = True
        tool_calls: list = None

        @property
        def text(self):
            if self.is_structured:
                raise ValueError("Response is structured")
            return self.content

        @property
        def parsed(self):
            if not self.is_structured:
                raise ValueError("Response is not structured")
            return {"name": "Alice"}

    session = Mock()
    session.generate = Mock(return_value=MockGenerationResponse())
    mock_applefoundationmodels.Session.return_value = session

    model = llm_apple.AppleModel()

    prompt = Mock()
    prompt.prompt = "Extract info"
    prompt.schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024

    # Even though stream=True is requested, schema should disable it
    result = model.execute(
        prompt=prompt, stream=True, response=mock_response, conversation=None
    )

    # Should return non-streaming result (string, not generator)
    assert isinstance(result, str)
    assert result == '{"name": "Alice"}'


@pytest.mark.asyncio
async def test_async_execute_with_schema(mock_applefoundationmodels, mock_response):
    """Test async execute method with schema returns parsed JSON."""
    from unittest.mock import AsyncMock
    from dataclasses import dataclass

    @dataclass
    class MockGenerationResponse:
        content: str = "Generated response"
        is_structured: bool = True
        tool_calls: list = None

        @property
        def text(self):
            if self.is_structured:
                raise ValueError("Response is structured")
            return self.content

        @property
        def parsed(self):
            if not self.is_structured:
                raise ValueError("Response is not structured")
            return {"name": "Bob", "age": 35, "city": "London"}

    # Update the async session mock
    session = AsyncMock()
    session.generate = AsyncMock(return_value=MockGenerationResponse())
    mock_applefoundationmodels.AsyncSession.return_value = session

    model = llm_apple.AppleAsyncModel()

    # Create prompt with schema
    prompt = Mock()
    prompt.prompt = "Extract person info: Bob is 35 and lives in London"
    prompt.schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
        },
        "required": ["name", "age", "city"],
    }
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024

    # Collect async results
    results = []
    async for chunk in model.execute(
        prompt=prompt, stream=False, response=mock_response, conversation=None
    ):
        results.append(chunk)

    # Verify schema was passed to generate
    session.generate.assert_called_once()
    call_kwargs = session.generate.call_args[1]
    assert call_kwargs["schema"] == prompt.schema

    # Result should be JSON-serialized parsed response
    assert len(results) == 1
    assert results[0] == json.dumps({"name": "Bob", "age": 35, "city": "London"})


@pytest.mark.asyncio
async def test_async_schema_automatically_disables_streaming(
    mock_applefoundationmodels, mock_response
):
    """Test that async schema automatically disables streaming."""
    from unittest.mock import AsyncMock
    from dataclasses import dataclass

    @dataclass
    class MockGenerationResponse:
        content: str = "Generated response"
        is_structured: bool = True
        tool_calls: list = None

        @property
        def text(self):
            if self.is_structured:
                raise ValueError("Response is structured")
            return self.content

        @property
        def parsed(self):
            if not self.is_structured:
                raise ValueError("Response is not structured")
            return {"name": "Bob"}

    session = AsyncMock()
    session.generate = AsyncMock(return_value=MockGenerationResponse())
    mock_applefoundationmodels.AsyncSession.return_value = session

    model = llm_apple.AppleAsyncModel()

    prompt = Mock()
    prompt.prompt = "Extract info"
    prompt.schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024

    # Even though stream=True is requested, schema should disable it
    results = []
    async for chunk in model.execute(
        prompt=prompt, stream=True, response=mock_response, conversation=None
    ):
        results.append(chunk)

    # Should return single result (streaming disabled)
    assert len(results) == 1
    assert results[0] == '{"name": "Bob"}'


def test_schema_passed_to_session_generate(mock_applefoundationmodels, mock_response):
    """Test that schema is correctly passed to session.generate()."""
    from dataclasses import dataclass

    @dataclass
    class MockGenerationResponse:
        content: str = "response"
        is_structured: bool = True
        tool_calls: list = None

        @property
        def text(self):
            if self.is_structured:
                raise ValueError("Response is structured")
            return self.content

        @property
        def parsed(self):
            return {"result": "data"}

    session = Mock()
    session.generate = Mock(return_value=MockGenerationResponse())
    mock_applefoundationmodels.Session.return_value = session

    model = llm_apple.AppleModel()

    test_schema = {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
        },
    }

    prompt = Mock()
    prompt.prompt = "Test"
    prompt.schema = test_schema
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 1024

    model.execute(
        prompt=prompt, stream=False, response=mock_response, conversation=None
    )

    # Verify the exact schema object was passed
    session.generate.assert_called_once_with(
        "Test", schema=test_schema, temperature=1.0, max_tokens=1024
    )
