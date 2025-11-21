"""
LLM plugin for Apple Foundation Models (Apple Intelligence)

This plugin exposes Apple's on-device Foundation Models through the llm CLI.
"""

import llm
from pydantic import Field
from typing import Optional, Dict, Any
import json

# Default configuration values
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 1024


@llm.hookimpl
def register_models(register):
    """Register Apple Foundation Models with llm."""
    register(AppleModel(), AppleAsyncModel())


class AppleModel(llm.Model):
    """Apple Foundation Models (Apple Intelligence) integration."""

    model_id = "apple"
    can_stream = True
    supports_tools = True
    supports_schemas = True

    class Options(llm.Options):
        """Options for Apple Foundation Models generation."""

        temperature: Optional[float] = Field(
            default=DEFAULT_TEMPERATURE,
            ge=0.0,
            le=2.0,
            description="Sampling temperature (0.0 = deterministic, 2.0 = very random)",
        )
        max_tokens: Optional[int] = Field(
            default=DEFAULT_MAX_TOKENS, gt=0, description="Maximum tokens to generate"
        )

    def __init__(self):
        """Initialize the Apple model."""
        self._sessions = {}
        self._availability_checked = False

    def _get_option_value(self, options, attr_name, default=None):
        """Get an option value with a fallback default."""
        if not hasattr(options, attr_name):
            return default
        value = getattr(options, attr_name)
        return value if value is not None else default

    def _is_valid_list_attribute(self, obj, attr_name):
        """
        Check if an object has a non-empty list/tuple attribute.

        Args:
            obj: Object to check
            attr_name: Attribute name to check

        Returns:
            bool: True if attribute exists, is not None, and is a non-empty list/tuple
        """
        value = getattr(obj, attr_name, None)
        return value and isinstance(value, (list, tuple)) and len(value) > 0

    def _check_availability(self):
        """Check Apple Intelligence availability (lazy check)."""
        if self._availability_checked:
            return

        from applefoundationmodels import Session, Availability

        status = Session.check_availability()
        if status != Availability.AVAILABLE:
            reason = Session.get_availability_reason()
            raise RuntimeError(
                f"Apple Intelligence not available: {reason or 'Unknown reason'}"
            )

        self._availability_checked = True

    def _create_session(
        self, instructions: Optional[str], tools: Optional[list] = None
    ):
        """Create a new session with the given instructions and tools."""
        from applefoundationmodels import Session

        self._check_availability()

        # Convert llm.Tool objects to Python functions for apple-foundation-models
        tool_functions = []
        if tools:
            for tool in tools:
                # Get the implementation function
                func = tool.implementation

                # Set docstring and type hints to match the tool schema
                if not func.__doc__:
                    func.__doc__ = tool.description or ""

                tool_functions.append(func)

        # Create session with tools if provided
        if tool_functions:
            return Session(instructions=instructions, tools=tool_functions)
        else:
            return Session(instructions=instructions)

    def _get_session(
        self,
        conversation_id: Optional[str],
        instructions: Optional[str],
        tools: Optional[list] = None,
    ):
        """Get or create a session for the conversation."""
        # If no conversation, create a new session each time
        if conversation_id is None:
            return self._create_session(instructions, tools)

        # Reuse existing session for conversation (without tools on first create)
        # Note: If tools are provided, we need to create a new session since
        # tools are registered at session creation time in 0.2.0
        if conversation_id not in self._sessions or tools:
            self._sessions[conversation_id] = self._create_session(instructions, tools)

        return self._sessions[conversation_id]

    def _extract_tool_calls_from_response(self, response: "GenerationResponse") -> list:
        """
        Extract tool calls from GenerationResponse.

        Args:
            response: GenerationResponse from Apple Foundation Models 0.2.0+

        Returns:
            List of llm.ToolCall objects
        """
        tool_calls = []

        if hasattr(response, "tool_calls") and response.tool_calls:
            for call in response.tool_calls:
                tool_calls.append(
                    llm.ToolCall(
                        name=call.function.name,
                        arguments=json.loads(call.function.arguments),
                        tool_call_id=call.id,
                    )
                )

        return tool_calls

    def _format_tool_results_as_prompt(self, tool_results: list) -> str:
        """
        Format tool results as a prompt continuation.

        In 0.2.0, apple-foundation-models handles tool execution automatically.
        When llm sends us tool_results (from its own execution), we format them
        as part of the next prompt.

        Args:
            tool_results: List of llm.ToolResult objects

        Returns:
            Formatted string with tool results
        """
        if not tool_results:
            return ""

        result_parts = []
        for result in tool_results:
            result_parts.append(f"{result.name}() returned: {result.output}")

        return "\n".join(result_parts)

    def execute(self, prompt, stream, response, conversation):
        """Execute a prompt against Apple Foundation Models."""
        # Extract options using helper method
        temperature = self._get_option_value(
            prompt.options, "temperature", DEFAULT_TEMPERATURE
        )
        max_tokens = self._get_option_value(
            prompt.options, "max_tokens", DEFAULT_MAX_TOKENS
        )

        # Use llm's built-in system prompt support
        system_prompt = getattr(prompt, "system", None)

        # Get conversation ID if available
        conversation_id = conversation.id if conversation else None

        # Check if we have tools
        has_tools = self._is_valid_list_attribute(prompt, "tools")
        tools = prompt.tools if has_tools else None

        # Check if we have a schema (schema should be a dict if provided)
        schema = getattr(prompt, "schema", None)
        # Only use schema if it's actually a dict (avoid Mock objects and None)
        schema = schema if isinstance(schema, dict) else None

        # Schema and streaming are incompatible
        if schema is not None and stream:
            raise ValueError(
                "Schema-based structured output is not compatible with streaming"
            )

        # Get or create session with tools
        # In 0.2.0, tools are registered at session creation time
        session = self._get_session(conversation_id, system_prompt, tools)

        # Get the actual prompt text (may be None for tool-only prompts)
        prompt_text = getattr(prompt, "prompt", None) or ""

        # Handle tool results if provided
        has_tool_results = self._is_valid_list_attribute(prompt, "tool_results")
        if has_tool_results:
            # Format tool results as part of the prompt
            tool_results_text = self._format_tool_results_as_prompt(prompt.tool_results)
            if tool_results_text:
                if prompt_text:
                    prompt_text = f"{tool_results_text}\n\n{prompt_text}"
                else:
                    prompt_text = f"{tool_results_text}\n\nPlease continue based on these results."

        # Generate response
        if stream:
            # For streaming, we can't extract tool calls from the chunks
            # Tool calls are only available in the non-streaming response
            result = self._stream_response(
                session, prompt_text, temperature, max_tokens
            )
        else:
            # Get the full response object to extract tool calls or structured output
            # Only pass schema parameter if it's actually set
            if schema is not None:
                gen_response = session.generate(
                    prompt_text,
                    schema=schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                gen_response = session.generate(
                    prompt_text, temperature=temperature, max_tokens=max_tokens
                )

            # Extract tool calls from response.tool_calls (0.2.0+ API)
            tool_calls = self._extract_tool_calls_from_response(gen_response)
            for tool_call in tool_calls:
                response.add_tool_call(tool_call)

            # Return the parsed content if schema was provided, otherwise text
            if schema is not None:
                result = json.dumps(gen_response.parsed)
            else:
                result = gen_response.text

        return result

    def _stream_response(self, session, prompt_text, temperature, max_tokens):
        """Stream response tokens."""
        # In 0.2.0, generate() with stream=True returns a sync iterator
        for chunk in session.generate(
            prompt_text, stream=True, temperature=temperature, max_tokens=max_tokens
        ):
            yield chunk.content


class AppleAsyncModel(llm.AsyncModel):
    """Async Apple Foundation Models (Apple Intelligence) integration."""

    model_id = "apple"
    can_stream = True
    supports_tools = True
    supports_schemas = True

    class Options(llm.Options):
        """Options for Apple Foundation Models generation."""

        temperature: Optional[float] = Field(
            default=DEFAULT_TEMPERATURE,
            ge=0.0,
            le=2.0,
            description="Sampling temperature (0.0 = deterministic, 2.0 = very random)",
        )
        max_tokens: Optional[int] = Field(
            default=DEFAULT_MAX_TOKENS, gt=0, description="Maximum tokens to generate"
        )

    def __init__(self):
        """Initialize the Apple async model."""
        self._sessions = {}
        self._availability_checked = False

    def _get_option_value(self, options, attr_name, default=None):
        """Get an option value with a fallback default."""
        if not hasattr(options, attr_name):
            return default
        value = getattr(options, attr_name)
        return value if value is not None else default

    def _is_valid_list_attribute(self, obj, attr_name):
        """
        Check if an object has a non-empty list/tuple attribute.

        Args:
            obj: Object to check
            attr_name: Attribute name to check

        Returns:
            bool: True if attribute exists, is not None, and is a non-empty list/tuple
        """
        value = getattr(obj, attr_name, None)
        return value and isinstance(value, (list, tuple)) and len(value) > 0

    def _check_availability(self):
        """Check Apple Intelligence availability (lazy check)."""
        if self._availability_checked:
            return

        from applefoundationmodels import AsyncSession, Availability

        status = AsyncSession.check_availability()
        if status != Availability.AVAILABLE:
            reason = AsyncSession.get_availability_reason()
            raise RuntimeError(
                f"Apple Intelligence not available: {reason or 'Unknown reason'}"
            )

        self._availability_checked = True

    def _create_session(
        self, instructions: Optional[str], tools: Optional[list] = None
    ):
        """Create a new async session with the given instructions and tools."""
        from applefoundationmodels import AsyncSession

        self._check_availability()

        # Convert llm.Tool objects to Python functions for apple-foundation-models
        tool_functions = []
        if tools:
            for tool in tools:
                # Get the implementation function
                func = tool.implementation

                # Set docstring and type hints to match the tool schema
                if not func.__doc__:
                    func.__doc__ = tool.description or ""

                tool_functions.append(func)

        # Create session with tools if provided
        if tool_functions:
            return AsyncSession(instructions=instructions, tools=tool_functions)
        else:
            return AsyncSession(instructions=instructions)

    def _get_session(
        self,
        conversation_id: Optional[str],
        instructions: Optional[str],
        tools: Optional[list] = None,
    ):
        """Get or create an async session for the conversation."""
        # If no conversation, create a new session each time
        if conversation_id is None:
            return self._create_session(instructions, tools)

        # Reuse existing session for conversation
        # Note: If tools are provided, we need to create a new session since
        # tools are registered at session creation time in 0.2.0
        if conversation_id not in self._sessions or tools:
            self._sessions[conversation_id] = self._create_session(instructions, tools)

        return self._sessions[conversation_id]

    def _extract_tool_calls_from_response(self, response: "GenerationResponse") -> list:
        """
        Extract tool calls from GenerationResponse.

        Args:
            response: GenerationResponse from Apple Foundation Models 0.2.0+

        Returns:
            List of llm.ToolCall objects
        """
        tool_calls = []

        if hasattr(response, "tool_calls") and response.tool_calls:
            for call in response.tool_calls:
                tool_calls.append(
                    llm.ToolCall(
                        name=call.function.name,
                        arguments=json.loads(call.function.arguments),
                        tool_call_id=call.id,
                    )
                )

        return tool_calls

    def _format_tool_results_as_prompt(self, tool_results: list) -> str:
        """
        Format tool results as a prompt continuation.

        In 0.2.0, apple-foundation-models handles tool execution automatically.
        When llm sends us tool_results (from its own execution), we format them
        as part of the next prompt.

        Args:
            tool_results: List of llm.ToolResult objects

        Returns:
            Formatted string with tool results
        """
        if not tool_results:
            return ""

        result_parts = []
        for result in tool_results:
            result_parts.append(f"{result.name}() returned: {result.output}")

        return "\n".join(result_parts)

    async def execute(self, prompt, stream, response, conversation):
        """Execute a prompt against Apple Foundation Models asynchronously."""
        # Extract options using helper method
        temperature = self._get_option_value(
            prompt.options, "temperature", DEFAULT_TEMPERATURE
        )
        max_tokens = self._get_option_value(
            prompt.options, "max_tokens", DEFAULT_MAX_TOKENS
        )

        # Use llm's built-in system prompt support
        system_prompt = getattr(prompt, "system", None)

        # Get conversation ID if available
        conversation_id = conversation.id if conversation else None

        # Check if we have tools
        has_tools = self._is_valid_list_attribute(prompt, "tools")
        tools = prompt.tools if has_tools else None

        # Check if we have a schema (schema should be a dict if provided)
        schema = getattr(prompt, "schema", None)
        # Only use schema if it's actually a dict (avoid Mock objects and None)
        schema = schema if isinstance(schema, dict) else None

        # Schema and streaming are incompatible
        if schema is not None and stream:
            raise ValueError(
                "Schema-based structured output is not compatible with streaming"
            )

        # Get or create session with tools
        # In 0.2.0, tools are registered at session creation time
        session = self._get_session(conversation_id, system_prompt, tools)

        # Get the actual prompt text (may be None for tool-only prompts)
        prompt_text = getattr(prompt, "prompt", None) or ""

        # Handle tool results if provided
        has_tool_results = self._is_valid_list_attribute(prompt, "tool_results")
        if has_tool_results:
            # Format tool results as part of the prompt
            tool_results_text = self._format_tool_results_as_prompt(prompt.tool_results)
            if tool_results_text:
                if prompt_text:
                    prompt_text = f"{tool_results_text}\n\n{prompt_text}"
                else:
                    prompt_text = f"{tool_results_text}\n\nPlease continue based on these results."

        # Generate response
        if stream:
            # For streaming, we can't extract tool calls from the chunks
            # Tool calls are only available in the non-streaming response
            async for chunk_text in self._stream_response(
                session, prompt_text, temperature, max_tokens
            ):
                yield chunk_text
        else:
            # Get the full response object to extract tool calls or structured output
            # Only pass schema parameter if it's actually set
            if schema is not None:
                gen_response = await session.generate(
                    prompt_text,
                    schema=schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                gen_response = await session.generate(
                    prompt_text, temperature=temperature, max_tokens=max_tokens
                )

            # Extract tool calls from response.tool_calls (0.2.0+ API)
            tool_calls = self._extract_tool_calls_from_response(gen_response)
            for tool_call in tool_calls:
                response.add_tool_call(tool_call)

            # Return the parsed content if schema was provided, otherwise text
            if schema is not None:
                yield json.dumps(gen_response.parsed)
            else:
                yield gen_response.text

    async def _stream_response(self, session, prompt_text, temperature, max_tokens):
        """Stream response tokens asynchronously."""
        # In 0.2.0, generate() with stream=True returns an async iterator
        async for chunk in session.generate(
            prompt_text, stream=True, temperature=temperature, max_tokens=max_tokens
        ):
            yield chunk.content
