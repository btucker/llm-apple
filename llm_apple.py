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
    register(AppleModel())


class AppleModel(llm.Model):
    """Apple Foundation Models (Apple Intelligence) integration."""

    model_id = "apple"
    can_stream = True
    supports_tools = True

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
        self._client = None
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

        from applefoundationmodels import Client, Availability

        status = Client.check_availability()
        if status != Availability.AVAILABLE:
            reason = Client.get_availability_reason()
            raise RuntimeError(
                f"Apple Intelligence not available: {reason or 'Unknown reason'}"
            )

        self._availability_checked = True

    def _get_client(self):
        """Get or create the client instance."""
        if self._client is None:
            from applefoundationmodels import Client

            self._check_availability()
            self._client = Client()
        return self._client

    def _create_session(self, instructions: Optional[str]):
        """Create a new session with the given instructions."""
        client = self._get_client()
        return client.create_session(instructions=instructions)

    def _get_session(self, conversation_id: Optional[str], instructions: Optional[str]):
        """Get or create a session for the conversation."""
        # If no conversation, create a new session each time
        if conversation_id is None:
            return self._create_session(instructions)

        # Reuse existing session for conversation
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = self._create_session(instructions)

        return self._sessions[conversation_id]

    def _register_tools_with_session(self, session, tools: list):
        """
        Register tools with the Apple Foundation Models session.

        Args:
            session: The session object
            tools: List of llm.Tool objects
        """
        if not tools:
            return

        def create_tool_wrapper(implementation):
            """Factory function to create tool wrapper with proper closure."""

            def wrapper(*args, **kwargs):
                return implementation(*args, **kwargs)

            return wrapper

        # Ensure _tools dict exists
        if not hasattr(session, "_tools"):
            session._tools = {}

        # Add all tools to the session
        for tool in tools:
            # Create a wrapper function with proper closure
            tool_func = create_tool_wrapper(tool.implementation)

            # Set the tool metadata directly on the function
            tool_func._tool_name = tool.name
            tool_func._tool_description = tool.description or ""
            tool_func._tool_parameters = tool.input_schema

            # Add tool to session's tools dict
            session._tools[tool.name] = tool_func

        # Register all tools at once with the FFI layer
        if hasattr(session, "_register_tools") and len(session._tools) > 0:
            session._register_tools()

    def _extract_tool_calls_from_transcript(self, transcript: list) -> list:
        """
        Extract tool calls from session transcript.

        Args:
            transcript: Session transcript from Apple Foundation Models

        Returns:
            List of llm.ToolCall objects
        """
        tool_calls = []

        for entry in transcript:
            if entry.get("type") == "tool_calls":
                for call in entry.get("tool_calls", []):
                    tool_calls.append(
                        llm.ToolCall(
                            name=call.get("name", ""),
                            arguments=json.loads(call.get("arguments", "{}")),
                            tool_call_id=call.get("id"),
                        )
                    )

        return tool_calls

    def _add_tool_results_to_session(self, session, tool_results: list):
        """
        Add tool results to the session history.

        Args:
            session: The session object
            tool_results: List of llm.ToolResult objects
        """
        if not tool_results:
            return

        # Add tool results as messages to the session
        for result in tool_results:
            # Add the tool result as a user message containing the result
            content = f"Tool {result.name} returned: {result.output}"
            session.add_message("user", content)

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

        # Check if we have tools - if so, we need a fresh session with tools
        has_tools = self._is_valid_list_attribute(prompt, "tools")

        if has_tools:
            # Create a new session specifically for tool calling
            # (can't add tools to existing session)
            session = self._create_session(system_prompt)
            self._register_tools_with_session(session, prompt.tools)

            # Store the tool-enabled session for this conversation
            # so subsequent turns reuse it instead of reverting to old session
            if conversation_id is not None:
                self._sessions[conversation_id] = session
        else:
            # Get or create session (may reuse for conversations)
            session = self._get_session(conversation_id, system_prompt)

        # Add tool results to session if provided
        has_tool_results = self._is_valid_list_attribute(prompt, "tool_results")
        if has_tool_results:
            self._add_tool_results_to_session(session, prompt.tool_results)

        # Get the actual prompt text (may be None for tool-only prompts)
        prompt_text = getattr(prompt, "prompt", None) or ""

        # If we have no prompt text but have tool results, create a continuation prompt
        if not prompt_text and has_tool_results:
            prompt_text = "Please continue based on the tool results above."

        # Generate response
        if stream:
            result = self._stream_response(
                session, prompt_text, temperature, max_tokens
            )
        else:
            result = self._generate_response(
                session, prompt_text, temperature, max_tokens
            )

        # Extract tool calls from transcript and add to response
        transcript = getattr(session, "transcript", None)
        if transcript and isinstance(transcript, (list, tuple)):
            tool_calls = self._extract_tool_calls_from_transcript(transcript)
            for tool_call in tool_calls:
                response.add_tool_call(tool_call)

        return result

    def _get_or_create_event_loop(self):
        """Get existing event loop or create a new one."""
        import asyncio

        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _stream_response(self, session, prompt_text, temperature, max_tokens):
        """Stream response tokens."""
        import asyncio

        loop = self._get_or_create_event_loop()

        # Run the async generator in the event loop
        async def _async_stream():
            async for chunk in session.generate_stream(
                prompt_text, temperature=temperature, max_tokens=max_tokens
            ):
                yield chunk

        # Convert async generator to sync generator
        async_gen = _async_stream()

        while True:
            try:
                chunk = loop.run_until_complete(async_gen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break

    def _generate_response(self, session, prompt_text, temperature, max_tokens):
        """Generate a complete response."""
        response = session.generate(
            prompt_text, temperature=temperature, max_tokens=max_tokens
        )
        return response
