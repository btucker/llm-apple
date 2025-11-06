"""
LLM plugin for Apple Foundation Models (Apple Intelligence)

This plugin exposes Apple's on-device Foundation Models through the llm CLI.
"""
import llm
from pydantic import Field
from typing import Optional

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

    class Options(llm.Options):
        """Options for Apple Foundation Models generation."""
        temperature: Optional[float] = Field(
            default=DEFAULT_TEMPERATURE,
            ge=0.0,
            le=2.0,
            description="Sampling temperature (0.0 = deterministic, 2.0 = very random)"
        )
        max_tokens: Optional[int] = Field(
            default=DEFAULT_MAX_TOKENS,
            gt=0,
            description="Maximum tokens to generate"
        )
        instructions: Optional[str] = Field(
            default=None,
            description="System instructions to guide AI behavior"
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

    def execute(self, prompt, stream, response, conversation):
        """Execute a prompt against Apple Foundation Models."""
        # Extract options using helper method
        temperature = self._get_option_value(prompt.options, 'temperature', DEFAULT_TEMPERATURE)
        max_tokens = self._get_option_value(prompt.options, 'max_tokens', DEFAULT_MAX_TOKENS)
        instructions = self._get_option_value(prompt.options, 'instructions', None)

        # Get conversation ID if available
        conversation_id = conversation.id if conversation else None

        # Get or create session
        session = self._get_session(conversation_id, instructions)

        # Generate response
        if stream:
            return self._stream_response(
                session,
                prompt.prompt,
                temperature,
                max_tokens
            )
        else:
            return self._generate_response(
                session,
                prompt.prompt,
                temperature,
                max_tokens
            )

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
                prompt_text,
                temperature=temperature,
                max_tokens=max_tokens
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
            prompt_text,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response
