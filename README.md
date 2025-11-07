# llm-apple

LLM plugin for Apple Foundation Models (Apple Intelligence)

This plugin exposes Apple's on-device Foundation Models through the [llm](https://llm.datasette.io/) CLI tool.

## Requirements

- macOS 15.1 or later
- Apple Silicon (M1, M2, M3, or later)
- Apple Intelligence enabled
- Python 3.9 or later
- [apple-foundation-models](https://pypi.org/project/apple-foundation-models/) >= 0.1.5 installed

## Installation

```bash
pip install llm # if llm is not already installed
llm install llm-apple
```

## Usage

Basic usage (streaming is enabled by default):

```bash
llm -m apple "What is the capital of France?"
```

Without streaming:

```bash
llm -m apple "Tell me a story" --no-stream
```

With options:

```bash
llm -m apple "Write a poem" -o temperature 1.5 -o max_tokens 500
```

With system instructions:

```bash
llm -m apple "What is Python?" --system "You are a helpful programming tutor"
```

### Conversations

The plugin supports conversations, maintaining context across multiple prompts:

```bash
# Start a conversation
llm -m apple "My name is Alice" --save conversation1

# Continue the conversation
llm -m apple "What is my name?" --continue conversation1
```

### Tool Calling

The plugin supports tool calling, allowing the model to call Python functions to access real-time data, perform actions, or integrate with external systems.

#### Basic Tool Usage

```python
import llm

def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # In a real implementation, this would call a weather API
    return f"Weather in {location}: 72°F, sunny"

model = llm.get_model("apple")
response = model.prompt(
    "What's the weather in San Francisco?",
    tools=[llm.Tool(
        name="get_weather",
        description="Get current weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        },
        implementation=get_weather
    )]
)
print(response.text())
```

#### Tool Types Supported

Tools can have various parameter signatures:

**No parameters:**
```python
def get_current_time() -> str:
    """Get the current time."""
    return "2:30 PM"
```

**Single parameter:**
```python
def search_docs(query: str) -> str:
    """Search documentation."""
    return f"Results for: {query}"
```

**Multiple parameters with mixed types:**
```python
def calculate(operation: str, x: int, y: int) -> str:
    """Perform a calculation."""
    ops = {"add": x + y, "multiply": x * y}
    return str(ops.get(operation, 0))
```

**Optional parameters:**
```python
def get_temperature(city: str, units: str = "celsius") -> str:
    """Get temperature for a city."""
    return f"Temperature in {city}: 20°{units[0].upper()}"
```

#### Multiple Tools

You can register multiple tools in a single call:

```python
tools = [
    llm.Tool(name="get_time", description="Get current time", ...),
    llm.Tool(name="get_date", description="Get current date", ...),
    llm.Tool(name="get_weather", description="Get weather", ...),
]

response = model.prompt(
    "What's the date, time, and weather?",
    tools=tools
)
```

The model will automatically select and call the appropriate tools based on the prompt.

### Available Options

- `temperature` (float, 0.0-2.0, default: 1.0): Controls randomness in generation
  - 0.0 = deterministic
  - 2.0 = very random
- `max_tokens` (int, default: 1024): Maximum tokens to generate

System prompts can be provided using llm's built-in `--system` or `-s` flag.

## Availability

The plugin checks Apple Intelligence availability on startup. If Apple Intelligence is not available, you'll see an error message with details on why.

Common reasons:

- Device not eligible (requires Apple Silicon)
- Apple Intelligence not enabled in Settings
- Model not ready (downloading or initializing)

## Examples

Creative writing with higher temperature:

```bash
llm -m apple "Write a creative story about a robot" -o temperature 1.8
```

Factual query with lower temperature:

```bash
llm -m apple "Explain quantum computing" -o temperature 0.3
```

With system prompt for career guidance:

```bash
llm -m apple "Should I learn Python or JavaScript?" \
  --system "You are a career counselor specializing in tech"
```

## Development

### Running Tests

```bash
# Run all tests (unit tests with mocks)
uv run pytest

# Run tests with coverage
uv run pytest --cov=llm_apple --cov-report=html --cov-report=term

# Run integration tests (requires Apple Intelligence)
uv run pytest tests/test_integration_tools.py -v -s
```

Most tests use mocks to simulate the Apple Foundation Models API, so they can run on any platform without requiring actual Apple Intelligence hardware.

Integration tests in `tests/test_integration_tools.py` require Apple Intelligence to be available and will be automatically skipped if not present.
