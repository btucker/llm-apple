# llm-apple

LLM plugin for Apple Foundation Models (Apple Intelligence)

This plugin exposes Apple's on-device Foundation Models through the [llm](https://llm.datasette.io/) CLI tool.

## Requirements

- macOS 15.1 or later
- Apple Silicon (M1/M2/M3/M4) or compatible device
- Apple Intelligence enabled
- Python 3.8 or later
- apple-foundation-models-py installed

## Installation

First, install apple-foundation-models:

```bash
# Clone and install apple-foundation-models-py
git clone https://github.com/btucker/apple-foundation-models-py
cd apple-foundation-models-py
pip install -e .
```

Then install this plugin:

```bash
# Install from path
llm install -e /path/to/llm-apple

# Or from current directory
cd llm-apple
llm install -e .
```

**Note:** `apple-foundation-models` is a runtime dependency that requires building from source with Xcode Command Line Tools installed. It's not required for running tests, which use mocks.

## Usage

Basic usage:

```bash
llm -m apple "What is the capital of France?"
```

With streaming:

```bash
llm -m apple "Tell me a story" --stream
```

With options:

```bash
llm -m apple "Write a poem" -o temperature 1.5 -o max_tokens 500
```

With system instructions:

```bash
llm -m apple "What is Python?" -o instructions "You are a helpful programming tutor"
```

### Conversations

The plugin supports conversations, maintaining context across multiple prompts:

```bash
# Start a conversation
llm -m apple "My name is Alice" --save conversation1

# Continue the conversation
llm -m apple "What is my name?" --continue conversation1
```

### Available Options

- `temperature` (float, 0.0-2.0, default: 1.0): Controls randomness in generation
  - 0.0 = deterministic
  - 2.0 = very random
- `max_tokens` (int, default: 1024): Maximum tokens to generate
- `instructions` (str): System instructions to guide AI behavior

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

With system instructions:

```bash
llm -m apple "Should I learn Python or JavaScript?" \
  -o instructions "You are a career counselor specializing in tech"
```

## Development

### Running Tests

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=llm_apple --cov-report=html --cov-report=term
```

The tests use mocks to simulate the Apple Foundation Models API, so they can run on any platform without requiring actual Apple Intelligence hardware.
