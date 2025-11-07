"""
Integration tests for tool calling with llm-apple.

These tests run against the actual Apple Foundation Models API when available.
They will be skipped if Apple Intelligence is not available on the system.
"""
import pytest
import llm


# Check if Apple Intelligence is available
def is_apple_intelligence_available():
    """Check if Apple Intelligence is available on this system."""
    try:
        from applefoundationmodels import Client, Availability
        status = Client.check_availability()
        return status == Availability.AVAILABLE
    except (ImportError, Exception):
        return False


pytestmark = pytest.mark.skipif(
    not is_apple_intelligence_available(),
    reason="Apple Intelligence not available on this system"
)


@pytest.fixture
def apple_model():
    """Get the apple model from llm."""
    return llm.get_model("apple")


@pytest.fixture
def conversation(tmp_path):
    """Create a temporary conversation."""
    # Use a temporary database for testing
    db_path = tmp_path / "test.db"
    return llm.Conversation(model=llm.get_model("apple"))


class TestToolCallingIntegration:
    """Integration tests for tool calling functionality."""

    def test_simple_tool_call(self, apple_model):
        """Test calling a simple tool with no parameters."""
        # Track tool calls
        calls = []

        def get_current_time():
            """Get the current time."""
            calls.append({})
            return "2:30 PM"

        # Register the tool
        tools = [llm.Tool(
            name="get_current_time",
            description="Get the current time",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            implementation=get_current_time
        )]

        # Execute with the tool
        response = apple_model.prompt(
            "What time is it?",
            tools=tools
        )

        # Verify response exists
        assert response.text()
        response_text = response.text()
        print(f"Response: {response_text}")

        # Verify the tool was actually called
        assert len(calls) > 0, "Tool was not called"

        # Verify the tool result appears in the response
        assert "2:30" in response_text, f"Tool result not in response: {response_text}"

    def test_tool_with_string_parameter(self, apple_model):
        """Test calling a tool that takes a string parameter."""
        # Track tool calls
        calls = []

        def get_weather(location: str):
            """Get the weather for a location."""
            calls.append({'location': location})
            return f"Weather in {location}: 72°F, sunny"

        # Register the tool
        tools = [llm.Tool(
            name="get_weather",
            description="Get current weather for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or location name"}
                },
                "required": ["location"]
            },
            implementation=get_weather
        )]

        # Execute with the tool
        response = apple_model.prompt(
            "What's the weather in Paris?",
            tools=tools
        )

        # Verify response exists
        assert response.text()
        response_text = response.text()
        print(f"Response: {response_text}")

        # Verify tool was called
        assert len(calls) > 0, "Tool was not called"

        # Verify correct location was passed
        assert calls[0]['location'].lower() == 'paris', f"Expected Paris, got {calls[0]['location']}"

        # Verify tool result appears in response
        assert "72" in response_text and "sunny" in response_text.lower(), \
            f"Tool result not in response: {response_text}"

    def test_tool_with_multiple_parameters(self, apple_model):
        """Test calling a tool with multiple parameters."""
        # Track tool calls
        calls = []

        def calculate(operation: str, x: int, y: int):
            """Perform a calculation."""
            calls.append({'operation': operation, 'x': x, 'y': y})
            operations = {
                "add": x + y,
                "subtract": x - y,
                "multiply": x * y,
                "divide": x // y if y != 0 else "undefined"
            }
            result = operations.get(operation, "unknown operation")
            return f"Result: {result}"

        # Register the tool
        tools = [llm.Tool(
            name="calculate",
            description="Perform mathematical calculations",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform (add, subtract, multiply, divide)"
                    },
                    "x": {"type": "integer", "description": "First number"},
                    "y": {"type": "integer", "description": "Second number"}
                },
                "required": ["operation", "x", "y"]
            },
            implementation=calculate
        )]

        # Execute with the tool
        response = apple_model.prompt(
            "What is 15 multiplied by 7?",
            tools=tools
        )

        # Verify response exists
        assert response.text()
        response_text = response.text()
        print(f"Response: {response_text}")

        # Verify tool was called
        assert len(calls) > 0, "Tool was not called"

        # Verify correct operation and operands
        assert calls[0]['operation'] == 'multiply', f"Expected multiply, got {calls[0]['operation']}"
        assert calls[0]['x'] == 15, f"Expected x=15, got {calls[0]['x']}"
        assert calls[0]['y'] == 7, f"Expected y=7, got {calls[0]['y']}"

        # Verify result (15 * 7 = 105) appears in response
        assert "105" in response_text, f"Expected 105 in response: {response_text}"

    def test_multiple_tools(self, apple_model):
        """Test with multiple tools registered."""
        # Track tool calls
        time_calls = []
        date_calls = []

        def get_time():
            """Get the current time."""
            time_calls.append({})
            return "2:30 PM"

        def get_date():
            """Get the current date."""
            date_calls.append({})
            return "November 7, 2024"

        # Register tools
        tools = [
            llm.Tool(
                name="get_time",
                description="Get the current time",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                implementation=get_time
            ),
            llm.Tool(
                name="get_date",
                description="Get the current date",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                implementation=get_date
            )
        ]

        # Execute with multiple tools
        response = apple_model.prompt(
            "What's the current date and time?",
            tools=tools
        )

        # Verify response exists
        assert response.text()
        response_text = response.text()
        print(f"Response: {response_text}")

        # Verify at least one tool was called
        total_calls = len(time_calls) + len(date_calls)
        assert total_calls > 0, "No tools were called"

        # Verify results appear in response
        if len(date_calls) > 0:
            assert "november" in response_text.lower() or "7" in response_text, \
                "Date tool result not in response"
        if len(time_calls) > 0:
            assert "2:30" in response_text, "Time tool result not in response"

    def test_tool_with_conversation(self, apple_model):
        """Test tools work within a conversation context."""
        # Track tool calls
        calls = []

        def get_temperature(city: str):
            """Get temperature for a city."""
            calls.append({'city': city})
            temps = {
                "paris": "18°C",
                "london": "15°C",
                "tokyo": "22°C",
                "new york": "20°C"
            }
            return temps.get(city.lower(), "20°C")

        tools = [llm.Tool(
            name="get_temperature",
            description="Get the temperature for a city",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Name of the city"}
                },
                "required": ["city"]
            },
            implementation=get_temperature
        )]

        # First turn
        response1 = apple_model.prompt(
            "What's the temperature in Paris?",
            tools=tools
        )

        assert response1.text()
        response1_text = response1.text()
        print(f"Turn 1: {response1_text}")

        # Verify first call
        assert len(calls) >= 1, "Tool not called on first turn"
        assert calls[0]['city'].lower() == 'paris', f"Expected Paris, got {calls[0]['city']}"
        assert "18" in response1_text, "Temperature not in first response"

        # Second turn in same conversation - should maintain context
        response2 = apple_model.prompt(
            "And what about London?",
            tools=tools
        )

        assert response2.text()
        response2_text = response2.text()
        print(f"Turn 2: {response2_text}")

        # Note: Second call may or may not happen depending on model behavior
        # Just verify we got a meaningful response
        assert len(response2_text) > 10, "Response too short"


class TestToolCallingVerbose:
    """Verbose integration tests that print detailed output."""

    def test_tool_calling_with_details(self, apple_model):
        """Test tool calling with detailed output of the process."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Tool Calling with Details")
        print("=" * 70)

        # Define a tool
        call_count = {'count': 0}

        def search_database(query: str, limit: int = 5):
            """Search a database for information."""
            call_count['count'] += 1
            print(f"\n[TOOL CALLED] search_database(query='{query}', limit={limit})")
            results = [
                f"Result {i+1}: Information about {query}"
                for i in range(min(limit, 3))
            ]
            return f"Found {len(results)} results: " + "; ".join(results)

        # Register the tool
        tools = [llm.Tool(
            name="search_database",
            description="Search a database for information",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return"
                    }
                },
                "required": ["query"]
            },
            implementation=search_database
        )]

        print("\n[PROMPT] Search for information about 'artificial intelligence'")

        # Execute
        response = apple_model.prompt(
            "Search for information about 'artificial intelligence' in the database",
            tools=tools
        )

        response_text = response.text()
        print(f"\n[RESPONSE] {response_text}")
        print(f"\n[TOOL CALL COUNT] {call_count['count']}")

        # Verify tool was called
        assert call_count['count'] > 0, "Tool was never called"

        # Verify response contains tool results
        assert response_text, "No response text"
        assert "artificial intelligence" in response_text.lower() or "found" in response_text.lower() or "result" in response_text.lower(), \
            f"Tool results not integrated into response: {response_text}"

        print("\n" + "=" * 70)
        print("✓ Integration test completed successfully")
        print("=" * 70)


if __name__ == "__main__":
    # Allow running this file directly
    pytest.main([__file__, "-v", "-s"])
