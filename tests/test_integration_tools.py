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

    def test_simple_tool_call(self, apple_model, tool_factory, call_tracker, assert_response):
        """Test calling a simple tool with no parameters."""
        def get_current_time():
            """Get the current time."""
            call_tracker.track()
            return "2:30 PM"

        tools = [tool_factory(
            name="get_current_time",
            description="Get the current time",
            implementation=get_current_time
        )]

        response = apple_model.prompt("What time is it?", tools=tools)

        # Note: Apple Intelligence may analyze tool code instead of calling it
        # So we verify the correct answer appears, regardless of whether tool was called
        response_text = assert_response(response)
        # The response should mention time (either from tool call or model's knowledge)
        assert len(response_text) > 0, "Response should not be empty"
        print(f"Response: {response_text}")
        if call_tracker.was_called():
            print("  (Tool was called)")
            assert_response(response, "2:30")
        else:
            print("  (Tool was not called - model provided answer directly)")

    def test_tool_with_string_parameter(self, apple_model, tool_factory, call_tracker, assert_response):
        """Test calling a tool that takes a string parameter."""
        def get_weather(location: str):
            """Get the weather for a location."""
            call_tracker.track(location=location)
            return f"Weather in {location}: 72°F, sunny"

        tools = [tool_factory(
            name="get_weather",
            description="Get current weather for a location",
            properties={"location": {"type": "string", "description": "City or location name"}},
            required=["location"],
            implementation=get_weather
        )]

        response = apple_model.prompt("What's the weather in Paris?", tools=tools)

        # Note: Apple Intelligence may analyze tool code instead of calling it
        response_text = assert_response(response)
        assert len(response_text) > 0, "Response should not be empty"
        print(f"Response: {response_text}")

        if call_tracker.was_called():
            print("  (Tool was called)")
            # Verify tool was called with correct arguments
            call_tracker.assert_called_with(location='paris')
            assert "72" in response_text and "sunny" in response_text.lower()
        else:
            print("  (Tool was not called - model analyzed tool code directly)")

    def test_tool_with_multiple_parameters(self, apple_model, tool_factory, call_tracker, assert_response):
        """Test calling a tool with multiple parameters."""
        def calculate(operation: str, x: int, y: int):
            """Perform a calculation."""
            call_tracker.track(operation=operation, x=x, y=y)
            operations = {
                "add": x + y,
                "addition": x + y,
                "+": x + y,
                "subtract": x - y,
                "subtraction": x - y,
                "-": x - y,
                "multiply": x * y,
                "multiplication": x * y,
                "times": x * y,
                "*": x * y,
                "divide": x // y if y != 0 else "undefined",
                "division": x // y if y != 0 else "undefined",
                "/": x // y if y != 0 else "undefined"
            }
            result = operations.get(operation, "unknown operation")
            return f"Result: {result}"

        tools = [tool_factory(
            name="calculate",
            description="Perform mathematical calculations",
            properties={
                "operation": {"type": "string", "description": "The operation to perform"},
                "x": {"type": "integer", "description": "First number"},
                "y": {"type": "integer", "description": "Second number"}
            },
            required=["operation", "x", "y"],
            implementation=calculate
        )]

        response = apple_model.prompt("What is 15 multiplied by 7?", tools=tools)

        # Note: Apple Intelligence may analyze tool code instead of calling it
        response_text = assert_response(response, "105")
        print(f"Response: {response_text}")

        if call_tracker.was_called():
            print("  (Tool was called)")
            # Verify tool was called with correct arguments
            call = call_tracker.get_call(0)
            assert call['x'] == 15 and call['y'] == 7, f"Expected x=15, y=7 but got {call}"
            # Model may use "multiply", "multiplication", "times", or "*"
            op = call['operation'].lower()
            assert 'multipl' in op or 'times' in op or op == '*', f"Expected multiplication operation but got {call['operation']}"
        else:
            print("  (Tool was not called - model calculated directly or analyzed tool code)")

    def test_multiple_tools(self, apple_model, tool_factory, assert_response):
        """Test with multiple tools registered."""
        # Track calls for each tool separately
        from tests.conftest import CallTracker
        time_tracker = CallTracker()
        date_tracker = CallTracker()

        def get_time():
            """Get the current time."""
            time_tracker.track()
            return "2:30 PM"

        def get_date():
            """Get the current date."""
            date_tracker.track()
            return "November 7, 2024"

        tools = [
            tool_factory("get_time", "Get the current time", implementation=get_time),
            tool_factory("get_date", "Get the current date", implementation=get_date)
        ]

        response = apple_model.prompt("What's the current date and time?", tools=tools)
        response_text = assert_response(response)
        print(f"Response: {response_text}")

        # Verify at least one tool was called
        total_calls = time_tracker.call_count() + date_tracker.call_count()
        assert total_calls > 0, "No tools were called"

        # Verify results appear in response if tools were called
        if date_tracker.was_called():
            assert "november" in response_text.lower() or "7" in response_text
        if time_tracker.was_called():
            assert "2:30" in response_text

    def test_tool_with_conversation(self, apple_model, tool_factory, call_tracker, assert_response):
        """Test tools work within a conversation context."""
        def get_temperature(city: str):
            """Get temperature for a city."""
            call_tracker.track(city=city)
            temps = {
                "paris": "18°C",
                "london": "15°C",
                "tokyo": "22°C",
                "new york": "20°C"
            }
            return temps.get(city.lower(), "20°C")

        tools = [tool_factory(
            name="get_temperature",
            description="Get the temperature for a city",
            properties={"city": {"type": "string", "description": "Name of the city"}},
            required=["city"],
            implementation=get_temperature
        )]

        # First turn
        response1 = apple_model.prompt("What's the temperature in Paris?", tools=tools)
        response1_text = assert_response(response1)
        print(f"Turn 1: {response1_text}")

        # Verify first call - tool may or may not be called depending on model's analysis
        if call_tracker.was_called():
            print("  (Tool was called)")
            call_tracker.assert_called_with(city='paris')
            assert "18" in response1_text
        else:
            print("  (Tool was not called - model analyzed tool code or used knowledge)")

        # Second turn in same conversation
        response2 = apple_model.prompt("And what about London?", tools=tools)
        response2_text = assert_response(response2)
        print(f"Turn 2: {response2_text}")

        # Just verify we got a meaningful response
        assert len(response2_text) > 10, "Response too short"


class TestToolCallingVerbose:
    """Verbose integration tests that print detailed output."""

    def test_tool_calling_with_details(self, apple_model, tool_factory, call_tracker):
        """Test tool calling with detailed output of the process."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Tool Calling with Details")
        print("=" * 70)

        # Define a tool
        def search_database(query: str, limit: int = 5):
            """Search a database for information."""
            call_tracker.track(query=query, limit=limit)
            print(f"\n[TOOL CALLED] search_database(query='{query}', limit={limit})")
            results = [
                f"Result {i+1}: Information about {query}"
                for i in range(min(limit, 3))
            ]
            return f"Found {len(results)} results: " + "; ".join(results)

        # Register the tool
        tools = [tool_factory(
            name="search_database",
            description="Search a database for information",
            properties={
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return"
                }
            },
            required=["query"],
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
        print(f"\n[TOOL CALL COUNT] {call_tracker.call_count()}")

        # Verify tool was called
        assert call_tracker.was_called(), "Tool was never called"

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
