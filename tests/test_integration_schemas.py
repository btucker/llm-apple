"""
Integration tests for schema (structured output) support with llm-apple.

These tests run against the actual Apple Foundation Models API when available.
They will be skipped if Apple Intelligence is not available on the system.
"""

import pytest
import llm
import json


def is_apple_intelligence_available():
    """Check if Apple Intelligence is available on this system."""
    try:
        from applefoundationmodels import Session, Availability

        status = Session.check_availability()
        return status == Availability.AVAILABLE
    except (ImportError, Exception):
        return False


pytestmark = pytest.mark.skipif(
    not is_apple_intelligence_available(),
    reason="Apple Intelligence not available on this system",
)


@pytest.fixture
def apple_model():
    """Get the apple model from llm."""
    return llm.get_model("apple")


class TestSchemaIntegration:
    """Integration tests for schema/structured output functionality."""

    def test_simple_schema_person_extraction(self, apple_model):
        """Test extracting person information with a schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"},
            },
            "required": ["name", "age", "city"],
        }

        response = apple_model.prompt(
            "Extract person info: Alice is 28 years old and lives in Paris",
            schema=schema,
            stream=False,
        )

        result = response.text()
        assert result, "Response should not be empty"

        # Parse the JSON response
        data = json.loads(result)
        assert "name" in data
        assert "age" in data
        assert "city" in data

        # Verify the extracted values are reasonable
        assert isinstance(data["name"], str)
        assert isinstance(data["age"], int)
        assert isinstance(data["city"], str)

        # Check that values are roughly correct (LLM might vary slightly)
        assert "alice" in data["name"].lower()
        assert data["age"] == 28
        assert "paris" in data["city"].lower()

    def test_schema_with_multiple_types(self, apple_model):
        """Test schema with various data types."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "year": {"type": "integer"},
                "rating": {"type": "number"},
                "available": {"type": "boolean"},
            },
            "required": ["title", "year", "rating", "available"],
        }

        response = apple_model.prompt(
            "The movie 'Inception' was released in 2010, has a rating of 8.8, and is currently available",
            schema=schema,
            stream=False,
        )

        result = response.text()
        data = json.loads(result)

        assert isinstance(data["title"], str)
        assert isinstance(data["year"], int)
        assert isinstance(data["rating"], (int, float))
        assert isinstance(data["available"], bool)

        assert "inception" in data["title"].lower()
        assert data["year"] == 2010
        assert data["rating"] > 8.0
        assert data["available"] is True

    def test_schema_with_nested_objects(self, apple_model):
        """Test schema with nested object structure."""
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
                "location": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "country": {"type": "string"},
                    },
                    "required": ["city", "country"],
                },
            },
            "required": ["person", "location"],
        }

        response = apple_model.prompt(
            "Bob is 35 years old and lives in London, United Kingdom",
            schema=schema,
            stream=False,
        )

        result = response.text()
        data = json.loads(result)

        assert "person" in data
        assert "location" in data
        assert "name" in data["person"]
        assert "age" in data["person"]
        assert "city" in data["location"]
        assert "country" in data["location"]

    def test_schema_extraction_from_longer_text(self, apple_model):
        """Test extracting structured data from longer narrative text."""
        schema = {
            "type": "object",
            "properties": {
                "product": {"type": "string"},
                "price": {"type": "number"},
                "inStock": {"type": "boolean"},
            },
            "required": ["product", "price", "inStock"],
        }

        response = apple_model.prompt(
            "I'm looking for the new MacBook Pro. The 14-inch model costs $1999 "
            "and it's currently in stock at our store. Would you like to purchase it?",
            schema=schema,
            stream=False,
        )

        result = response.text()
        data = json.loads(result)

        assert "product" in data
        assert "price" in data
        assert "inStock" in data
        assert isinstance(data["price"], (int, float))
        assert isinstance(data["inStock"], bool)

    def test_schema_automatically_disables_streaming(self, apple_model):
        """Test that schema automatically disables streaming even when requested."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        # Request streaming, but schema should disable it
        response = apple_model.prompt(
            "Alice is a developer", schema=schema, stream=True
        )

        # Response should be non-streaming (get text directly)
        result = response.text()
        assert result, "Response should not be empty"

        # Should be valid JSON
        data = json.loads(result)
        assert "name" in data


class TestAsyncSchemaIntegration:
    """Integration tests for async schema functionality."""

    @pytest.mark.asyncio
    async def test_async_schema_person_extraction(self):
        """Test async model with schema."""
        model = llm.get_async_model("apple")

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "occupation": {"type": "string"},
            },
            "required": ["name", "occupation"],
        }

        response = model.prompt(
            "Dr. Smith is a neuroscientist at Stanford University",
            schema=schema,
            stream=False,
        )

        result_text = ""
        async for chunk in response:
            result_text += chunk

        assert result_text, "Response should not be empty"

        data = json.loads(result_text)
        assert "name" in data
        assert "occupation" in data

    @pytest.mark.asyncio
    async def test_async_schema_automatically_disables_streaming(self):
        """Test that async schema automatically disables streaming."""
        model = llm.get_async_model("apple")

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        # Request streaming, but schema should disable it
        response = model.prompt("Alice is a developer", schema=schema, stream=True)

        # Should get single result (not streaming)
        result_text = ""
        async for chunk in response:
            result_text += chunk

        assert result_text, "Response should not be empty"

        # Should be valid JSON
        data = json.loads(result_text)
        assert "name" in data
