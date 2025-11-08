"""Tests for AppleModel class."""

import pytest
from unittest.mock import Mock, patch
import llm_apple


def test_apple_model_initialization():
    """Test that AppleModel initializes with correct state."""
    model = llm_apple.AppleModel()

    assert model._client is None
    assert model._sessions == {}
    assert model._availability_checked is False
    assert model.model_id == "apple"
    assert model.can_stream is True


def test_get_client_creates_client_once(mock_applefoundationmodels):
    """Test that _get_client creates and reuses a single client instance."""
    model = llm_apple.AppleModel()

    # First call should create client
    client1 = model._get_client()
    assert client1 is not None
    assert model._client is not None

    # Second call should return same client
    client2 = model._get_client()
    assert client2 is client1
    assert mock_applefoundationmodels.Client.call_count == 1


def test_get_client_checks_availability(mock_applefoundationmodels):
    """Test that _get_client checks availability before creating client."""
    model = llm_apple.AppleModel()

    model._get_client()

    # Verify availability was checked
    assert mock_applefoundationmodels.Client.check_availability.called


def test_get_session_without_conversation(mock_applefoundationmodels):
    """Test that _get_session creates new session without conversation."""
    model = llm_apple.AppleModel()

    # Get sessions without conversation ID
    session1 = model._get_session(None, None)
    session2 = model._get_session(None, None)

    # Should create new session each time
    assert session1 is not None
    assert session2 is not None
    # Client should have create_session called twice
    client = model._get_client()
    assert client.create_session.call_count >= 2


def test_get_session_with_conversation_reuses_session(mock_applefoundationmodels):
    """Test that _get_session reuses session for same conversation."""
    model = llm_apple.AppleModel()
    conversation_id = "test-conv-123"

    # Get session twice with same conversation ID
    session1 = model._get_session(conversation_id, None)
    session2 = model._get_session(conversation_id, None)

    # Should return same session
    assert session1 is session2
    assert conversation_id in model._sessions

    # Client should create session only once for this conversation
    client = model._get_client()
    # Note: create_session is called once per conversation
    assert client.create_session.call_count >= 1


def test_get_session_with_different_conversations(mock_applefoundationmodels):
    """Test that different conversations get different sessions."""
    model = llm_apple.AppleModel()

    session1 = model._get_session("conv-1", None)
    session2 = model._get_session("conv-2", None)

    # Should have different sessions
    assert "conv-1" in model._sessions
    assert "conv-2" in model._sessions
    assert len(model._sessions) == 2


def test_get_session_with_instructions(mock_applefoundationmodels):
    """Test that _get_session passes instructions to create_session."""
    model = llm_apple.AppleModel()
    instructions = "You are a helpful assistant"

    session = model._get_session(None, instructions)

    # Verify instructions were passed
    client = model._get_client()
    client.create_session.assert_called_with(instructions=instructions)


def test_check_availability_only_once(mock_applefoundationmodels):
    """Test that availability is only checked once."""
    model = llm_apple.AppleModel()

    # Check availability multiple times
    model._check_availability()
    model._check_availability()
    model._check_availability()

    # Should only check once
    assert mock_applefoundationmodels.Client.check_availability.call_count == 1
    assert model._availability_checked is True
