"""Tests for Apple Intelligence availability checking."""
import pytest
from unittest.mock import Mock, patch
import sys
import llm_apple


def test_availability_check_when_available():
    """Test availability check when Apple Intelligence is available."""
    # Create mock module
    mock_module = Mock()
    mock_module.Availability = Mock()
    mock_module.Availability.AVAILABLE = 1
    mock_module.Client = Mock()
    mock_module.Client.check_availability = Mock(return_value=1)
    mock_module.Client.get_availability_reason = Mock(return_value=None)

    sys.modules['applefoundationmodels'] = mock_module

    try:
        model = llm_apple.AppleModel()
        # Should not raise
        model._check_availability()
        assert model._availability_checked is True
    finally:
        del sys.modules['applefoundationmodels']


def test_availability_check_when_unavailable():
    """Test availability check when Apple Intelligence is unavailable."""
    # Create mock module
    mock_module = Mock()
    mock_module.Availability = Mock()
    mock_module.Availability.AVAILABLE = 1
    mock_module.Availability.UNAVAILABLE = 0
    mock_module.Client = Mock()
    mock_module.Client.check_availability = Mock(return_value=0)  # Unavailable
    mock_module.Client.get_availability_reason = Mock(
        return_value="Device not supported"
    )

    sys.modules['applefoundationmodels'] = mock_module

    try:
        model = llm_apple.AppleModel()

        with pytest.raises(RuntimeError) as exc_info:
            model._check_availability()

        assert "Apple Intelligence not available" in str(exc_info.value)
        assert "Device not supported" in str(exc_info.value)
    finally:
        del sys.modules['applefoundationmodels']


def test_availability_check_unavailable_no_reason():
    """Test availability check when unavailable with no specific reason."""
    # Create mock module
    mock_module = Mock()
    mock_module.Availability = Mock()
    mock_module.Availability.AVAILABLE = 1
    mock_module.Client = Mock()
    mock_module.Client.check_availability = Mock(return_value=0)
    mock_module.Client.get_availability_reason = Mock(return_value=None)

    sys.modules['applefoundationmodels'] = mock_module

    try:
        model = llm_apple.AppleModel()

        with pytest.raises(RuntimeError) as exc_info:
            model._check_availability()

        assert "Apple Intelligence not available" in str(exc_info.value)
        assert "Unknown reason" in str(exc_info.value)
    finally:
        del sys.modules['applefoundationmodels']


def test_get_client_propagates_availability_error():
    """Test that _get_client propagates availability errors."""
    # Create mock module
    mock_module = Mock()
    mock_module.Availability = Mock()
    mock_module.Availability.AVAILABLE = 1
    mock_module.Client = Mock()
    mock_module.Client.check_availability = Mock(return_value=0)
    mock_module.Client.get_availability_reason = Mock(
        return_value="Not enabled"
    )

    sys.modules['applefoundationmodels'] = mock_module

    try:
        model = llm_apple.AppleModel()

        with pytest.raises(RuntimeError) as exc_info:
            model._get_client()

        assert "Apple Intelligence not available" in str(exc_info.value)
        assert "Not enabled" in str(exc_info.value)
    finally:
        del sys.modules['applefoundationmodels']


def test_availability_lazy_check():
    """Test that availability is checked lazily, not on initialization."""
    # Create mock module
    mock_module = Mock()
    mock_module.Availability = Mock()
    mock_module.Availability.AVAILABLE = 1
    mock_module.Client = Mock()
    mock_module.Client.check_availability = Mock(return_value=1)

    sys.modules['applefoundationmodels'] = mock_module

    try:
        # Creating model should not check availability
        model = llm_apple.AppleModel()
        assert mock_module.Client.check_availability.call_count == 0
        assert model._availability_checked is False

        # Only when we call _check_availability or _get_client
        model._check_availability()
        assert mock_module.Client.check_availability.call_count == 1
        assert model._availability_checked is True
    finally:
        del sys.modules['applefoundationmodels']
