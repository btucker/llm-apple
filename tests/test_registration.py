"""Tests for model registration."""
import pytest
from unittest.mock import Mock
import llm_apple


def test_register_models():
    """Test that register_models function exists and is callable."""
    assert hasattr(llm_apple, 'register_models')
    assert callable(llm_apple.register_models)


def test_register_models_calls_register():
    """Test that register_models calls the register function with AppleModel."""
    mock_register = Mock()
    llm_apple.register_models(mock_register)

    # Verify register was called once
    assert mock_register.call_count == 1

    # Verify it was called with an AppleModel instance
    call_args = mock_register.call_args[0]
    assert len(call_args) == 1
    assert isinstance(call_args[0], llm_apple.AppleModel)


def test_apple_model_has_correct_attributes():
    """Test that AppleModel has the expected attributes."""
    model = llm_apple.AppleModel()

    assert model.model_id == "apple"
    assert model.can_stream is True
    assert hasattr(model, 'Options')


def test_apple_model_options():
    """Test that AppleModel.Options has correct fields."""
    from pydantic import ValidationError

    # Valid options
    options = llm_apple.AppleModel.Options(
        temperature=1.0,
        max_tokens=500
    )
    assert options.temperature == 1.0
    assert options.max_tokens == 500

    # Default values
    options_default = llm_apple.AppleModel.Options()
    assert options_default.temperature == 1.0
    assert options_default.max_tokens == 1024


def test_apple_model_options_validation():
    """Test that AppleModel.Options validates constraints."""
    from pydantic import ValidationError

    # Temperature too low
    with pytest.raises(ValidationError):
        llm_apple.AppleModel.Options(temperature=-0.1)

    # Temperature too high
    with pytest.raises(ValidationError):
        llm_apple.AppleModel.Options(temperature=2.1)

    # Max tokens must be positive
    with pytest.raises(ValidationError):
        llm_apple.AppleModel.Options(max_tokens=0)

    with pytest.raises(ValidationError):
        llm_apple.AppleModel.Options(max_tokens=-1)
