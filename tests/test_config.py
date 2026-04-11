"""
test_config.py — Validation for config.py
==========================================
Pytest-compatible test module. Each test is isolated and uses fixtures
to manage environment state. No real API calls are made — everything
uses MOCK_MODE.

Run with:
    pytest tests/test_config.py -v

Or as a standalone script:
    python tests/test_config.py
"""

import importlib
import os
import sys

import pytest


@pytest.fixture
def mock_mode_env(monkeypatch):
    """Force MOCK_MODE and clear real API keys, then reload config."""
    monkeypatch.setenv("MOCK_MODE", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    if "config" in sys.modules:
        del sys.modules["config"]
    import config
    importlib.reload(config)
    return config


@pytest.fixture
def no_key_env(monkeypatch):
    """Clear all keys, prevent .env loading, then reload config."""
    monkeypatch.delenv("MOCK_MODE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Prevent python-dotenv from re-reading .env on reimport
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    if "config" in sys.modules:
        del sys.modules["config"]
    import config
    importlib.reload(config)
    return config


def test_mock_mode_returns_string(mock_mode_env):
    """call_llm() in mock mode returns a string containing [MOCK]."""
    result = mock_mode_env.call_llm(
        system_prompt="You are a safety expert.",
        user_message="Evaluate this AI agent.",
    )
    assert isinstance(result, str), f"expected str, got {type(result)}"
    assert "[MOCK]" in result, f"expected [MOCK] marker, got: {result}"


def test_model_override_accepted_in_mock_mode(mock_mode_env):
    """The optional model parameter is accepted (and ignored) in mock mode."""
    result = mock_mode_env.call_llm(
        system_prompt="Test system prompt",
        user_message="Test user message",
        model="mixtral-8x7b-32768",
    )
    assert isinstance(result, str)
    assert "[MOCK]" in result


def test_helper_functions_return_values(mock_mode_env):
    """get_provider() and get_default_model() return expected values in mock mode."""
    provider = mock_mode_env.get_provider()
    model = mock_mode_env.get_default_model()
    assert provider == "mock", f"expected 'mock', got '{provider}'"
    assert isinstance(model, str), f"expected str, got {type(model)}"


def test_no_key_raises_runtime_error(no_key_env):
    """call_llm() raises RuntimeError when no API key is set and mock mode is off."""
    with pytest.raises(RuntimeError) as exc_info:
        no_key_env.call_llm("test", "test")
    error_msg = str(exc_info.value)
    assert "No API key" in error_msg or "cannot make LLM calls" in error_msg, (
        f"unexpected error message: {error_msg}"
    )


# ---------------------------------------------------------------------------
# Standalone script fallback (so `python tests/test_config.py` still works)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))