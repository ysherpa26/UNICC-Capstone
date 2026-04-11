"""
test_config.py — Quick validation for config.py
=================================================
Run this from the same folder as config.py.

Usage:
    python test_config.py

What it tests:
    1. Mock mode works (no API key needed)
    2. call_llm() returns a string, not None
    3. get_provider() and get_default_model() return expected values
    4. call_llm() fails clearly when no key is set

Each test prints PASS or FAIL. All 4 should say PASS.
No real API calls are made — everything uses MOCK_MODE.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Setup: Force MOCK_MODE so we don't need real API keys
# ---------------------------------------------------------------------------
os.environ["MOCK_MODE"] = "1"
# Clear any real keys so they don't interfere
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("GROQ_API_KEY", None)

# Force fresh import (in case config was already imported)
if "config" in sys.modules:
    del sys.modules["config"]

passed = 0
failed = 0

# ---------------------------------------------------------------------------
# Test 1: Mock mode — call_llm returns a string
# ---------------------------------------------------------------------------
print("\n--- Test 1: Mock mode returns a string ---")
try:
    from config import call_llm
    result = call_llm(
        system_prompt="You are a safety expert.",
        user_message="Evaluate this AI agent."
    )
    if isinstance(result, str) and "[MOCK]" in result:
        print(f"  PASS — got: {result[:80]}...")
        passed += 1
    else:
        print(f"  FAIL — expected string with [MOCK], got: {type(result)} = {result}")
        failed += 1
except Exception as e:
    print(f"  FAIL — raised: {e}")
    failed += 1

# ---------------------------------------------------------------------------
# Test 2: Mock mode — model param is accepted but ignored
# ---------------------------------------------------------------------------
print("\n--- Test 2: Model override accepted in mock mode ---")
try:
    result = call_llm(
        system_prompt="Test system prompt",
        user_message="Test user message",
        model="mixtral-8x7b-32768"
    )
    if isinstance(result, str) and "[MOCK]" in result:
        print(f"  PASS — model param accepted, got mock response")
        passed += 1
    else:
        print(f"  FAIL — unexpected result: {result}")
        failed += 1
except Exception as e:
    print(f"  FAIL — raised: {e}")
    failed += 1

# ---------------------------------------------------------------------------
# Test 3: get_provider() and get_default_model() exist and return strings
# ---------------------------------------------------------------------------
print("\n--- Test 3: Helper functions return values ---")
try:
    from config import get_provider, get_default_model
    provider = get_provider()
    model = get_default_model()
    if provider == "mock" and isinstance(model, str):
        print(f"  PASS — provider='{provider}', model='{model}'")
        passed += 1
    else:
        print(f"  FAIL — provider='{provider}' (expected 'mock'), model='{model}'")
        failed += 1
except ImportError as e:
    print(f"  FAIL — function not found: {e}")
    failed += 1
except Exception as e:
    print(f"  FAIL — raised: {e}")
    failed += 1

# ---------------------------------------------------------------------------
# Test 4: No key + no mock = call_llm raises RuntimeError
# ---------------------------------------------------------------------------
print("\n--- Test 4: No API key and no mock mode raises error ---")
try:
    # Remove mock mode and reload config fresh
    os.environ.pop("MOCK_MODE", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("GROQ_API_KEY", None)

    # Force reimport
    if "config" in sys.modules:
        del sys.modules["config"]

    import importlib
    import config as cfg2
    importlib.reload(cfg2)

    # Now try to call — should raise RuntimeError
    cfg2.call_llm("test", "test")
    print(f"  FAIL — no error was raised (should have raised RuntimeError)")
    failed += 1

except RuntimeError as e:
    error_msg = str(e)
    if "No API key" in error_msg or "cannot make LLM calls" in error_msg:
        print(f"  PASS — got RuntimeError: {error_msg[:80]}...")
        passed += 1
    else:
        print(f"  FAIL — RuntimeError but wrong message: {error_msg}")
        failed += 1
except Exception as e:
    print(f"  FAIL — raised {type(e).__name__} instead of RuntimeError: {e}")
    failed += 1

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED — check output above")
print(f"{'='*50}\n")

sys.exit(0 if failed == 0 else 1)