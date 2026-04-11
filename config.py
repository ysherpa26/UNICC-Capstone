"""
config.py — LLM Gateway
========================
Single entry point for all LLM calls in the UNICC AI Safety Lab.

What this file does:
    1. Reads environment variables to detect which LLM provider to use
    2. Exposes call_llm() — the ONE function every other file uses for LLM calls
    3. Supports mock mode for testing without burning API tokens

What this file does NOT do:
    - No FastAPI routes (that's server.py)
    - No schema validation (that's schemas.py)
    - No pipeline logic (that's pipeline.py / agents.py / judge.py)

Usage by other files:
    from config import call_llm
    response = call_llm(system_prompt="...", user_message="...")
"""

import os
import time

# ---------------------------------------------------------------------------
# 1. Load .env file (optional convenience — works fine without it)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed — that's fine, we'll read os.environ directly
    pass

# ---------------------------------------------------------------------------
# 2. Read environment variables
# ---------------------------------------------------------------------------
# .strip() prevents silent failures from keys like "ANTHROPIC_API_KEY= " (trailing space)
_anthropic_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
_groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
_mock_mode = (os.getenv("MOCK_MODE") or "").strip().lower() in ("1", "true")

# ---------------------------------------------------------------------------
# 3. Detect provider and initialize SDK client
# ---------------------------------------------------------------------------
_provider = None      # "anthropic" | "groq" | "mock"
_client = None        # Anthropic() or Groq() instance
_default_model = None # default model string for the active provider
_init_error = None    # if set, call_llm() raises this instead of calling the API

if _mock_mode:
    _provider = "mock"
    print("[config] MOCK_MODE enabled — no live LLM calls will be made")

elif _anthropic_key:
    try:
        from anthropic import Anthropic
        _client = Anthropic(api_key=_anthropic_key)
        _provider = "anthropic"
        _default_model = "claude-sonnet-4-20250514"
        print(f"[config] Using Anthropic API (model: {_default_model})")
    except ImportError:
        _init_error = (
            "ANTHROPIC_API_KEY is set but the 'anthropic' package is not installed. "
            "Run: pip install anthropic"
        )
        print(f"[config] ERROR: {_init_error}")

    if _groq_key:
        # Both keys present — let the user know which one won
        print("[config] Note: Both ANTHROPIC_API_KEY and GROQ_API_KEY are set. Using Anthropic.")

elif _groq_key:
    try:
        from groq import Groq
        _client = Groq(api_key=_groq_key)
        _provider = "groq"
        _default_model = (os.getenv("AGENT_1_MODEL") or "openai/gpt-oss-20b").strip()
        print(f"[config] Using Groq API (default model: {_default_model})")
    except ImportError:
        _init_error = (
            "GROQ_API_KEY is set but the 'groq' package is not installed. "
            "Run: pip install groq"
        )
        print(f"[config] ERROR: {_init_error}")

else:
    _init_error = (
        "No API key found. Set one of these environment variables:\n"
        "  ANTHROPIC_API_KEY=your_key_here\n"
        "  GROQ_API_KEY=your_key_here\n"
        "Or set MOCK_MODE=1 to test without live LLM calls."
    )
    print(f"[config] ERROR: {_init_error}")


# ---------------------------------------------------------------------------
# 4. Helper functions — so other files can check what's active
# ---------------------------------------------------------------------------

def get_provider() -> str:
    """Returns 'anthropic', 'groq', or 'mock'. Used by judge.py to fill ensemble_meta."""
    return _provider or "none"


def get_default_model() -> str:
    """Returns the default model string. Used by judge.py to fill agent model fields."""
    return _default_model or "unknown"


# ---------------------------------------------------------------------------
# 5. Main function — the ONE function every other file calls
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_message: str, model: str = None) -> str:
    """
    Send a prompt to the configured LLM and return the text response.

    Args:
        system_prompt: Instructions for the LLM (e.g., "You are a risk assessment expert...")
        user_message:  The actual content to process (e.g., the model_profile JSON)
        model:         Optional model override.
                       - Groq: used for per-agent models ("mixtral-8x7b-32768", etc.)
                       - Anthropic: ignored — always uses Claude Sonnet
                       - Mock: ignored

    Returns:
        The LLM's text response as a string.

    Raises:
        RuntimeError: if no provider is configured or the API call fails.
    """

    # --- Check for init errors (no key, missing package, etc.) ---
    if _init_error:
        raise RuntimeError(f"config.py cannot make LLM calls: {_init_error}")

    # --- Mock mode: return a canned response without calling any API ---
    if _provider == "mock":
        return (
            f"[MOCK] This is a mock LLM response. "
            f"System prompt: {len(system_prompt)} chars. "
            f"User message: {len(user_message)} chars."
        )

    # --- Anthropic path ---
    if _provider == "anthropic":
        try:
            response = _client.messages.create(
                model=_default_model,  # always Claude Sonnet — ignores model param
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                timeout=120.0,  # 2 minutes — prevents hanging forever
            )

            # Validate that we actually got text back
            if not response.content or len(response.content) == 0:
                raise RuntimeError("Anthropic returned an empty response (no content blocks)")

            text = response.content[0].text
            if not text or not text.strip():
                raise RuntimeError("Anthropic returned an empty text response")

            return text

        except RuntimeError:
            raise  # re-raise our own validation errors as-is
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}") from e

    # --- Groq path ---
    if _provider == "groq":
        use_model = model or _default_model

        # One retry on transient failures (rate limits, brief outages)
        last_error = None
        for attempt in range(2):
            try:
                response = _client.chat.completions.create(
                    model=use_model,
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    timeout=120.0,
                )

                # Validate that we actually got text back
                if not response.choices or len(response.choices) == 0:
                    raise RuntimeError("Groq returned an empty response (no choices)")

                text = response.choices[0].message.content
                if not text or not text.strip():
                    raise RuntimeError("Groq returned an empty text response")

                return text

            except RuntimeError:
                raise  # re-raise our own validation errors as-is
            except Exception as e:
                last_error = e
                if attempt == 0:
                    # Wait 2 seconds before retry — handles rate limit (429) errors
                    time.sleep(2)

        raise RuntimeError(f"Groq API error after 2 attempts: {last_error}") from last_error

    # --- Should never reach here, but just in case ---
    raise RuntimeError(f"Unknown provider state: {_provider}")