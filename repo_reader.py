"""Clones GitHub repo, extracts 7 model_profile fields via LLM"""


def extract_model_profile(github_url: str) -> dict:
    """TEMPORARY STUB — returns hardcoded profile for testing server.py routes.
    Replace with real implementation that clones the repo and calls call_llm().
    """
    return {
        "name": "StubBot",
        "type": "chatbot",
        "use_case": "testing",
        "deployment": "cloud",
        "auth": "none",
        "finetune_data": "none",
        "logging": "disabled",
    }
