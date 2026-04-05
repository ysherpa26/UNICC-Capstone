"""Clones GitHub repo, extracts 7 model_profile fields via LLM"""

"""
repo_reader.py — P2 Council Orchestration
UNICC AI Safety Lab — Capstone Spring 2026

Clones a GitHub repo, reads priority files, and extracts a 7-field
model_profile dict via LLM call.

Public function:
    extract_model_profile(github_url: str) -> dict

This is one of the two functions P3's server.py imports from P2.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from config import call_llm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Files to read from the cloned repo, in priority order
PRIORITY_FILES = [
    "README.md",
    "readme.md",
    "README.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "requirements.txt",
    "main.py",
    "app.py",
    "config.py",
    "config.yaml",
    "config.yml",
    "config.json",
    ".env.example",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
]

# Max characters to read per file (avoid blowing up the LLM context)
MAX_CHARS_PER_FILE = 3000

# Max total characters sent to LLM
MAX_TOTAL_CHARS = 12000

# Valid enum values from the P2/P3 contract schema
VALID_DEPLOYMENT = {"cloud", "on_prem", "hybrid"}
VALID_LOGGING = {"enabled", "disabled"}

# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are an AI model profile extraction agent.

You will receive the contents of files from a GitHub repository. Your job is to extract \
a structured model_profile with exactly 7 fields.

You must output VALID JSON ONLY - no preamble, no explanation, no markdown fences.

Output schema (exactly these 7 fields, no more, no less):
{
  "name": "string — the name of the AI model or system",
  "type": "string — the type of AI system (e.g. LLM, classifier, recommender, chatbot, agent)",
  "use_case": "string — what the system is designed to do",
  "deployment": "cloud | on_prem | hybrid",
  "auth": "string — authentication method (e.g. API key, OAuth, none, JWT)",
  "finetune_data": "string — description of fine-tuning data if any, or 'none' if not fine-tuned",
  "logging": "enabled | disabled"
}

Rules:
- deployment MUST be exactly one of: cloud, on_prem, hybrid
- logging MUST be exactly one of: enabled, disabled
- If a field cannot be determined from the repo contents, make a reasonable inference \
  based on context clues (tech stack, README, config files)
- If truly no information is available for a field, use a safe default:
    name: derived from repo name
    type: "LLM-based application"
    use_case: "general purpose AI assistant"
    deployment: "cloud"
    auth: "none"
    finetune_data: "none"
    logging: "disabled"
- Be concise — each field value should be one sentence or less
"""


# ---------------------------------------------------------------------------
# Git clone helper
# ---------------------------------------------------------------------------

def _clone_repo(github_url: str, target_dir: str) -> bool:
    """
    Shallow-clones the repo into target_dir.
    Returns True on success, False on failure.
    """
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", github_url, target_dir],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _repo_name_from_url(github_url: str) -> str:
    """Extracts repo name from GitHub URL for use as fallback model name."""
    try:
        path = urlparse(github_url).path  # e.g. /ysherpa26/UNICC-Capstone
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2:
            return parts[-1].replace("-", " ").replace("_", " ").title()
        return "Unknown AI System"
    except Exception:
        return "Unknown AI System"


# ---------------------------------------------------------------------------
# File reader
# ---------------------------------------------------------------------------

def _read_repo_files(repo_dir: str) -> str:
    """
    Reads priority files from the cloned repo.
    Returns concatenated content string up to MAX_TOTAL_CHARS.
    """
    repo_path = Path(repo_dir)
    collected = []
    total_chars = 0

    for filename in PRIORITY_FILES:
        file_path = repo_path / filename
        if not file_path.exists():
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            # Truncate per-file
            if len(content) > MAX_CHARS_PER_FILE:
                content = content[:MAX_CHARS_PER_FILE] + "\n[... truncated ...]"

            section = f"=== {filename} ===\n{content}\n"
            if total_chars + len(section) > MAX_TOTAL_CHARS:
                # Add partial content to fill up to the limit
                remaining = MAX_TOTAL_CHARS - total_chars
                if remaining > 100:
                    collected.append(section[:remaining] + "\n[... truncated ...]")
                break

            collected.append(section)
            total_chars += len(section)

        except (OSError, PermissionError):
            continue

    # If no priority files found, try to read any .md or .txt files
    if not collected:
        for ext in ("*.md", "*.txt", "*.py"):
            for file_path in list(repo_path.glob(ext))[:3]:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    content = content[:MAX_CHARS_PER_FILE]
                    section = f"=== {file_path.name} ===\n{content}\n"
                    collected.append(section)
                    total_chars += len(section)
                    if total_chars >= MAX_TOTAL_CHARS:
                        break
                except (OSError, PermissionError):
                    continue
            if collected:
                break

    return "\n".join(collected) if collected else ""


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def _extract_via_llm(repo_content: str, repo_name: str) -> dict:
    """
    Calls LLM to extract model_profile from repo content.
    Returns parsed dict or raises ValueError on parse failure.
    """
    user_message = (
        f"Extract the model_profile from this GitHub repository (repo name: '{repo_name}').\n\n"
        f"Repository contents:\n\n{repo_content}"
    )

    raw = call_llm(EXTRACTION_SYSTEM_PROMPT, user_message)

    # Strip markdown fences if LLM added them despite instructions
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    result = json.loads(cleaned)

    if not isinstance(result, dict):
        raise ValueError("LLM returned non-dict JSON")

    return result


# ---------------------------------------------------------------------------
# Validation and normalization
# ---------------------------------------------------------------------------

def _normalize_profile(profile: dict, repo_name: str) -> dict:
    """
    Enforces schema constraints and fills missing fields with safe defaults.
    Always returns a complete 7-field dict.
    """
    def clean_str(val, default):
        if isinstance(val, str) and val.strip():
            return val.strip()
        return default

    name = clean_str(profile.get("name"), repo_name)
    model_type = clean_str(profile.get("type"), "LLM-based application")
    use_case = clean_str(profile.get("use_case"), "general purpose AI assistant")
    auth = clean_str(profile.get("auth"), "none")
    finetune_data = clean_str(profile.get("finetune_data"), "none")

    # Enforce enum: deployment
    deployment = str(profile.get("deployment", "cloud")).strip().lower()
    if deployment not in VALID_DEPLOYMENT:
        # Try to infer from common synonyms
        if any(kw in deployment for kw in ("on-prem", "on_prem", "local", "self-host", "private")):
            deployment = "on_prem"
        elif any(kw in deployment for kw in ("hybrid", "mixed")):
            deployment = "hybrid"
        else:
            deployment = "cloud"

    # Enforce enum: logging
    logging_val = str(profile.get("logging", "disabled")).strip().lower()
    if logging_val not in VALID_LOGGING:
        if any(kw in logging_val for kw in ("true", "yes", "on", "enable")):
            logging_val = "enabled"
        else:
            logging_val = "disabled"

    return {
        "name": name,
        "type": model_type,
        "use_case": use_case,
        "deployment": deployment,
        "auth": auth,
        "finetune_data": finetune_data,
        "logging": logging_val,
    }


def _safe_defaults(repo_name: str) -> dict:
    """Returns a fully populated model_profile using safe defaults."""
    return {
        "name": repo_name,
        "type": "LLM-based application",
        "use_case": "general purpose AI assistant",
        "deployment": "cloud",
        "auth": "none",
        "finetune_data": "none",
        "logging": "disabled",
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_model_profile(github_url: str) -> dict:
    """
    Clones a GitHub repo, reads priority files, and extracts a 7-field
    model_profile dict via LLM.

    This is one of the two functions P3's server.py imports from P2.

    Args:
        github_url: Full GitHub repo URL, e.g. https://github.com/user/repo

    Returns:
        7-field model_profile dict:
        {name, type, use_case, deployment, auth, finetune_data, logging}
        Always returns a complete dict — never raises, always falls back to defaults.
    """
    repo_name = _repo_name_from_url(github_url)
    tmp_dir = tempfile.mkdtemp(prefix="p2_repo_")

    try:
        # Step 1 — Clone
        clone_success = _clone_repo(github_url, tmp_dir)

        if not clone_success:
            # Can't clone — return safe defaults
            return _safe_defaults(repo_name)

        # Step 2 — Read priority files
        repo_content = _read_repo_files(tmp_dir)

        if not repo_content:
            # Repo cloned but no readable files — return safe defaults
            return _safe_defaults(repo_name)

        # Step 3 — LLM extraction
        try:
            raw_profile = _extract_via_llm(repo_content, repo_name)
        except (json.JSONDecodeError, ValueError):
            # LLM parse failed — return safe defaults
            return _safe_defaults(repo_name)

        # Step 4 — Normalize and validate
        return _normalize_profile(raw_profile, repo_name)

    except Exception:
        # Catch-all — always return something usable
        return _safe_defaults(repo_name)

    finally:
        # Step 5 — Always clean up temp directory
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
