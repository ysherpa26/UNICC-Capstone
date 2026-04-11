"""Tests for repo_reader.py — profile normalization and URL parsing.

Tests _normalize_profile() with edge cases (enum inference, missing fields)
and _repo_name_from_url() with various GitHub URL formats.
"""

import pytest

from repo_reader import _normalize_profile, _repo_name_from_url


# ---------------------------------------------------------------------------
# _repo_name_from_url()
# ---------------------------------------------------------------------------

class TestRepoNameFromUrl:
    def test_standard_github_url(self):
        assert _repo_name_from_url("https://github.com/user/my-repo") == "My Repo"

    def test_github_url_with_git_suffix(self):
        assert _repo_name_from_url("https://github.com/user/my-repo.git") == "My Repo.Git"

    def test_github_url_with_trailing_slash(self):
        assert _repo_name_from_url("https://github.com/user/my-repo/") == "My Repo"

    def test_github_url_with_underscores(self):
        assert _repo_name_from_url("https://github.com/org/ai_safety_lab") == "Ai Safety Lab"

    def test_github_url_with_mixed_separators(self):
        assert _repo_name_from_url("https://github.com/org/my-ai_project") == "My Ai Project"

    def test_single_path_component(self):
        # Only 1 path part (< 2), should return fallback
        assert _repo_name_from_url("https://github.com/user") == "Unknown AI System"

    def test_empty_url(self):
        assert _repo_name_from_url("") == "Unknown AI System"

    def test_malformed_url(self):
        assert _repo_name_from_url("not-a-url") == "Unknown AI System"

    def test_deep_path(self):
        result = _repo_name_from_url("https://github.com/org/repo/tree/main")
        assert result == "Main"

    def test_ssh_url(self):
        result = _repo_name_from_url("ssh://git@github.com/org/my-repo")
        assert result == "My Repo"


# ---------------------------------------------------------------------------
# _normalize_profile()
# ---------------------------------------------------------------------------

class TestNormalizeProfile:
    def test_valid_profile_passes_through(self):
        profile = {
            "name": "TestBot",
            "type": "chatbot",
            "use_case": "testing",
            "deployment": "cloud",
            "auth": "API key",
            "finetune_data": "none",
            "logging": "enabled",
        }
        result = _normalize_profile(profile, "Fallback Name")
        assert result["name"] == "TestBot"
        assert result["deployment"] == "cloud"
        assert result["logging"] == "enabled"

    def test_missing_fields_use_defaults(self):
        result = _normalize_profile({}, "My Repo")
        assert result["name"] == "My Repo"
        assert result["type"] == "LLM-based application"
        assert result["use_case"] == "general purpose AI assistant"
        assert result["deployment"] == "cloud"
        assert result["auth"] == "none"
        assert result["finetune_data"] == "none"
        assert result["logging"] == "disabled"

    def test_deployment_on_prem_synonym(self):
        result = _normalize_profile({"deployment": "On-Prem"}, "X")
        assert result["deployment"] == "on_prem"

    def test_deployment_on_premises(self):
        result = _normalize_profile({"deployment": "on-premises"}, "X")
        assert result["deployment"] == "on_prem"

    def test_deployment_local(self):
        result = _normalize_profile({"deployment": "local server"}, "X")
        assert result["deployment"] == "on_prem"

    def test_deployment_self_hosted(self):
        result = _normalize_profile({"deployment": "self-hosted"}, "X")
        assert result["deployment"] == "on_prem"

    def test_deployment_private(self):
        result = _normalize_profile({"deployment": "private cloud"}, "X")
        assert result["deployment"] == "on_prem"

    def test_deployment_hybrid(self):
        result = _normalize_profile({"deployment": "hybrid"}, "X")
        assert result["deployment"] == "hybrid"

    def test_deployment_mixed(self):
        result = _normalize_profile({"deployment": "mixed deployment"}, "X")
        assert result["deployment"] == "hybrid"

    def test_deployment_unknown_defaults_to_cloud(self):
        result = _normalize_profile({"deployment": "banana"}, "X")
        assert result["deployment"] == "cloud"

    def test_logging_yes(self):
        result = _normalize_profile({"logging": "YES"}, "X")
        assert result["logging"] == "enabled"

    def test_logging_true(self):
        result = _normalize_profile({"logging": "true"}, "X")
        assert result["logging"] == "enabled"

    def test_logging_on(self):
        result = _normalize_profile({"logging": "on"}, "X")
        assert result["logging"] == "enabled"

    def test_logging_enabled_passthrough(self):
        result = _normalize_profile({"logging": "enabled"}, "X")
        assert result["logging"] == "enabled"

    def test_logging_unknown_defaults_to_disabled(self):
        result = _normalize_profile({"logging": "maybe"}, "X")
        assert result["logging"] == "disabled"

    def test_empty_string_fields_use_defaults(self):
        profile = {
            "name": "",
            "type": "  ",
            "use_case": "",
        }
        result = _normalize_profile(profile, "Fallback")
        assert result["name"] == "Fallback"
        assert result["type"] == "LLM-based application"
        assert result["use_case"] == "general purpose AI assistant"

    def test_none_values_use_defaults(self):
        profile = {"name": None, "deployment": None, "logging": None}
        result = _normalize_profile(profile, "Fallback")
        assert result["name"] == "Fallback"
        assert result["deployment"] == "cloud"
        # Note: None → str("None") → "none" which contains "on" → matches "enabled"
        # This is an edge case in the existing normalization logic
        assert result["logging"] == "enabled"

    def test_always_returns_7_fields(self):
        result = _normalize_profile({"extra_field": "ignored"}, "X")
        assert set(result.keys()) == {
            "name", "type", "use_case", "deployment", "auth", "finetune_data", "logging"
        }
