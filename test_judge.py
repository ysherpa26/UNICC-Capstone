"""Tests for judge.py — majority vote, disagreement detection, confidence computation.

Covers all 5 vote scenarios:
  1. 3-way unanimous agreement
  2. 2/3 majority (majority wins)
  3. 2/3 majority with conservative escalation (minority hold_and_fix wins)
  4. 2 succeeded but disagree (1 failed)
  5. 3-way split (all different recommendations)

Also tests _build_disagreements() and _run_judge_llm() parsing.
"""

import pytest

from judge import (
    _majority_vote,
    _build_disagreements,
    _run_judge_llm,
    _compute_severity_spread,
    _dissenting_agent_avg_severity,
    CONFIDENCE_ALL_AGREE,
    CONFIDENCE_MAJORITY,
    CONFIDENCE_NO_MAJORITY,
)


def _make_agent_result(
    agent_id, recommendation, category_summary=None, error=False, lens="Test"
):
    """Helper to build a mock agent result dict."""
    if error:
        return {
            "error": True,
            "error_code": "UNKNOWN",
            "error_message": f"{agent_id} failed",
            "_agent_id": agent_id,
            "_agent_model": "test-model",
            "_agent_lens": lens,
        }
    return {
        "_agent_id": agent_id,
        "_agent_model": "test-model",
        "_agent_lens": lens,
        "final_recommendation": recommendation,
        "executive_summary": f"{agent_id} summary",
        "top_risks": [
            {"name": "Test Risk", "severity": 0.5, "judgement": "needs_attention"}
        ],
        "category_summary": category_summary
        or {
            "Technical": {"avg_severity": 0.5, "pass": 1, "fail": 0, "needs_attention": 0},
            "Ethical": {"avg_severity": 0.3, "pass": 1, "fail": 0, "needs_attention": 0},
            "Legal": {"avg_severity": 0.4, "pass": 0, "fail": 0, "needs_attention": 1},
            "Societal": {"avg_severity": 0.2, "pass": 1, "fail": 0, "needs_attention": 0},
        },
    }


# ---------------------------------------------------------------------------
# Scenario 1: 3/3 unanimous agreement
# ---------------------------------------------------------------------------

class TestUnanimousAgreement:
    def test_3_way_approve(self):
        results = [
            _make_agent_result("agent_1", "approve"),
            _make_agent_result("agent_2", "approve"),
            _make_agent_result("agent_3", "approve"),
        ]
        rec, conf, disagreements = _majority_vote(results)
        assert rec == "approve"
        # Base confidence is 0.95 with possible severity spread adjustment
        assert conf <= CONFIDENCE_ALL_AGREE
        assert conf > 0.8
        assert len(disagreements) == 0 or all("severity spread" in d.lower() or "failed" in d.lower() for d in disagreements)

    def test_3_way_hold_and_fix(self):
        results = [
            _make_agent_result("agent_1", "hold_and_fix"),
            _make_agent_result("agent_2", "hold_and_fix"),
            _make_agent_result("agent_3", "hold_and_fix"),
        ]
        rec, conf, disagreements = _majority_vote(results)
        assert rec == "hold_and_fix"
        assert conf <= CONFIDENCE_ALL_AGREE

    def test_2_way_agree_1_failed(self):
        results = [
            _make_agent_result("agent_1", "approve"),
            _make_agent_result("agent_2", "approve"),
            _make_agent_result("agent_3", None, error=True),
        ]
        rec, conf, _ = _majority_vote(results)
        assert rec == "approve"
        assert conf <= 0.60  # capped because only 2/3 succeeded


# ---------------------------------------------------------------------------
# Scenario 2: 2/3 majority (majority wins, no conservative escalation)
# ---------------------------------------------------------------------------

class TestMajorityWins:
    def test_2_approve_1_conditions(self):
        results = [
            _make_agent_result("agent_1", "approve"),
            _make_agent_result("agent_2", "approve"),
            _make_agent_result("agent_3", "approve_with_conditions"),
        ]
        rec, conf, disagreements = _majority_vote(results)
        assert rec == "approve"
        assert conf <= CONFIDENCE_MAJORITY
        assert len(disagreements) >= 1


# ---------------------------------------------------------------------------
# Scenario 3: 2/3 majority with conservative escalation
# ---------------------------------------------------------------------------

class TestConservativeEscalation:
    def test_minority_hold_and_fix_escalates(self):
        """When minority says hold_and_fix and dissenting severity >= 0.3, it wins."""
        high_sev_summary = {
            "Technical": {"avg_severity": 0.8, "pass": 0, "fail": 1, "needs_attention": 0},
            "Ethical": {"avg_severity": 0.6, "pass": 0, "fail": 0, "needs_attention": 1},
            "Legal": {"avg_severity": 0.7, "pass": 0, "fail": 1, "needs_attention": 0},
            "Societal": {"avg_severity": 0.5, "pass": 0, "fail": 0, "needs_attention": 1},
        }
        results = [
            _make_agent_result("agent_1", "approve"),
            _make_agent_result("agent_2", "approve"),
            _make_agent_result("agent_3", "hold_and_fix", category_summary=high_sev_summary),
        ]
        rec, conf, disagreements = _majority_vote(results)
        assert rec == "hold_and_fix"
        assert any("escalation" in d.lower() for d in disagreements)

    def test_minority_hold_and_fix_low_severity_no_escalation(self):
        """When dissenting agent has avg severity < 0.3, majority 'approve' prevails."""
        low_sev_summary = {
            "Technical": {"avg_severity": 0.1, "pass": 1, "fail": 0, "needs_attention": 0},
            "Ethical": {"avg_severity": 0.1, "pass": 1, "fail": 0, "needs_attention": 0},
            "Legal": {"avg_severity": 0.2, "pass": 1, "fail": 0, "needs_attention": 0},
            "Societal": {"avg_severity": 0.1, "pass": 1, "fail": 0, "needs_attention": 0},
        }
        results = [
            _make_agent_result("agent_1", "approve"),
            _make_agent_result("agent_2", "approve"),
            _make_agent_result("agent_3", "hold_and_fix", category_summary=low_sev_summary),
        ]
        rec, conf, disagreements = _majority_vote(results)
        assert rec == "approve"
        assert any("below 0.3" in d.lower() for d in disagreements)


# ---------------------------------------------------------------------------
# Scenario 4: 2 succeeded but disagree (1 failed)
# ---------------------------------------------------------------------------

class TestTwoSucceededDisagree:
    def test_2_disagree_1_failed(self):
        results = [
            _make_agent_result("agent_1", "approve"),
            _make_agent_result("agent_2", "hold_and_fix"),
            _make_agent_result("agent_3", None, error=True),
        ]
        rec, conf, disagreements = _majority_vote(results)
        # Most conservative wins
        assert rec == "hold_and_fix"
        assert conf <= CONFIDENCE_NO_MAJORITY
        assert len(disagreements) >= 1


# ---------------------------------------------------------------------------
# Scenario 5: 3-way split
# ---------------------------------------------------------------------------

class TestThreeWaySplit:
    def test_all_different(self):
        results = [
            _make_agent_result("agent_1", "approve"),
            _make_agent_result("agent_2", "approve_with_conditions"),
            _make_agent_result("agent_3", "hold_and_fix"),
        ]
        rec, conf, disagreements = _majority_vote(results)
        # Most conservative wins
        assert rec == "hold_and_fix"
        assert conf <= CONFIDENCE_NO_MAJORITY
        assert any("split" in d.lower() for d in disagreements)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_0_agents_succeeded(self):
        results = [
            _make_agent_result("agent_1", None, error=True),
            _make_agent_result("agent_2", None, error=True),
            _make_agent_result("agent_3", None, error=True),
        ]
        rec, conf, disagreements = _majority_vote(results)
        assert rec == "hold_and_fix"
        assert conf == 0.0
        assert any("all agents failed" in d.lower() for d in disagreements)

    def test_1_agent_succeeded(self):
        results = [
            _make_agent_result("agent_1", "approve_with_conditions"),
            _make_agent_result("agent_2", None, error=True),
            _make_agent_result("agent_3", None, error=True),
        ]
        rec, conf, disagreements = _majority_vote(results)
        assert rec == "approve_with_conditions"
        assert conf == 0.35
        assert any("1 of 3" in d for d in disagreements)

    def test_invalid_recommendation_normalized(self):
        results = [
            _make_agent_result("agent_1", "invalid_value"),
            _make_agent_result("agent_2", "approve"),
            _make_agent_result("agent_3", "approve"),
        ]
        rec, _, _ = _majority_vote(results)
        # "invalid_value" gets normalized to "approve_with_conditions"
        # Then 2 approve vs 1 approve_with_conditions → majority approve wins
        assert rec == "approve"


# ---------------------------------------------------------------------------
# _build_disagreements()
# ---------------------------------------------------------------------------

class TestBuildDisagreements:
    def test_flags_significant_severity_spread(self):
        """Agents with >0.3 severity spread on a category should be flagged."""
        results = [
            _make_agent_result(
                "agent_1", "approve",
                category_summary={
                    "Technical": {"avg_severity": 0.8, "pass": 0, "fail": 1, "needs_attention": 0},
                    "Ethical": {"avg_severity": 0.2, "pass": 1, "fail": 0, "needs_attention": 0},
                    "Legal": {"avg_severity": 0.3, "pass": 0, "fail": 0, "needs_attention": 1},
                    "Societal": {"avg_severity": 0.1, "pass": 1, "fail": 0, "needs_attention": 0},
                },
            ),
            _make_agent_result(
                "agent_2", "approve",
                category_summary={
                    "Technical": {"avg_severity": 0.3, "pass": 1, "fail": 0, "needs_attention": 0},
                    "Ethical": {"avg_severity": 0.2, "pass": 1, "fail": 0, "needs_attention": 0},
                    "Legal": {"avg_severity": 0.3, "pass": 0, "fail": 0, "needs_attention": 1},
                    "Societal": {"avg_severity": 0.1, "pass": 1, "fail": 0, "needs_attention": 0},
                },
            ),
        ]
        disagreements = _build_disagreements(results, [])
        # Technical spread = 0.8 - 0.3 = 0.5 > 0.3 → should flag
        assert any("Technical" in d and "spread" in d for d in disagreements)
        # Ethical/Legal/Societal spread <= 0.3 → should NOT flag
        assert not any("Ethical" in d and "spread" in d for d in disagreements)

    def test_no_flags_when_agents_agree(self):
        results = [
            _make_agent_result("agent_1", "approve"),
            _make_agent_result("agent_2", "approve"),
        ]
        disagreements = _build_disagreements(results, [])
        assert len(disagreements) == 0

    def test_preserves_base_disagreements(self):
        results = [
            _make_agent_result("agent_1", "approve"),
            _make_agent_result("agent_2", "approve"),
        ]
        base = ["Existing disagreement note."]
        disagreements = _build_disagreements(results, base)
        assert "Existing disagreement note." in disagreements

    def test_single_valid_agent_returns_base_only(self):
        results = [
            _make_agent_result("agent_1", "approve"),
            _make_agent_result("agent_2", None, error=True),
        ]
        base = ["Only one agent."]
        disagreements = _build_disagreements(results, base)
        assert disagreements == base


# ---------------------------------------------------------------------------
# _compute_severity_spread()
# ---------------------------------------------------------------------------

class TestComputeSeveritySpread:
    def test_zero_spread_when_identical(self):
        results = [
            _make_agent_result("agent_1", "approve"),
            _make_agent_result("agent_2", "approve"),
        ]
        spread = _compute_severity_spread(results)
        assert spread == 0.0

    def test_nonzero_spread(self):
        results = [
            _make_agent_result(
                "agent_1", "approve",
                category_summary={
                    "Technical": {"avg_severity": 0.9},
                    "Ethical": {"avg_severity": 0.1},
                    "Legal": {"avg_severity": 0.5},
                    "Societal": {"avg_severity": 0.3},
                },
            ),
            _make_agent_result(
                "agent_2", "approve",
                category_summary={
                    "Technical": {"avg_severity": 0.2},
                    "Ethical": {"avg_severity": 0.1},
                    "Legal": {"avg_severity": 0.5},
                    "Societal": {"avg_severity": 0.3},
                },
            ),
        ]
        spread = _compute_severity_spread(results)
        assert spread == pytest.approx(0.7, abs=0.01)

    def test_all_errored_returns_zero(self):
        results = [
            _make_agent_result("a1", None, error=True),
            _make_agent_result("a2", None, error=True),
        ]
        assert _compute_severity_spread(results) == 0.0


# ---------------------------------------------------------------------------
# _run_judge_llm() — free-text parsing (mock mode)
# ---------------------------------------------------------------------------

class TestRunJudgeLLM:
    def test_returns_fallback_on_mock(self, mock_model_profile):
        """In mock mode, LLM returns mock text → fallback summary is used."""
        results = [
            _make_agent_result("agent_1", "approve"),
            _make_agent_result("agent_2", "hold_and_fix"),
            _make_agent_result("agent_3", "approve"),
        ]
        summary, log = _run_judge_llm(results, mock_model_profile)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert isinstance(log, list)
        assert len(log) >= 1
