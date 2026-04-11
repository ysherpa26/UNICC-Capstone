"""Tests for pipeline.py — P2 6-stage pipeline functions.

Uses golden response fixtures (pre-recorded LLM outputs) to verify that
parsing, severity clamping, rollup math, and test case generation all
work correctly on realistic data.
"""

import pytest

from pipeline import (
    generate_test_cases,
    rollup_results,
    evaluate_responses,
    CATEGORIES,
    VALID_JUDGEMENTS,
    SEVERITY_DEFAULTS,
    ADVERSARIAL_TEMPLATES,
    _select_adversarial_prompt,
)


# ---------------------------------------------------------------------------
# Stage 2 — generate_test_cases()
# ---------------------------------------------------------------------------

class TestGenerateTestCases:
    def test_produces_one_case_per_risk(self, golden_risk_mapping, mock_model_profile):
        cases = generate_test_cases(golden_risk_mapping, mock_model_profile)
        assert len(cases) == len(golden_risk_mapping["top_risks"])

    def test_test_id_format(self, golden_risk_mapping, mock_model_profile):
        cases = generate_test_cases(golden_risk_mapping, mock_model_profile)
        for case in cases:
            assert case["test_id"].startswith("TC-R")
            assert case["risk_id"] in case["test_id"]

    def test_category_preserved(self, golden_risk_mapping, mock_model_profile):
        cases = generate_test_cases(golden_risk_mapping, mock_model_profile)
        risk_categories = {r["risk_id"]: r["category"] for r in golden_risk_mapping["top_risks"]}
        for case in cases:
            assert case["category"] == risk_categories[case["risk_id"]]

    def test_severity_preserved(self, golden_risk_mapping, mock_model_profile):
        cases = generate_test_cases(golden_risk_mapping, mock_model_profile)
        risk_severities = {r["risk_id"]: r["severity"] for r in golden_risk_mapping["top_risks"]}
        for case in cases:
            assert case["severity"] == risk_severities[case["risk_id"]]

    def test_prompt_is_adversarial(self, golden_risk_mapping, mock_model_profile):
        """Test prompts should contain adversarial patterns, not just 'respond naturally'."""
        cases = generate_test_cases(golden_risk_mapping, mock_model_profile)
        for case in cases:
            assert "Respond naturally" not in case["prompt"]
            # Prompt should reference the risk name
            assert case["risk_name"] in case["prompt"]

    def test_empty_risks_produces_no_cases(self, mock_model_profile):
        empty_risk_map = {"top_risks": []}
        cases = generate_test_cases(empty_risk_map, mock_model_profile)
        assert cases == []

    def test_required_fields_present(self, golden_risk_mapping, mock_model_profile):
        cases = generate_test_cases(golden_risk_mapping, mock_model_profile)
        required = {"test_id", "risk_id", "risk_name", "category", "severity", "test_name", "prompt"}
        for case in cases:
            assert required.issubset(case.keys())


# ---------------------------------------------------------------------------
# Stage 2 — _select_adversarial_prompt()
# ---------------------------------------------------------------------------

class TestSelectAdversarialPrompt:
    def test_high_severity_uses_aggressive_template(self):
        indices = {}
        prompt = _select_adversarial_prompt(
            category="Technical",
            risk_name="Prompt Injection",
            risk_description="Vulnerable to injection",
            severity=0.9,
            model_name="TestBot",
            model_type="chatbot",
            use_case="support",
            used_indices=indices,
        )
        # High severity should skip index 0 (starts at 1+)
        assert "Prompt Injection" in prompt

    def test_rotates_through_templates(self):
        indices = {}
        prompts = []
        for i in range(3):
            p = _select_adversarial_prompt(
                category="Ethical",
                risk_name=f"Risk {i}",
                risk_description=f"Desc {i}",
                severity=0.3,
                model_name="Bot",
                model_type="AI",
                use_case="tasks",
                used_indices=indices,
            )
            prompts.append(p)
        # Should produce different prompts for different calls
        assert len(set(prompts)) == 3

    def test_unknown_category_falls_back_to_technical(self):
        indices = {}
        prompt = _select_adversarial_prompt(
            category="UnknownCategory",
            risk_name="Mystery",
            risk_description="unknown",
            severity=0.5,
            model_name="Bot",
            model_type="AI",
            use_case="tasks",
            used_indices=indices,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# Stage 5 — rollup_results()
# ---------------------------------------------------------------------------

class TestRollupResults:
    def test_category_summary_math(self, golden_evaluated_results):
        """Given 5 risks with known severities and judgements, verify rollup math."""
        rollup = rollup_results(golden_evaluated_results)
        cs = rollup["category_summary"]

        # Technical: 2 risks — R001 (sev=0.85, fail) and R005 (sev=0.4, needs_attention)
        tech = cs["Technical"]
        assert tech["fail"] == 1
        assert tech["needs_attention"] == 1
        assert tech["pass"] == 0
        expected_avg = round((0.85 + 0.4) / 2, 4)
        assert tech["avg_severity"] == expected_avg

        # Ethical: 1 risk — R002 (sev=0.55, needs_attention)
        eth = cs["Ethical"]
        assert eth["needs_attention"] == 1
        assert eth["fail"] == 0
        assert eth["pass"] == 0
        assert eth["avg_severity"] == 0.55

        # Legal: 1 risk — R003 (sev=0.7, fail)
        legal = cs["Legal"]
        assert legal["fail"] == 1
        assert legal["avg_severity"] == 0.7

        # Societal: 1 risk — R004 (sev=0.2, pass)
        soc = cs["Societal"]
        assert soc["pass"] == 1
        assert soc["avg_severity"] == 0.2

    def test_top_risks_sorted_by_severity(self, golden_evaluated_results):
        rollup = rollup_results(golden_evaluated_results)
        severities = [r["severity"] for r in rollup["top_risks"]]
        assert severities == sorted(severities, reverse=True)

    def test_top_risks_has_required_fields(self, golden_evaluated_results):
        rollup = rollup_results(golden_evaluated_results)
        required = {"risk_id", "name", "category", "severity", "judgement"}
        for risk in rollup["top_risks"]:
            assert required.issubset(risk.keys())

    def test_empty_input_produces_zero_averages(self):
        rollup = rollup_results([])
        for cat in CATEGORIES:
            assert rollup["category_summary"][cat]["avg_severity"] == 0.0
            assert rollup["category_summary"][cat]["pass"] == 0
            assert rollup["category_summary"][cat]["fail"] == 0

    def test_unknown_category_defaults_to_technical(self):
        results = [
            {
                "test_id": "TC-R999",
                "risk_id": "R999",
                "risk_name": "Alien Risk",
                "category": "Alien",
                "test_name": "Test",
                "judgement": "fail",
                "severity": 0.9,
                "confidence": 0.8,
                "rationale": "test",
            }
        ]
        rollup = rollup_results(results)
        assert rollup["category_summary"]["Technical"]["fail"] == 1


# ---------------------------------------------------------------------------
# Stage 4 — evaluate_responses() with fallback
# ---------------------------------------------------------------------------

class TestEvaluateResponses:
    def test_fallback_on_invalid_llm(self, golden_risk_mapping, mock_agent_config):
        """In mock mode, LLM returns mock text → fallback kicks in."""
        simulated = [
            {
                "test_id": "TC-R001",
                "risk_id": "R001",
                "risk_name": "Prompt Injection",
                "category": "Technical",
                "severity": 0.8,
                "test_name": "Adversarial Text Perturbation Resilience Test",
                "prompt": "test prompt",
                "response": "test response",
            }
        ]
        evaluated = evaluate_responses(simulated, golden_risk_mapping, mock_agent_config)
        assert len(evaluated) == 1
        # Fallback assigns needs_attention
        assert evaluated[0]["judgement"] in VALID_JUDGEMENTS
        assert 0.0 <= evaluated[0]["severity"] <= 1.0
        assert 0.0 <= evaluated[0]["confidence"] <= 1.0
