from __future__ import annotations
"""Output schema definitions (single agent + ensemble wrapper)"""

"""
schemas.py — Contract Schema Enforcement
==========================================
Validates all JSON data flowing through the system.

What this file does:
    1. Validates INPUT from the user (7 model_profile fields or GitHub URL)
    2. Validates OUTPUT from P2's pipeline (single-agent reports + ensemble wrapper)
    3. Provides safe validation helpers so server.py doesn't crash on bad data

What this file does NOT do:
    - No API calls (that's config.py)
    - No routing (that's server.py)
    - No pipeline logic (that's P2's files)

Key design decision:
    INPUT models are STRICT (reject bad user submissions with clear errors).
    OUTPUT models are LENIENT (accept extra/missing fields, fill defaults).
    Why? We control what the user sends. We do NOT control what P2's LLM returns.
    A partial result is better than a crash.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ===========================================================================
# ENUMS — allowed values for controlled fields
# ===========================================================================

class DeploymentType(str, Enum):
    cloud = "cloud"
    on_prem = "on_prem"
    hybrid = "hybrid"


class LoggingType(str, Enum):
    enabled = "enabled"
    disabled = "disabled"


class Recommendation(str, Enum):
    approve = "approve"
    approve_with_conditions = "approve_with_conditions"
    hold_and_fix = "hold_and_fix"


class Judgement(str, Enum):
    pass_judgement = "pass"
    fail = "fail"
    needs_attention = "needs_attention"


# ===========================================================================
# INPUT MODELS — strict validation (we control the user's form)
# ===========================================================================

class ModelProfile(BaseModel):
    """The 7 fields describing the AI agent being evaluated.
    Used for validating user form submissions (strict mode).
    """
    name: str = Field(..., min_length=1, description="Name of the AI agent")
    type: str = Field(..., min_length=1, description="What kind of agent (chatbot, classifier, etc.)")
    use_case: str = Field(..., min_length=1, description="What the agent does")
    deployment: DeploymentType = Field(..., description="Where it runs: cloud, on_prem, or hybrid")
    auth: str = Field(default="none", description="Access control method")
    finetune_data: str = Field(default="none", description="Training data description")
    logging: LoggingType = Field(default=LoggingType.disabled, description="Whether logging is active")


class EvaluateRequest(BaseModel):
    """What server.py receives from the browser.
    Must have EITHER a GitHub URL OR a model_profile (or both).
    """
    github_url: Optional[str] = None
    model_profile: Optional[ModelProfile] = None

    @model_validator(mode="after")
    def check_at_least_one_input(self):
        if not self.github_url and not self.model_profile:
            raise ValueError(
                "Submit either a GitHub URL or fill in the model profile fields. "
                "Both cannot be empty."
            )
        return self

    @field_validator("github_url")
    @classmethod
    def validate_url_format(cls, v):
        if v is not None and v.strip():
            v = v.strip()
            if not v.startswith(("https://", "http://")):
                raise ValueError(
                    f"GitHub URL must start with https:// — got: {v[:50]}"
                )
        return v


# ===========================================================================
# OUTPUT MODELS — lenient validation (we do NOT control what the LLM returns)
#
# All output models use:
#   extra = "allow"  →  ignore unexpected fields instead of crashing
#   defaults on everything  →  missing fields get safe fallbacks
# ===========================================================================

class ReportMeta(BaseModel):
    """Metadata about a single agent's evaluation run."""
    model_config = ConfigDict(extra="allow")

    run_id: str = ""
    generated_at: str = ""
    framework_version: str = ""
    protocol_version: str = ""
    generator: str = ""


class ModelSummary(BaseModel):
    """Echo of the model_profile inside agent output.
    LENIENT — no enums, because the LLM might return 'Cloud' instead of 'cloud'.
    This is different from ModelProfile (which validates user input strictly).
    """
    model_config = ConfigDict(extra="allow")

    name: str = ""
    type: str = ""
    use_case: str = ""
    deployment: str = ""
    auth: str = ""
    finetune_data: str = ""
    logging: str = ""


class CategoryScore(BaseModel):
    """Score breakdown for one risk category (Technical, Ethical, Legal, or Societal)."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    avg_severity: float = 0.0
    pass_count: int = Field(default=0, alias="pass")
    fail: int = 0
    needs_attention: int = 0

    @field_validator("avg_severity")
    @classmethod
    def clamp_severity(cls, v):
        """Severity must be 0.0-1.0. If LLM returns a percentage (e.g. 75),
        normalize it. If negative, clamp to 0."""
        if v is None:
            return 0.0
        if v > 1.0:
            # LLM probably returned a percentage — normalize to 0-1
            return min(v / 100.0, 1.0)
        if v < 0.0:
            return 0.0
        return v


class TopRisk(BaseModel):
    """A single identified risk from an agent's assessment."""
    model_config = ConfigDict(extra="allow")

    risk_id: str = ""
    name: str = ""
    category: str = "Technical"  # lenient — string not enum, LLM may capitalize differently
    severity: float = 0.0
    judgement: str = "needs_attention"  # lenient — string not enum

    @field_validator("severity")
    @classmethod
    def clamp_risk_severity(cls, v):
        if v is None:
            return 0.0
        if v > 1.0:
            return min(v / 100.0, 1.0)
        if v < 0.0:
            return 0.0
        return v


class GapAction(BaseModel):
    """A gap finding with recommended action."""
    model_config = ConfigDict(extra="allow")

    gap: str = ""
    impact: str = "Medium"
    recommended_action: str = ""
    priority: str = "medium"
    eta_days: int = 0


class SingleAgentReport(BaseModel):
    """Complete output from one agent's evaluation.
    All fields have defaults so partial data doesn't crash the system.
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    report_meta: ReportMeta = Field(default_factory=ReportMeta)
    model_summary: ModelSummary = Field(default_factory=ModelSummary)
    category_summary: dict[str, Any] = Field(default_factory=dict)
    top_risks: list[TopRisk] = Field(default_factory=list)
    compliance_mapping: dict[str, Any] = Field(default_factory=dict)
    gaps_and_actions: list[GapAction] = Field(default_factory=list)
    final_recommendation: str = "approve_with_conditions"  # lenient — string not enum
    executive_summary: str = ""


class AgentMeta(BaseModel):
    """Identifies one agent in the ensemble."""
    model_config = ConfigDict(extra="allow")

    id: str = ""
    model: str = ""


class EnsembleMeta(BaseModel):
    """Metadata about the ensemble evaluation run."""
    model_config = ConfigDict(extra="allow")

    run_id: str = ""
    timestamp: str = ""
    agent_count: int = 3
    agents: list[AgentMeta] = Field(default_factory=list)


class EnsembleResponse(BaseModel):
    """The final response sent to the browser.
    This is the top-level envelope containing all 3 agent reports + judge synthesis.
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    ensemble_meta: EnsembleMeta = Field(default_factory=EnsembleMeta)
    final_recommendation: str = "approve_with_conditions"  # lenient — string not enum
    synthesized_summary: str = ""
    confidence: float = 0.0
    agent_assessments: list[SingleAgentReport] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    deliberation_log: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v):
        """Confidence must be 0.0-1.0. Normalize if LLM returns percentage."""
        if v is None:
            return 0.0
        if v > 1.0:
            return min(v / 100.0, 1.0)
        if v < 0.0:
            return 0.0
        return v


class ErrorResponse(BaseModel):
    """Returned when the pipeline fails."""
    error: bool = True
    error_code: str = "UNKNOWN"
    error_message: str = ""
    partial_results: Optional[dict] = None


# ===========================================================================
# HELPER FUNCTIONS — used by server.py
# ===========================================================================

def validate_ensemble_response(raw_data: dict) -> dict:
    """Safely validate and normalize pipeline output.

    Takes the raw dict from P2's run_ensemble(), validates it through
    the EnsembleResponse model, and returns a clean dict ready for
    the browser.

    If validation fails, returns an ErrorResponse dict instead of crashing.

    Args:
        raw_data: Raw dict from the pipeline

    Returns:
        Clean dict matching the ensemble wrapper schema,
        OR an error response dict if validation fails.
    """
    try:
        validated = EnsembleResponse(**raw_data)
        return validated.model_dump(by_alias=True)
    except Exception as e:
        return ErrorResponse(
            error_code="PARSE_FAILURE",
            error_message=f"Pipeline returned data that doesn't match the expected format: {e}"
        ).model_dump()