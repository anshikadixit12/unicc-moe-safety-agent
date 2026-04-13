# core/models.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared Pydantic models used by router, experts, aggregator, and API layer.
# Every expert MUST return an ExpertResult. The aggregator produces a
# FinalVerdict. These two contracts are the glue of the whole MoE system.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid
import datetime


# ── Enumerations ──────────────────────────────────────────────────────────────

class RiskTier(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"

class Verdict(str, Enum):
    PASS           = "pass"
    NEEDS_REVIEW   = "needs_review"
    FAIL           = "fail"

class SafetyDomain(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK        = "jailbreak"
    TOXICITY         = "toxicity"
    BIAS_FAIRNESS    = "bias_fairness"
    HALLUCINATION    = "hallucination"
    PII_LEAKAGE      = "pii_leakage"
    COMPLIANCE       = "compliance"
    GENERAL          = "general"


# ── Input model ───────────────────────────────────────────────────────────────

class EvaluationInput(BaseModel):
    """
    Everything a user can submit for safety testing.
    At least one of pdf_text, prompt, or prompt_response_pair must be provided.
    """
    pdf_text: Optional[str]            = Field(None, description="Extracted text from uploaded PDF")
    prompt: Optional[str]              = Field(None, description="A single prompt to evaluate")
    ai_response: Optional[str]         = Field(None, description="The AI system's response to the prompt")
    system_name: Optional[str]         = Field("Unknown AI System", description="Name of the AI being tested")
    selected_policies: list[str]       = Field(
        default=["eu_ai_act", "us_nist", "iso", "unesco"],
        description="Compliance frameworks to evaluate against"
    )
    selected_experts: Optional[list[str]] = Field(
        None,
        description="If None, router decides. Otherwise forces specific experts."
    )


# ── Router output ─────────────────────────────────────────────────────────────

class RouterDecision(BaseModel):
    """
    Output of the MoE router.
    weights: float 0.0–1.0 per expert. Higher = more relevant to this input.
    detected_domains: safety domains identified in the input.
    reasoning: plain-English explanation of routing decision (useful for UI).
    """
    weights: dict[str, float] = Field(
        description="Expert name → weight. E.g. {'expert1': 0.9, 'expert2': 0.6, 'expert3': 0.4}"
    )
    detected_domains: list[SafetyDomain] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Why these weights were assigned")


# ── Expert output ─────────────────────────────────────────────────────────────

class Finding(BaseModel):
    """A single safety issue found by an expert."""
    domain:      SafetyDomain
    severity:    int           = Field(..., ge=1, le=5, description="1=low, 5=critical")
    title:       str
    description: str
    policy_refs: list[str]     = Field(default_factory=list, description="e.g. ['EU AI Act Art.13']")

class ExpertResult(BaseModel):
    """
    Standardized output every expert must return.
    This is the contract between experts and the aggregator.
    """
    expert_id:    str                    # "expert1_petri" | "expert2_policy" | "expert3_gemini"
    expert_name:  str                    # Human-readable name
    score:        float                  = Field(..., ge=0, le=100)
    verdict:      Verdict
    risk_tier:    RiskTier
    findings:     list[Finding]          = Field(default_factory=list)
    summary:      str                    = ""
    raw_output:   Optional[dict]         = None   # original response from the expert (for debugging)
    weight_used:  float                  = 1.0    # filled in by aggregator after routing
    confidence:   float                  = 0.0

# ── Aggregator / Final output ─────────────────────────────────────────────────

class PolicyAlignment(BaseModel):
    policy_id: str
    status:    str   # "aligned" | "partially_aligned" | "not_aligned" | "not_assessed"
    reason:    str   = ""

class FinalVerdict(BaseModel):
    """
    The unified response returned by POST /evaluate.
    Designed to be drop-in compatible with the Team 2 frontend's expected JSON shape.
    """
    # Top-level scores (Team 2 UI reads these directly)
    overall_score:    float
    risk_tier:        RiskTier
    verdict:          Verdict
    status:           str        # "Passed" | "Needs Review" | "Failed"

    # MoE breakdown (new panel in UI)
    expert_results:   list[ExpertResult]
    router_decision:  RouterDecision

    # Narrative (Team 2 UI renders these)
    summary:          str   = ""
    recommendation:   str   = ""

    # Structured data (Team 2 UI renders these sections)
    top_risks:        list[str]           = Field(default_factory=list)
    policy_alignment: list[PolicyAlignment] = Field(default_factory=list)

    metrics_snapshot: dict = Field(default_factory=dict)
    disagreement_detected: bool = False
    disagreement_gap: float = 0.0
    asrb_status: str = "NOT_TRIGGERED"
    evaluation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    evaluated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())