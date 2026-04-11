import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    from core.models import EvaluationInput, FinalVerdict, RiskTier, Verdict
    assert RiskTier.LOW == "low"
    assert Verdict.PASS == "pass"

def test_risk_tiers():
    from core.models import RiskTier
    assert RiskTier.CRITICAL == "critical"
    assert RiskTier.HIGH == "high"
    assert RiskTier.MEDIUM == "medium"

def test_evaluation_input():
    from core.models import EvaluationInput
    inp = EvaluationInput(prompt="test prompt")
    assert inp.prompt == "test prompt"

def test_score_to_verdict():
    from core.aggregator import _score_to_verdict
    from core.models import Verdict
    assert _score_to_verdict(85) == Verdict.PASS
    assert _score_to_verdict(70) == Verdict.NEEDS_REVIEW
    assert _score_to_verdict(40) == Verdict.FAIL

def test_score_to_risk():
    from core.aggregator import _score_to_risk
    from core.models import RiskTier
    assert _score_to_risk(90) == RiskTier.LOW
    assert _score_to_risk(75) == RiskTier.MEDIUM
    assert _score_to_risk(55) == RiskTier.HIGH
    assert _score_to_risk(30) == RiskTier.CRITICAL

def test_pdf_utils():
    from core.pdf_utils import build_evaluation_text
    text = build_evaluation_text(None, "test prompt", "test response")
    assert "test prompt" in text
    assert "test response" in text

def test_disagreement_detection():
    from core.models import FinalVerdict, RiskTier, Verdict, RouterDecision, SafetyDomain
    verdict = FinalVerdict(
        overall_score=50.0,
        risk_tier=RiskTier.MEDIUM,
        verdict=Verdict.NEEDS_REVIEW,
        status="Needs Review",
        expert_results=[],
        router_decision=RouterDecision(
            weights={},
            detected_domains=[SafetyDomain.GENERAL],
            reasoning="test"
        ),
        disagreement_detected=True,
        disagreement_gap=35.0
    )
    assert verdict.disagreement_detected == True
    assert verdict.disagreement_gap == 35.0
