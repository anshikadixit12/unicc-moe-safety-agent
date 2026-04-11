def test_imports():
    from core.models import EvaluationInput, FinalVerdict, RiskTier, Verdict
    assert RiskTier.LOW == "low"
    assert Verdict.PASS == "pass"

def test_evaluation_input():
    from core.models import EvaluationInput
    inp = EvaluationInput(prompt="test prompt")
    assert inp.prompt == "test prompt"

def test_aggregator_score_to_verdict():
    from core.aggregator import _score_to_verdict
    from core.models import Verdict
    assert _score_to_verdict(85) == Verdict.PASS
    assert _score_to_verdict(70) == Verdict.NEEDS_REVIEW
    assert _score_to_verdict(40) == Verdict.FAIL
