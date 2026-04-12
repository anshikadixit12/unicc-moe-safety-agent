# main.py
# ─────────────────────────────────────────────────────────────────────────────
# UNICC MoE AI Safety Agent — FastAPI Backend
#
# Endpoints:
#   POST /evaluate          Main evaluation endpoint (multipart/form-data)
#   POST /evaluate/text     Text-only evaluation (JSON body)
#   GET  /health            Health check
#   GET  /experts           List available experts and their status
#
# Run locally:
#   uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# Test with curl (see tests/test_curl.sh for full examples):
#   curl -X POST http://localhost:8000/evaluate \
#     -F "file=@your_document.pdf" \
#     -F 'selected_policies=["eu_ai_act","us_nist"]'
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import asyncio, logging, os, time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json

from core.models import EvaluationInput, FinalVerdict
from core.pdf_utils import extract_text_from_pdf, build_evaluation_text
from routers.moe_router import route
from core.aggregator import aggregate
import experts.expert1_petri  as expert1
import experts.expert2_policy as expert2
import experts.expert3_gemini as expert3

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
)
logger = logging.getLogger("unicc.moe")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="UNICC MoE AI Safety Agent",
    description="Mixture of Experts AI Safety Testing Platform — UNICC Capstone 2025",
    version="1.0.0"
)

# CORS — allow the frontend (Team 2 HTML / React) to call this backend
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Main evaluation pipeline ──────────────────────────────────────────────────

async def run_evaluation_pipeline(
    evaluation_input: EvaluationInput,
    pdf_bytes: bytes | None = None
) -> FinalVerdict:
    """
    Core MoE pipeline:
    1. Build evaluation text from all inputs
    2. Router assigns weights
    3. All 3 experts run in parallel (asyncio.gather)
    4. Aggregator combines results
    5. Return FinalVerdict
    """
    start_time = time.time()

    # Build unified evaluation text
    evaluation_text = build_evaluation_text(
        pdf_text=evaluation_input.pdf_text,
        prompt=evaluation_input.prompt,
        ai_response=evaluation_input.ai_response
    )
    logger.info(f"Evaluation text built: {len(evaluation_text)} chars")

    # Router decides weights
    logger.info("Running MoE router...")
    router_decision = await route(evaluation_text)
    logger.info(f"Router decision: {router_decision.weights}")

    # Run all 3 experts in parallel
    logger.info("Running all 3 experts in parallel...")
    expert1_task = expert1.evaluate(evaluation_text, evaluation_input.selected_policies)
    expert2_task = expert2.evaluate(evaluation_text, evaluation_input.selected_policies, pdf_bytes)
    expert3_task = expert3.evaluate(evaluation_text, evaluation_input.selected_policies)

    results = await asyncio.gather(
        expert1_task,
        expert2_task,
        expert3_task,
        return_exceptions=True   # Don't let one failed expert crash everything
    )

    # Handle any exceptions from experts gracefully
    expert_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            expert_name = ["Expert 1 (Petri)", "Expert 2 (Policy)", "Expert 3 (Gemini)"][i]
            logger.error(f"{expert_name} raised exception: {result}")
            # Create a neutral fallback result so aggregation can still proceed
            from core.models import ExpertResult, Verdict, RiskTier
            expert_results.append(ExpertResult(
                expert_id=f"expert{i+1}_error",
                expert_name=f"{expert_name} [Error]",
                score=50.0,
                verdict=Verdict.NEEDS_REVIEW,
                risk_tier=RiskTier.MEDIUM,
                summary=f"Expert could not complete evaluation: {str(result)}"
            ))
        else:
            expert_results.append(result)

    logger.info(f"Expert scores: {[f'{r.expert_id}={r.score:.0f}' for r in expert_results]}")

    # Aggregate into final verdict
    logger.info("Running aggregator...")
    final_verdict = await aggregate(
        expert_results=expert_results,
        router_decision=router_decision,
        system_name=evaluation_input.system_name or "AI System Under Test"
    )

    elapsed = time.time() - start_time
    logger.info(f"Pipeline complete in {elapsed:.1f}s — score={final_verdict.overall_score}, risk={final_verdict.risk_tier}")

    return final_verdict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/evaluate", response_model=None)
async def evaluate_pdf(
    file: Optional[UploadFile]     = File(None,  description="PDF file to evaluate"),
    prompt: Optional[str]          = Form(None,  description="Prompt text to evaluate"),
    ai_response: Optional[str]     = Form(None,  description="AI system response to evaluate"),
    system_name: Optional[str]     = Form("AI System Under Test"),
    selected_policies: Optional[str] = Form('["eu_ai_act","us_nist","iso","unesco"]',
                                            description="JSON array of policy framework IDs")
):
    """
    Primary evaluation endpoint.
    Accepts: PDF file upload, prompt text, AI response text — or any combination.
    Returns: FinalVerdict JSON compatible with Team 2 frontend.

    This endpoint is a DROP-IN REPLACEMENT for the Team 2 n8n webhook.
    Point the frontend's WEBHOOK_URL at http://localhost:8000/evaluate
    and everything works without any other frontend changes.
    """
    # Parse inputs
    pdf_bytes = None
    pdf_text  = None

    if file and file.filename:
        try:
            pdf_bytes = await file.read()
            pdf_text = extract_text_from_pdf(pdf_bytes)
            logger.info(f"PDF uploaded: {file.filename} ({len(pdf_bytes)} bytes)")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF processing error: {str(e)}")

    if not pdf_text and not prompt:
        raise HTTPException(
            status_code=400,
            detail="Provide either a PDF file, a prompt, or both."
        )

    # Parse policies
    try:
        policies = json.loads(selected_policies) if selected_policies else ["eu_ai_act", "us_nist"]
    except json.JSONDecodeError:
        policies = ["eu_ai_act", "us_nist"]

    evaluation_input = EvaluationInput(
        pdf_text=pdf_text,
        prompt=prompt,
        ai_response=ai_response,
        system_name=system_name,
        selected_policies=policies
    )

    try:
        verdict = await run_evaluation_pipeline(evaluation_input, pdf_bytes)
        # Return as dict so Team 2 UI parses it correctly
        return JSONResponse(content=verdict.model_dump(mode="json"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


class TextEvaluationRequest(BaseModel):
    prompt: str
    ai_response: Optional[str] = None
    system_name: Optional[str] = "AI System Under Test"
    selected_policies: list[str] = ["eu_ai_act", "us_nist", "iso", "unesco"]

@app.post("/evaluate/text", response_model=None)
async def evaluate_text(request: TextEvaluationRequest):
    """
    JSON body evaluation endpoint — useful for testing without a PDF.
    Perfect for Postman/curl testing on Day 1.
    """
    evaluation_input = EvaluationInput(
        prompt=request.prompt,
        ai_response=request.ai_response,
        system_name=request.system_name,
        selected_policies=request.selected_policies
    )
    try:
        verdict = await run_evaluation_pipeline(evaluation_input)
        return JSONResponse(content=verdict.model_dump(mode="json"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check — use this to verify the server is running."""
    return {
        "status": "healthy",
        "service": "UNICC MoE AI Safety Agent",
        "version": "1.0.0",
        "experts": ["expert1_petri", "expert2_policy", "expert3_gemini"],
        "llm_provider": os.getenv("ROUTER_MODEL_PROVIDER", "anthropic")
    }

@app.get("/trends/{system_name}")
async def get_trends(system_name: str):
    """
    Returns trend analysis for a system tested multiple times.
    Shows whether safety scores are improving or degrading over time.
    """
    # In a real system this would query a database
    # For the prototype we return the trend structure
    return {
        "system_name": system_name,
        "trend": "insufficient_data",
        "message": "Need at least 3 evaluations to calculate trend",
        "recommendation": "Run multiple evaluations over time to track safety improvements",
        "data_points": []
    }


@app.get("/health/detailed")
async def detailed_health():
    """Detailed health check showing all expert statuses."""
    return {
        "status": "healthy",
        "service": "UNICC MoE AI Safety Agent",
        "version": "1.0.0",
        "experts": {
            "expert1_petri": {"status": "active", "methodology": "adversarial red-teaming"},
            "expert2_policy": {"status": "active", "methodology": "UN compliance auditing"},
            "expert3_gemini": {"status": "active", "methodology": "quantitative risk assessment"}
        },
        "features": [
            "disagreement_detection",
            "audit_trail",
            "confidence_scoring",
            "trend_analysis",
            "policy_alignment"
        ]
    }
    
@app.post("/evaluate/calibrate")
async def calibrate_with_verimedia(
    moe_score: float,
    verimedia_score: float,
    system_name: str = "AI System"
):
    """
    Cross-calibrates MoE safety scores against VeriMedia toxicity scores.
    This demonstrates council synthesis by comparing independent evaluations.
    """
    difference = abs(moe_score - verimedia_score)
    agreement = "strong" if difference < 10 else "moderate" if difference < 25 else "weak"
    
    return {
        "system_name": system_name,
        "moe_score": moe_score,
        "verimedia_score": verimedia_score,
        "score_difference": round(difference, 1),
        "agreement_level": agreement,
        "calibration_note": f"MoE and VeriMedia {'agree' if agreement == 'strong' else 'partially agree' if agreement == 'moderate' else 'disagree'} on this system's safety level",
        "recommendation": "Scores within 10 points indicate strong cross-system validation" if agreement == "strong" else "Significant disagreement warrants human ASRB review"
    }

@app.get("/experts")
async def list_experts():
    """Returns the 3 experts and their capabilities."""
    return {
        "experts": [
            {
                "id": "expert1_petri",
                "name": "Petri Red-Teaming Agent",
                "team": "Team 1 (hg3016-guo)",
                "specialties": ["prompt_injection", "jailbreak", "compliance", "harmful_content"],
                "framework": "inspect_ai",
                "status": "active"
            },
            {
                "id": "expert2_policy",
                "name": "Policy Alignment Agent",
                "team": "Team 2 (Lisayjn749)",
                "specialties": ["policy_alignment", "toxicity", "bias_fairness", "content_safety"],
                "framework": "n8n webhook",
                "status": "active"
            },
            {
                "id": "expert3_gemini",
                "name": "Gemini Risk Assessment Agent",
                "team": "Team 3 (RyanYang1390)",
                "specialties": ["hallucination", "pii_leakage", "risk_classification", "governance"],
                "framework": "Google Gemini",
                "status": "active"
            }
        ]
    }
