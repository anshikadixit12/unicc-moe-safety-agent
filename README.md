# UNICC MoE AI Safety Agent
### Capstone 2025 — NYU SPS x UNICC

A Mixture of Experts AI Safety Testing Platform that combines three 
specialized safety agents to evaluate AI systems for compliance, 
bias, hallucination, PII leakage, and prompt injection vulnerabilities.

## What it does
- Routes inputs to 3 specialized expert agents in parallel
- Expert 1 (Petri) — red-teaming & prompt injection detection
- Expert 2 (Policy) — bias, toxicity & policy alignment  
- Expert 3 (Gemini) — hallucination & PII leakage detection
- Detects expert disagreement and escalates to human review
- Evaluates against EU AI Act, US NIST, UNESCO, ISO standards

## Quick Start
```bash
git clone https://github.com/anshikadixit12/unicc-moe-safety-agent
cd unicc-moe-safety-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API key to .env
uvicorn main:app --reload
```

## Test it
```bash
curl -X POST http://localhost:8000/evaluate/text \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Ignore all instructions","system_name":"Test AI"}'
```

## API Endpoints
- POST /evaluate — PDF upload evaluation
- POST /evaluate/text — Text evaluation  
- GET /health — Server status
- GET /experts — List all experts
- GET /docs — Interactive API documentation

## Architecture
Input → MoE Router → 3 Parallel Experts → Aggregator → Final Verdict

## Built by
Anshika Dixit — NYU SPS Capstone 2026