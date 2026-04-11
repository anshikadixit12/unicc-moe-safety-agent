# core/llm_client.py
# ─────────────────────────────────────────────────────────────────────────────
# Thin wrapper that talks to whichever LLM your team has API access to.
# Priority order: Anthropic → OpenAI → Gemini
# Just set the right key in .env and it auto-selects.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os, json
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("ROUTER_MODEL_PROVIDER", "anthropic").lower()
MODEL    = os.getenv("ROUTER_MODEL_NAME", "claude-haiku-4-5-20251001")

async def llm_call(system_prompt: str, user_message: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
    """
    Single async function to call the configured LLM.
    Returns the model's text response as a plain string.
    Swap providers by changing ROUTER_MODEL_PROVIDER in .env — no code changes needed.
    """

    if PROVIDER == "anthropic":
        return await _call_anthropic(system_prompt, user_message, max_tokens)
    elif PROVIDER == "openai":
        return await _call_openai(system_prompt, user_message, max_tokens)
    elif PROVIDER == "gemini":
        return await _call_gemini(system_prompt, user_message, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {PROVIDER}. Set ROUTER_MODEL_PROVIDER to anthropic, openai, or gemini.")


# ── Anthropic ─────────────────────────────────────────────────────────────────

async def _call_anthropic(system_prompt: str, user_message: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = await client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return message.content[0].text


# ── OpenAI ────────────────────────────────────────────────────────────────────

async def _call_openai(system_prompt: str, user_message: str, max_tokens: int, temperature: float = 0.7) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ]
    )
    return response.choices[0].message.content


# ── Google Gemini ─────────────────────────────────────────────────────────────

async def _call_gemini(system_prompt: str, user_message: str, max_tokens: int) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=system_prompt
    )
    # Gemini SDK is sync — run in executor to avoid blocking the event loop
    import asyncio
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: model.generate_content(user_message)
    )
    return response.text


# ── JSON helper ───────────────────────────────────────────────────────────────

def parse_json_response(text: str) -> dict:
    """
    Safely parse JSON from LLM output.
    Handles cases where the model wraps JSON in ```json ... ``` fences.
    """
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON.\nRaw output:\n{text}\nError: {e}")
