"""
LLM-based intent classifier using a local Ollama model.

How it works
------------
1. Build a structured prompt that lists all 10 allowed intents with
   their definitions and instructs the model to return JSON only.
2. POST the prompt to Ollama's /api/generate endpoint using urllib
   (no extra dependencies needed).
3. Parse and validate the JSON response.
4. Return a dict: {"intent": str, "confidence": float, "reason": str}

Configuration (via environment variables)
-----------------------------------------
OLLAMA_URL   - Ollama API endpoint  (default: http://localhost:11434/api/generate)
OLLAMA_MODEL - Model to use         (default: llama3.2:3b)
"""

import json
import os
import sys
import textwrap
import urllib.request
import urllib.error

# Intent taxonomy lives here - single source of truth for the whole project
sys.path.insert(0, os.path.dirname(__file__))
from intent_taxonomy import INTENTS, format_intents_for_prompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434/api/generate")
# OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

ALLOWED_INTENTS = set(INTENTS.keys())

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(customer_text: str) -> str:
    """
    Construct the full prompt sent to the LLM.

    The prompt:
    - Sets the task context (Apple customer support triage)
    - Lists every allowed intent with its definition
    - Shows the exact JSON format required
    - Forbids inventing new intents
    - Forbids returning anything other than the JSON block
    """
    intent_list = format_intents_for_prompt()

    prompt = textwrap.dedent(f"""
        You are an expert Apple customer support triage assistant.
        Your job is to classify a customer message into exactly one of the
        10 allowed support intents listed below.

        ALLOWED INTENTS (name: definition)
        -----------------------------------
        {intent_list}

        RULES
        -----
        1. You MUST choose exactly one intent from the list above.
        2. Do NOT invent a new intent or use any label not in the list.
        3. Respond with ONLY a single valid JSON object - no preamble,
           no explanation, no markdown fences.
        4. The JSON must have exactly these three keys:
             "intent"     : the chosen intent name (string)
             "confidence" : your confidence from 0.0 to 1.0 (float)
             "reason"     : one short sentence explaining your choice (string)

        EXAMPLE RESPONSE
        ----------------
        {{"intent": "software_update", "confidence": 0.92, "reason": "The customer is asking how to roll back an iOS update."}}

        CUSTOMER MESSAGE
        ----------------
        {customer_text}

        JSON RESPONSE:
    """).strip()

    return prompt

# ---------------------------------------------------------------------------
# Ollama HTTP call
# ---------------------------------------------------------------------------

def call_ollama(prompt: str) -> str:
    """
    Send the prompt to Ollama and return the raw response text.

    Uses urllib so there are zero extra dependencies.
    Ollama's /api/generate endpoint streams responses by default;
    setting stream=False returns a single JSON object with the full
    completion in the "response" field.
    """
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,          # wait for the full response, not a stream
        "format": "json",         # tell Ollama we want JSON output mode
        "options": {
            "temperature": 0.0,   # deterministic; we want the single best label
        },
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Could not reach Ollama at {OLLAMA_URL}.\n"
            f"Is Ollama running?  Start it with: ollama serve\n"
            f"Original error: {exc}"
        ) from exc

    # Ollama wraps the model output inside {"response": "..."} 
    outer = json.loads(raw)
    return outer.get("response", "").strip()

# ---------------------------------------------------------------------------
# Response parser and validator
# ---------------------------------------------------------------------------

def parse_response(raw_text: str) -> dict:
    """
    Parse the model's output into a validated dict.

    Raises ValueError if:
    - The output is not valid JSON
    - Required keys are missing
    - The intent is not one of the 10 allowed values
    - Confidence is not a number between 0 and 1
    """
    # Sometimes models wrap JSON in markdown fences despite instructions.
    # Strip those defensively.
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model returned non-JSON output.\n"
            f"Raw output: {raw_text!r}\n"
            f"JSON error: {exc}"
        ) from exc

    # Validate required keys
    for key in ("intent", "confidence", "reason"):
        if key not in data:
            raise ValueError(
                f"Model response is missing required key '{key}'.\n"
                f"Got: {data}"
            )

    # Validate intent value
    intent = data["intent"]
    if intent not in ALLOWED_INTENTS:
        raise ValueError(
            f"Model returned an invalid intent: '{intent}'.\n"
            f"Allowed intents: {sorted(ALLOWED_INTENTS)}"
        )

    # Validate confidence is numeric and in [0, 1]
    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'confidence' must be a number between 0 and 1, got: {data['confidence']!r}"
        ) from exc

    if not (0.0 <= confidence <= 1.0):
        raise ValueError(
            f"'confidence' must be between 0.0 and 1.0, got: {confidence}"
        )

    return {
        "intent":     intent,
        "confidence": confidence,
        "reason":     str(data["reason"]),
    }

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_intent(customer_text: str) -> dict:
    """
    Classify a customer support message into one of the 10 allowed intents.

    Parameters
    ----------
    customer_text : str
        The raw customer message (tweet / support ticket text).

    Returns
    -------
    dict with keys:
        "intent"     : str   - one of the 10 allowed intent names
        "confidence" : float - model's self-reported confidence (0.0 - 1.0)
        "reason"     : str   - short explanation of the choice

    Raises
    ------
    ConnectionError   - if Ollama is not reachable
    ValueError        - if the model returns an invalid response
    """
    prompt   = build_prompt(customer_text)
    raw      = call_ollama(prompt)
    result   = parse_response(raw)
    return result

# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_MESSAGE = (
        "My iPhone battery dies before lunch after the latest update."
    )

    sep = "=" * 60

    print()
    print(sep)
    print("  AppleSupport LLM Classifier -- Quick Test")
    print(sep)
    print()
    print(f"  Model    : {OLLAMA_MODEL}")
    print(f"  Endpoint : {OLLAMA_URL}")
    print()
    print(f"  Input message:")
    print(f"    \"{TEST_MESSAGE}\"")
    print()
    print("  Calling Ollama... (this may take a few seconds)")
    print()

    try:
        result = classify_intent(TEST_MESSAGE)
    except (ConnectionError, ValueError) as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    print(sep)
    print("  RESULT")
    print(sep)
    print(json.dumps(result, indent=2))
    print()
    print(sep)
    print(f"  Intent     : {result['intent']}")
    print(f"  Confidence : {result['confidence']:.2f}  ({result['confidence']*100:.0f}%)")
    print(f"  Reason     : {result['reason']}")
    print(sep)
    print()
