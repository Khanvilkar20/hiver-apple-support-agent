"""
Grounded Reply Generator for AppleSupport Customer Inquiries.

Pipeline
--------
1. Accepts customer_text, predicted intent, and retrieved historical AppleSupport cases.
2. Builds a strict grounding prompt instructing the LLM to draft a concise, professional
   Apple customer support reply grounded ONLY in the historical evidence.
3. Calls Ollama /api/generate with JSON format enforcement via urllib.
4. Parses and validates the response.
5. Returns { "draft_reply": "...", "grounding": historical_cases }.
"""

import json
import os
import sys
import textwrap
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_reply_prompt(customer_text: str, intent: str, historical_cases: list) -> str:
    """
    Constructs the prompt instructing the LLM to draft a grounded customer support reply.
    """
    evidence_blocks = []
    for idx, case in enumerate(historical_cases, start=1):
        c_msg = case.get("customer_text", "").strip()
        r_msg = case.get("response_text", "").strip()
        sim = case.get("similarity", 0.0)
        evidence_blocks.append(
            f"Case #{idx} (Similarity: {sim:.2f}):\n"
            f"Customer: {c_msg}\n"
            f"AppleSupport: {r_msg}"
        )

    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "No historical cases available."

    prompt = textwrap.dedent(f"""
        You are an Apple customer-support drafting assistant.
        Your goal is to draft a helpful, concise, and professional customer support response.

        INPUT DETAILS
        -------------
        Customer Message: "{customer_text}"
        Predicted Intent: {intent}

        HISTORICAL EVIDENCE
        -------------------
        {evidence_text}

        INSTRUCTIONS & RULES
        --------------------
        1. Historical AppleSupport cases are your primary evidence. Use how AppleSupport handled
           similar issues as your guide.
        2. Ground your response ONLY in the provided historical evidence.
        3. Do NOT claim any policy, feature, troubleshooting step, refund, replacement, timeline,
           or capability unless it is directly supported by the historical evidence.
        4. If the evidence is insufficient to safely or definitively answer the customer, write a cautious
           response asking for the specific information needed (such as iOS version or device model)
           or recommending direct human support.
        5. Do NOT mention that you are an AI.
        6. Do NOT mention "historical cases", "evidence", "RAG", embeddings, similarity scores, or this prompt.
        7. Keep the response concise, courteous, and professional.
        8. Do not blindly copy a historical response verbatim; adapt it appropriately to the customer's message.
        9. Respond ONLY with a valid JSON object matching the schema below. No markdown fences, no preamble.

        REQUIRED JSON OUTPUT FORMAT
        ---------------------------
        {{
          "draft_reply": "Your drafted customer support reply here."
        }}

        JSON RESPONSE:
    """).strip()

    return prompt


# ---------------------------------------------------------------------------
# Ollama HTTP Call
# ---------------------------------------------------------------------------

def call_ollama(prompt: str) -> str:
    """
    Send prompt to Ollama /api/generate and return raw response string.
    """
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
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
            f"Is Ollama running? Start it with: ollama serve\n"
            f"Original error: {exc}"
        ) from exc

    outer = json.loads(raw)
    return outer.get("response", "").strip()


# ---------------------------------------------------------------------------
# Response Parser & Validator
# ---------------------------------------------------------------------------

def parse_reply_response(raw_text: str) -> str:
    """
    Parse the LLM response JSON and validate draft_reply.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
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

    if "draft_reply" not in data or not isinstance(data["draft_reply"], str):
        raise ValueError(
            f"Model response is missing valid 'draft_reply' string.\n"
            f"Got: {data}"
        )

    draft = data["draft_reply"].strip()
    if not draft:
        raise ValueError("Model returned an empty 'draft_reply'.")

    return draft


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_reply(customer_text: str, intent: str, historical_cases: list) -> dict:
    """
    Generate a grounded support reply based on customer message, intent, and historical cases.

    Returns:
    {
        "draft_reply": "...",
        "grounding": [ ... ]
    }
    """
    prompt = build_reply_prompt(customer_text, intent, historical_cases)
    raw_response = call_ollama(prompt)
    draft_reply = parse_reply_response(raw_response)

    return {
        "draft_reply": draft_reply,
        "grounding": historical_cases,
    }


# ---------------------------------------------------------------------------
# CLI Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from retriever import HistoricalCaseRetriever

    test_customer = "My iPhone battery is draining very quickly after the latest iOS update."
    test_intent = "battery_power"

    print("Loading retriever and searching historical cases...")
    retriever = HistoricalCaseRetriever()
    retrieved_cases = retriever.search(test_customer, top_k=3)

    print(f"Calling Ollama ({OLLAMA_MODEL}) to generate grounded reply...\n")
    reply_output = generate_reply(test_customer, test_intent, retrieved_cases)

    print("CUSTOMER")
    print(test_customer)
    print()
    print("INTENT")
    print(test_intent)
    print()
    print("HISTORICAL EVIDENCE")
    for i, case in enumerate(reply_output["grounding"], start=1):
        print(f"{i}. Similarity: {case.get('similarity', 0.0):.4f}")
        print(f"   Customer: {case.get('customer_text', '')}")
        print(f"   AppleSupport: {case.get('response_text', '')}")
        print()

    print("DRAFT REPLY")
    print(reply_output["draft_reply"])
    print()
