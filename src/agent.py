"""
AppleSupport AI Customer Support Agent Orchestrator.

Pipeline
--------
Customer message
    ↓
1. LLM Intent Classifier (src/llm_classifier.py)
    ↓
2. Historical Case Retriever (src/retriever.py)
    ↓
3. Grounded Reply Generator (src/reply_generator.py)
    ↓
4. Escalation Policy (src/escalation.py)
    ↓
Final Structured Result
"""

import os
import sys
import json

# Force UTF-8 standard output for terminal display (emojis, unicode in historical tweets)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure src modules are resolvable
sys.path.insert(0, os.path.dirname(__file__))

from llm_classifier import classify_intent
from retriever import HistoricalCaseRetriever
from reply_generator import generate_reply
from escalation import decide_escalation

# Reuse a single instance of the retriever across queries
_RETRIEVER_INSTANCE = None


def _get_retriever() -> HistoricalCaseRetriever:
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = HistoricalCaseRetriever()
    return _RETRIEVER_INSTANCE


def run_agent(customer_text: str) -> dict:
    """
    Executes the full end-to-end AppleSupport agent workflow for an incoming inquiry.

    Parameters
    ----------
    customer_text : str
        The raw customer support message.

    Returns
    -------
    dict
        Structured record containing intent classification, retrieved historical evidence,
        grounded draft reply, and final escalation decision.
    """
    if not customer_text or not customer_text.strip():
        raise ValueError("customer_text cannot be empty.")

    cleaned_text = customer_text.strip()

    # Step 1: Intent Classification via LLM
    classification_res = classify_intent(cleaned_text)
    intent_name = classification_res.get("intent")
    confidence = float(classification_res.get("confidence", 0.0))
    intent_reason = classification_res.get("reason", "")

    # Step 2: Retrieve Top Historical AppleSupport Cases
    retriever = _get_retriever()
    historical_cases = retriever.search(cleaned_text, top_k=3)

    # Step 3: Grounded Reply Generation via LLM
    reply_res = generate_reply(
        customer_text=cleaned_text,
        intent=intent_name,
        historical_cases=historical_cases,
    )
    draft_reply = reply_res.get("draft_reply", "")

    # Step 4: Rule-based Escalation Policy Evaluation
    escalation_res = decide_escalation(
        intent=intent_name,
        confidence=confidence,
        historical_cases=historical_cases,
        customer_text=cleaned_text,
    )

    return {
        "customer_text": cleaned_text,
        "intent": {
            "name": intent_name,
            "confidence": confidence,
            "reason": intent_reason,
        },
        "historical_evidence": historical_cases,
        "reply": draft_reply,
        "escalation": {
            "decision": escalation_res.get("decision", "escalate"),
            "reason": escalation_res.get("reason", ""),
            "signals": escalation_res.get("signals", []),
        },
    }


def print_case_summary(result: dict):
    """Formats and prints the agent pipeline result clearly for demo/interview."""
    print("==================================================")
    print("APPLE SUPPORT AGENT")
    print("==================================================")
    print()
    print("CUSTOMER")
    print(result["customer_text"])
    print()
    print("INTENT")
    print(result["intent"]["name"])
    print()
    print("CONFIDENCE")
    print(f"{result['intent']['confidence']:.2f} ({result['intent']['confidence']*100:.0f}%)")
    print()
    print("HISTORICAL EVIDENCE")
    for i, case in enumerate(result["historical_evidence"], start=1):
        sim = case.get("similarity", 0.0)
        c_msg = case.get("customer_text", "")
        r_msg = case.get("response_text", "")
        print(f"{i}. Similarity: {sim:.4f}")
        print(f"   Customer: {c_msg}")
        print(f"   AppleSupport: {r_msg}")
        print()

    print("DRAFT REPLY")
    print(result["reply"])
    print()
    decision_str = result["escalation"]["decision"].upper()
    if decision_str == "AUTO_HANDLE":
        decision_label = "AUTO-HANDLE"
    else:
        decision_label = "ESCALATE"

    print("DECISION")
    print(decision_label)
    print()
    print("REASON")
    print(result["escalation"]["reason"])
    print()
    print("SIGNALS")
    signals = result["escalation"]["signals"]
    if signals:
        print(json.dumps(signals))
    else:
        print("None")
    print()


# ---------------------------------------------------------------------------
# CLI Demo Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "My iPhone battery is draining very quickly after the latest iOS update.",
        "I'm locked out of my iPhone and cannot access my account.",
        "Why does my camera suddenly stop focusing after the update?",
    ]

    print("Initializing AppleSupport Agent demo suite...\n")

    for idx, query in enumerate(test_queries, start=1):
        print(f"\n--- RUNNING TEST CASE {idx} ---")
        try:
            res = run_agent(query)
            print_case_summary(res)
        except Exception as e:
            print(f"Error executing agent pipeline for case {idx}: {e}")
            import traceback
            traceback.print_exc()
