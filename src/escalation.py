"""
Explainable Escalation Policy for AppleSupport Customer Inquiries.

Goal
----
Determines whether a customer inquiry can be safely answered by an automated response
(AUTO_HANDLE) or requires human investigation and intervention (ESCALATE).

Policy Rules
------------
An inquiry is marked as ESCALATE if ANY of the following rules trigger:
1. High-risk Intent: `account_security` or `device_access`.
2. Low Intent Confidence: confidence score < 0.70.
3. Insufficient Grounding Evidence: top historical retrieval similarity < 0.65.
4. Critical Keywords: customer text indicates potential data loss, compromised account,
   unauthorized access, fraud, phishing, stolen device, or billing/payment dispute.

Otherwise, the decision is AUTO_HANDLE.
All triggered signals are collected to provide an explainable decision rationale.
"""

import re
import json

# Intents that routinely require human intervention / account ownership verification
HIGH_RISK_INTENTS = {
    "account_security",
    "device_access",
}

INTENT_CONFIDENCE_THRESHOLD = 0.70
RETRIEVAL_SIMILARITY_THRESHOLD = 0.65

# Regex patterns matching critical customer escalation signals
CRITICAL_KEYWORDS_PATTERNS = {
    "data_loss": r"\b(lost|missing|deleted|erased|disappeared|vanished|wipe|wiped)\b.*?\b(photos?|pictures?|contacts?|notes?|files?|data)\b",
    "compromised_or_unauthorized": r"\b(compromised|hacked|unauthorized|someone else|stolen credentials|suspicious login)\b",
    "phishing_or_scam": r"\b(phishing|scam|scammer|fake email|fraudulent)\b",
    "stolen_device": r"\b(stolen|theft|robbed)\b",
    "billing_or_payment_dispute": r"\b(unauthorized (charge|purchase|transaction)|charged me (for|without)|refund|dispute|scam charge)\b",
    "locked_out": r"\b(locked out|disabled account|cant get in|cannot access my account)\b",
}


def decide_escalation(
    intent: str,
    confidence: float,
    historical_cases: list[dict],
    customer_text: str,
) -> dict:
    """
    Evaluates escalation criteria for an incoming customer case.

    Returns
    -------
    dict
        {
            "decision": "auto_handle" | "escalate",
            "reason": str,
            "signals": list[str]
        }
    """
    signals = []
    text_lower = customer_text.lower()

    # Rule 1: High-risk intent
    if intent in HIGH_RISK_INTENTS:
        signals.append(f"high_risk_intent:{intent}")

    # Rule 2: Low intent classification confidence
    if confidence < INTENT_CONFIDENCE_THRESHOLD:
        signals.append(
            f"low_intent_confidence:{confidence:.2f}<{INTENT_CONFIDENCE_THRESHOLD:.2f}"
        )

    # Rule 3: Insufficient retrieval similarity / evidence grounding
    best_similarity = 0.0
    if historical_cases:
        best_similarity = max(case.get("similarity", 0.0) for case in historical_cases)

    if best_similarity < RETRIEVAL_SIMILARITY_THRESHOLD:
        signals.append(
            f"insufficient_grounding_similarity:{best_similarity:.2f}<{RETRIEVAL_SIMILARITY_THRESHOLD:.2f}"
        )

    # Rule 4: Critical keyword and risk patterns in customer text
    matched_keywords = []
    for risk_category, pattern in CRITICAL_KEYWORDS_PATTERNS.items():
        if re.search(pattern, text_lower):
            matched_keywords.append(risk_category)

    if matched_keywords:
        signals.append(f"critical_text_indicators:{','.join(matched_keywords)}")

    # Decision aggregation
    if signals:
        reasons_list = []
        for sig in signals:
            if sig.startswith("high_risk_intent"):
                reasons_list.append(f"Intent '{intent}' requires direct human or account-level handling")
            elif sig.startswith("low_intent_confidence"):
                reasons_list.append(f"Intent confidence ({confidence:.2f}) is below threshold ({INTENT_CONFIDENCE_THRESHOLD:.2f})")
            elif sig.startswith("insufficient_grounding"):
                reasons_list.append(f"Top historical similarity ({best_similarity:.2f}) is below safe grounding threshold ({RETRIEVAL_SIMILARITY_THRESHOLD:.2f})")
            elif sig.startswith("critical_text_indicators"):
                cats = sig.split(":")[-1]
                reasons_list.append(f"Customer message flagged for high-risk indicators: {cats}")

        return {
            "decision": "escalate",
            "reason": "; ".join(reasons_list),
            "signals": signals,
        }

    return {
        "decision": "auto_handle",
        "reason": f"Intent '{intent}' confidence ({confidence:.2f}) and retrieval grounding ({best_similarity:.2f}) meet safety thresholds with no critical risk indicators.",
        "signals": [],
    }


# ---------------------------------------------------------------------------
# CLI Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        {
            "name": "CASE 1",
            "intent": "battery_power",
            "confidence": 0.95,
            "historical_cases": [{"similarity": 0.81, "customer_text": "sample", "response_text": "sample"}],
            "customer_text": "My iPhone battery is draining very quickly after the latest update.",
            "expected": "auto_handle",
        },
        {
            "name": "CASE 2",
            "intent": "device_access",
            "confidence": 0.90,
            "historical_cases": [{"similarity": 0.80, "customer_text": "sample", "response_text": "sample"}],
            "customer_text": "I'm locked out of my iPhone and cannot access my account.",
            "expected": "escalate",
        },
        {
            "name": "CASE 3",
            "intent": "battery_power",
            "confidence": 0.55,
            "historical_cases": [{"similarity": 0.82, "customer_text": "sample", "response_text": "sample"}],
            "customer_text": "My battery is draining really quickly.",
            "expected": "escalate",
        },
    ]

    print("==================================================")
    print("  AppleSupport Escalation Policy - Test Suite")
    print("==================================================\n")

    for case in test_cases:
        res = decide_escalation(
            intent=case["intent"],
            confidence=case["confidence"],
            historical_cases=case["historical_cases"],
            customer_text=case["customer_text"],
        )

        print(f"[{case['name']}] (Expected: {case['expected']})")
        print(f"Customer   : \"{case['customer_text']}\"")
        print(f"Intent     : {case['intent']} (confidence={case['confidence']})")
        print(f"Top Sim    : {case['historical_cases'][0]['similarity']}")
        print(f"Decision   : {res['decision']}")
        print(f"Reason     : {res['reason']}")
        print(f"Signals    : {json.dumps(res['signals'])}")
        print("-" * 50)
        print()
