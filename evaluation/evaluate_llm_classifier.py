"""
Evaluation script for Ollama-based LLM Intent Classifier against Golden Set.

Pipeline
--------
1. Load 200 human-reviewed cases from evaluation/golden_set.csv.
2. For each case, classify customer_text using classify_intent() from src/llm_classifier.py.
3. Compare against ground truth `intent`.
4. Calculate and report:
   - Overall Accuracy & Macro F1
   - Per-intent Precision, Recall, and F1
   - Confusion Matrix
   - Average model confidence
   - Coverage and accuracy for confident predictions (confidence >= 0.70)
5. Save granular results to evaluation/llm_predictions.csv.
"""

import os
import sys
import csv
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Force UTF-8 output to prevent windows charmap encoding crashes
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Link to src
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_DIR)

from llm_classifier import classify_intent, OLLAMA_MODEL
from intent_taxonomy import INTENTS

GOLDEN_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "golden_set.csv"))
PREDICTIONS_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "llm_predictions.csv"))

ORDERED_INTENTS = list(INTENTS.keys())


def main():
    if not os.path.exists(GOLDEN_CSV):
        raise FileNotFoundError(f"Golden set not found at {GOLDEN_CSV}")

    # Read ground truth
    rows = []
    with open(GOLDEN_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total_examples = len(rows)
    print(f"Loaded {total_examples} examples from {GOLDEN_CSV}")
    print(f"Evaluating with Ollama model: {OLLAMA_MODEL}\n")

    results = []
    y_true = []
    y_pred = []
    confidences = []

    for idx, row in enumerate(rows, start=1):
        eid = row.get("example_id", str(idx))
        customer_text = row.get("customer_text", "").strip()
        true_intent = row.get("intent", "").strip()

        print(f"Evaluating {idx}/{total_examples} (ID: {eid})...", flush=True)

        predicted_intent = "error_or_none"
        confidence = 0.0
        reason = "Failed request"

        try:
            res = classify_intent(customer_text)
            predicted_intent = res.get("intent", "other")
            confidence = float(res.get("confidence", 0.0))
            reason = res.get("reason", "")
        except Exception as e:
            print(f"  Warning: Row {eid} failed with error: {e}", flush=True)
            reason = f"Error: {e}"

        is_correct = (predicted_intent == true_intent)

        y_true.append(true_intent)
        y_pred.append(predicted_intent)
        confidences.append(confidence)

        results.append({
            "example_id": eid,
            "customer_text": customer_text,
            "true_intent": true_intent,
            "predicted_intent": predicted_intent,
            "confidence": f"{confidence:.4f}",
            "reason": reason,
            "correct": str(is_correct).lower(),
        })

    # Save detailed predictions
    fieldnames = [
        "example_id",
        "customer_text",
        "true_intent",
        "predicted_intent",
        "confidence",
        "reason",
        "correct",
    ]
    with open(PREDICTIONS_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDetailed predictions saved to {PREDICTIONS_CSV}\n")

    # Metrics Computation
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    avg_conf = np.mean(confidences) if confidences else 0.0

    # High-confidence metrics (>= 0.70)
    high_conf_indices = [i for i, c in enumerate(confidences) if c >= 0.70]
    high_conf_coverage = len(high_conf_indices) / total_examples if total_examples > 0 else 0.0
    if high_conf_indices:
        high_conf_true = [y_true[i] for i in high_conf_indices]
        high_conf_pred = [y_pred[i] for i in high_conf_indices]
        high_conf_acc = accuracy_score(high_conf_true, high_conf_pred)
    else:
        high_conf_acc = 0.0

    # Classification report & confusion matrix
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred, labels=ORDERED_INTENTS, zero_division=0, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    # Print Final Terminal Report
    sep = "=" * 50
    print(sep)
    print("AppleSupport LLM Intent Evaluation")
    print(sep)
    print(f"\nExamples: {total_examples}")
    print(f"Model: {OLLAMA_MODEL}\n")

    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Average confidence: {avg_conf:.4f}\n")

    print("Confidence >= 0.70")
    print(f"Coverage: {high_conf_coverage:.4f} ({high_conf_coverage*100:.2f}%) ({len(high_conf_indices)}/{total_examples})")
    print(f"Accuracy: {high_conf_acc:.4f} ({high_conf_acc*100:.2f}%)\n")

    print("PER-INTENT METRICS")
    print(report)

    print("CONFUSION MATRIX")
    print("Rows = true label, Columns = predicted label\n")
    abbrev = [lb[:8] for lb in unique_labels]
    col_w = max(len(a) for a in abbrev) + 2
    header = "".ljust(col_w) + "".join(a.rjust(col_w) for a in abbrev)
    print(header)
    print("-" * len(header))
    for i, row_label in enumerate(unique_labels):
        row_str = row_label[:col_w - 1].ljust(col_w)
        row_str += "".join(str(v).rjust(col_w) for v in cm[i])
        print(row_str)
    print("\n" + sep)


if __name__ == "__main__":
    main()
