"""
Majority-class baseline for AppleSupport intent classification.

Strategy
--------
Find the single most frequent intent in the training set, then predict
that same label for every example in the test set.

This is the simplest possible classifier and sets the absolute floor
that any real model must beat.  It is useful as a sanity check:
  - If a model barely beats majority-class accuracy, it has learned
    almost nothing beyond the class imbalance.
  - Macro F1 under a majority baseline is typically near zero because
    minority classes are never predicted.
"""

import os
import csv
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GOLDEN_CSV = os.path.join(
    os.path.dirname(__file__), "..", "evaluation", "golden_set.csv"
)

INTENTS = [
    "software_update",
    "battery_power",
    "hardware_device",
    "icloud_data",
    "account_security",
    "app_service",
    "device_access",
    "purchase_tradein",
    "general_troubleshooting",
    "other",
]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_golden_set(path):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text   = row["customer_text"].strip()
            intent = row["intent"].strip()
            if text and intent:
                texts.append(text)
                labels.append(intent)
    return texts, labels

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    texts, labels = load_golden_set(GOLDEN_CSV)

    # Same split as TF-IDF baseline - must be identical for fair comparison
    _, _, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    # Most frequent intent in TRAINING set only
    train_counts = Counter(y_train)
    majority_class = train_counts.most_common(1)[0][0]
    majority_count = train_counts[majority_class]

    # Predict majority class for every test example
    y_pred = [majority_class] * len(y_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report   = classification_report(
        y_test,
        y_pred,
        labels=INTENTS,
        zero_division=0,
        digits=3,
    )

    sep = "=" * 62

    print()
    print(sep)
    print("  AppleSupport Intent Classifier -- Majority Baseline")
    print(sep)
    print()
    print(f"  Strategy  : always predict the most frequent training intent")
    print(f"  Dataset   : {len(labels)} total examples")
    print(f"  Train     : {len(y_train)} examples (80%)")
    print(f"  Test      : {len(y_test)} examples  (20%)")
    print(f"  Split     : stratified, random_state=42")
    print()
    print(sep)
    print("  TRAINING INTENT DISTRIBUTION")
    print(sep)
    for intent, count in sorted(train_counts.items(), key=lambda x: -x[1]):
        bar   = "#" * count
        flag  = "  <-- majority" if intent == majority_class else ""
        print(f"  {intent:<25}  {count:>3}  {bar}{flag}")
    print()
    print(sep)
    print(f"  MAJORITY CLASS : '{majority_class}'  ({majority_count}/{len(y_train)} training examples)")
    print(sep)
    print()
    print(sep)
    print("  OVERALL METRICS")
    print(sep)
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print()
    print(sep)
    print("  PER-INTENT METRICS")
    print(sep)
    print(report)
    print(sep)
    print("  INTERPRETATION")
    print(sep)
    print(f"  Any real classifier must exceed {accuracy*100:.1f}% accuracy")
    print(f"  and {macro_f1:.4f} macro F1 to beat this trivial baseline.")
    print(f"  Macro F1 ~ 0 here because all minority classes are never")
    print(f"  predicted, giving them F1 = 0 (only '{majority_class}'")
    print(f"  gets a non-zero score).")
    print()
    print(sep)
    print("  Done.")
    print(sep)
    print()


if __name__ == "__main__":
    run()
