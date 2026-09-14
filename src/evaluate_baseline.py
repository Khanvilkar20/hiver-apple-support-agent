"""
Baseline evaluation: TF-IDF + Logistic Regression for intent classification.

Pipeline
--------
1. Load the 200-row human-reviewed golden set.
2. Vectorize `customer_text` with TF-IDF (unigrams + bigrams).
3. Classify with Logistic Regression.
4. 80/20 stratified split (random_state=42) for reproducibility.
5. Report accuracy, macro F1, per-intent metrics, and confusion matrix.

Why this baseline?
------------------
TF-IDF captures how distinctive a word is for a given intent across all
messages.  Logistic Regression is linear, fast to train, and its per-class
weights are directly interpretable - making it easy to explain during an
interview or design review.
"""

import os
import sys
import csv
import textwrap

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Configuration
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
# Step 1 - Load data
# ---------------------------------------------------------------------------

def load_golden_set(path: str):
    """Read the golden CSV and return (texts, labels) as plain Python lists."""
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text   = row["customer_text"].strip()
            intent = row["intent"].strip()
            if not text or not intent:
                continue  # skip any incomplete rows (should be none)
            texts.append(text)
            labels.append(intent)
    return texts, labels


# ---------------------------------------------------------------------------
# Step 2 - Build pipeline components
# ---------------------------------------------------------------------------

def build_vectorizer() -> TfidfVectorizer:
    """
    TF-IDF with unigrams and bigrams.
    - lowercase=True   : normalise case
    - ngram_range=(1,2): capture single words AND common two-word phrases
                         (e.g. "battery drain", "activation lock")
    - min_df=1         : keep every token (small dataset, cannot afford pruning)
    """
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
    )


def build_classifier() -> LogisticRegression:
    """
    Logistic Regression trained with L2 regularisation (default).
    max_iter=2000 ensures convergence on the TF-IDF feature space.
    """
    return LogisticRegression(max_iter=2000)


# ---------------------------------------------------------------------------
# Step 3 - Train / evaluate
# ---------------------------------------------------------------------------

def run_evaluation():
    # Load
    texts, labels = load_golden_set(GOLDEN_CSV)
    n_total = len(texts)

    # Split - stratify preserves class proportions in both halves
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    # Vectorise - fit ONLY on training data to avoid leakage
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # Train
    clf = build_classifier()
    clf.fit(X_train_vec, y_train)

    # Predict
    y_pred = clf.predict(X_test_vec)

    # Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    macro_f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_test,
        y_pred,
        labels=INTENTS,
        zero_division=0,
        digits=3,
    )

    present_labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)

    # Print report
    sep = "=" * 62

    print()
    print(sep)
    print("  AppleSupport Intent Classifier -- Baseline Evaluation")
    print(sep)
    print()
    print("  Model   : TF-IDF (1-2 grams) + Logistic Regression")
    print(f"  Dataset : {n_total} examples  ({GOLDEN_CSV})")
    print(f"  Train   : {len(y_train)} examples (80%)")
    print(f"  Test    : {len(y_test)} examples  (20%)")
    print(f"  Split   : stratified, random_state=42")
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

    # Confusion matrix
    print(sep)
    print("  CONFUSION MATRIX")
    print("  Rows = true label   Cols = predicted label")
    print(sep)

    abbrev = [lb[:8] for lb in present_labels]
    col_w  = max(len(a) for a in abbrev) + 2
    header = "".ljust(col_w) + "".join(a.rjust(col_w) for a in abbrev)
    print(header)
    print("-" * len(header))
    for i, row_label in enumerate(present_labels):
        row_str = row_label[:col_w-1].ljust(col_w)
        row_str += "".join(str(v).rjust(col_w) for v in cm[i])
        print(row_str)

    print()
    print(sep)
    print("  TEST-SET PREDICTIONS (id | true | predicted)")
    print(sep)

    # Reload with example_ids for the detail printout
    all_ids, all_texts, all_labels = [], [], []
    with open(GOLDEN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text   = row["customer_text"].strip()
            intent = row["intent"].strip()
            if not text or not intent:
                continue
            all_ids.append(row["example_id"].strip())
            all_texts.append(text)
            all_labels.append(intent)

    _, _, _, _, idx_train, idx_test = train_test_split(
        all_texts,
        all_labels,
        list(range(len(all_texts))),
        test_size=0.20,
        random_state=42,
        stratify=all_labels,
    )

    errors = []
    for rank, orig_idx in enumerate(idx_test):
        eid   = all_ids[orig_idx]
        true  = y_test[rank]
        pred  = y_pred[rank]
        mark  = "   " if true == pred else "[X]"
        print(f"  {mark} id={eid:<4}  true={true:<25}  pred={pred}")
        if true != pred:
            errors.append((eid, true, pred, all_texts[orig_idx]))

    print()
    print(sep)
    print(f"  MISCLASSIFIED  ({len(errors)} of {len(y_test)})")
    print(sep)
    for eid, true, pred, text in errors:
        snippet = textwrap.shorten(text, width=70, placeholder="...")
        print(f"  id={eid}  true={true}  ->  pred={pred}")
        print(f"    \"{snippet}\"")
    print()
    print(sep)
    print("  Done.")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
