from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


DATA_PATH = Path("data/processed/apple_support_cases.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    # Temporary labels for the baseline.
    # We will replace these with manually labelled golden data.
    keywords = {
        "software_update": ["update", "ios", "install", "upgrade"],
        "battery_power": ["battery", "charging", "charge", "overheat", "hot"],
        "hardware_device": ["screen", "camera", "speaker", "button", "display"],
        "icloud_data": ["icloud", "photo", "photos", "backup", "sync"],
        "account_security": ["password", "apple id", "phishing", "scam", "account"],
        "app_service": ["facetime", "music", "health", "mail", "app"],
        "device_access": ["locked", "passcode", "activation", "restore"],
        "purchase_tradein": ["buy", "purchase", "trade", "trade-in", "reservation"],
    }

    def weak_label(text):
        text = str(text).lower()

        for intent, terms in keywords.items():
            if any(term in text for term in terms):
                return intent

        return "other"

    df["intent"] = df["customer_text"].apply(weak_label)

    print("Label distribution:")
    print(df["intent"].value_counts())
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        df["customer_text"],
        df["intent"],
        test_size=0.2,
        random_state=42,
        stratify=df["intent"],
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Macro F1: {f1_score(y_test, predictions, average='macro'):.4f}")


if __name__ == "__main__":
    main()