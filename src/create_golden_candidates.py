from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/apple_support_cases.csv")
OUTPUT_PATH = Path("evaluation/golden_set.csv")

SAMPLE_SIZE = 200
RANDOM_SEED = 42


def main():
    df = pd.read_csv(INPUT_PATH)

    # Create broad keyword signals only for sampling diversity.
    # These are NOT the final ground-truth labels.
    groups = {
        "software_update": r"\b(update|ios|upgrade|downgrade|install)\b",
        "battery_power": r"\b(battery|charging|charge|overheat|overheating)\b",
        "hardware_device": r"\b(screen|display|camera|speaker|button|keyboard)\b",
        "icloud_data": r"\b(icloud|photo|photos|backup|sync)\b",
        "account_security": r"\b(password|apple id|phishing|scam|security)\b",
        "app_service": r"\b(facetime|music|health|mail|itunes)\b",
        "device_access": r"\b(locked|passcode|activation|restore)\b",
        "purchase_tradein": r"\b(buy|purchase|trade.?in|reservation)\b",
    }

    df["_group"] = "other"

    text = df["customer_text"].fillna("").str.lower()

    for group, pattern in groups.items():
        mask = text.str.contains(pattern, regex=True, na=False)
        df.loc[mask & df["_group"].eq("other"), "_group"] = group

    # Aim for roughly balanced coverage across detected groups.
    per_group = SAMPLE_SIZE // len(groups)

    samples = []

    for group in groups:
        group_df = df[df["_group"].eq(group)]

        if len(group_df) > per_group:
            group_df = group_df.sample(
                n=per_group,
                random_state=RANDOM_SEED,
            )

        samples.append(group_df)

    used = pd.concat(samples, ignore_index=True)

    remaining = SAMPLE_SIZE - len(used)

    if remaining > 0:
        unused = df[~df["tweet_id"].isin(used["tweet_id"])]

        extra = unused.sample(
            n=remaining,
            random_state=RANDOM_SEED,
        )

        used = pd.concat([used, extra], ignore_index=True)

    used = used.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    used.insert(0, "example_id", range(1, len(used) + 1))

    used["intent"] = ""
    used["escalate"] = ""
    used["label_notes"] = ""

    used = used[
        [
            "example_id",
            "customer_text",
            "response_text",
            "intent",
            "escalate",
            "label_notes",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    used.to_csv(OUTPUT_PATH, index=False)

    print(f"Created golden-set candidates: {len(used)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()