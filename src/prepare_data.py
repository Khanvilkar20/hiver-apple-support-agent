from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/twcs.csv")
OUTPUT_PATH = Path("data/processd/apple_support_cases.csv")

SAMPLE_SIZE = 10_000
RANDOM_SEED = 42


def main():
    print("Loading TWCS dataset...")

    columns = [
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "in_response_to_tweet_id",
    ]

    df = pd.read_csv(RAW_PATH, usecols=columns)

    print(f"Total tweets loaded: {len(df):,}")

    # AppleSupport's tweets are historical support responses.
    apple_replies = df[
        df["author_id"].eq("AppleSupport")
        & df["in_response_to_tweet_id"].notna()
    ][
        ["tweet_id", "text", "in_response_to_tweet_id"]
    ].rename(
        columns={
            "tweet_id": "response_tweet_id",
            "text": "response_text",
            "in_response_to_tweet_id": "customer_tweet_id",
        }
    )

    # Find the customer tweets that AppleSupport replied to.
    customer_tweets = df[
        df["tweet_id"].isin(
            apple_replies["customer_tweet_id"].astype("int64")
        )
        & df["inbound"].eq(True)
    ][
        ["tweet_id", "text"]
    ].rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "text": "customer_text",
        }
    )

    cases = customer_tweets.merge(
        apple_replies,
        on="customer_tweet_id",
        how="inner",
    )

    # Basic cleaning.
    cases["customer_text"] = cases["customer_text"].fillna("").str.strip()
    cases["response_text"] = cases["response_text"].fillna("").str.strip()

    cases = cases[
        (cases["customer_text"] != "")
        & (cases["response_text"] != "")
    ]

    # Remove exact duplicate customer/response pairs.
    cases = cases.drop_duplicates(
        subset=["customer_text", "response_text"]
    )

    # Keep a reproducible working sample.
    if len(cases) > SAMPLE_SIZE:
        cases = cases.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_SEED,
        )

    cases = cases.reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(OUTPUT_PATH, index=False)

    print(f"AppleSupport cases available: {len(customer_tweets):,}")
    print(f"Clean cases: {len(cases):,}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()