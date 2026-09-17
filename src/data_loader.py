"""
Data loading and text cleaning for the Twitter US Airline Sentiment dataset.

Source: Kaggle "Twitter US Airline Sentiment" dataset
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

Corresponds to notebook 01_data_loading_and_eda.ipynb, sections 1-3.
"""

from __future__ import annotations

import re

import pandas as pd

RAW_DATA_PATH = "data/raw/Tweets.csv"
TARGET_COL = "airline_sentiment"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Tweets.csv as-is (14,640 rows x 15 columns)."""
    return pd.read_csv(path)


def clean_tweet(text) -> str:
    """Strip @mentions, URLs, HTML entities, and punctuation/numbers, then
    lowercase. Matches the cleaning used identically across all three
    notebooks."""
    text = str(text)
    text = re.sub(r"@\w+", "", text)  # remove @mentions
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"&\w+;", " ", text)  # remove HTML entities like &amp;
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text.lower()


def load_clean_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw data and add a cleaned `text_clean` column, dropping any
    rows where cleaning left an empty string (rare, but possible for very
    short tweets that are pure @mentions/URLs)."""
    df = load_raw_data(path)
    df["text_clean"] = df["text"].apply(clean_tweet)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    return df


def data_quality_summary(df: pd.DataFrame) -> dict:
    """Quick data-quality snapshot, matching the checks run in notebook 01."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "airlines": sorted(df["airline"].unique().tolist()) if "airline" in df.columns else None,
        "sentiment_classes": sorted(df[TARGET_COL].unique().tolist()) if TARGET_COL in df.columns else None,
        "sentiment_distribution": (
            df[TARGET_COL].value_counts(normalize=True).round(3).to_dict()
            if TARGET_COL in df.columns
            else None
        ),
    }


if __name__ == "__main__":
    data = load_clean_data()
    print(data_quality_summary(data))
