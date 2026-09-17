"""
Stratified train/test split and TF-IDF vectorization.

Corresponds to notebook 02_text_preprocessing_and_baseline.ipynb, sections 2-3.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.data_loader import TARGET_COL

RANDOM_STATE = 42
TEST_SIZE = 0.2

TFIDF_PARAMS = dict(
    max_features=5000,
    min_df=3,
    ngram_range=(1, 2),
    stop_words="english",
)


def make_train_test_split(df: pd.DataFrame, target_col: str = TARGET_COL):
    """Stratified 80/20 split on sentiment, identical across all three
    notebooks (seed=42) -- preserves the 62.7% / 21.2% / 16.1%
    negative/neutral/positive class balance in both train and test."""
    X = df["text_clean"]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train: pd.Series, X_test: pd.Series):
    """Fit a TfidfVectorizer on TRAIN only, then transform both train and
    test -- same leakage principle as scaling numeric features in the
    other projects.

    max_features caps vocabulary size to keep things fast; min_df=3 drops
    extremely rare words (more noise than signal); ngram_range=(1, 2)
    includes both single words and two-word phrases (e.g. "customer
    service").

    Returns (X_train_tfidf, X_test_tfidf, vectorizer).
    """
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer
