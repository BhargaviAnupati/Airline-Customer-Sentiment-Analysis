"""
Model training: Logistic Regression baseline, Multinomial Naive Bayes, XGBoost.

Corresponds to notebook 02 (baseline) and notebook 03 (Naive Bayes, XGBoost,
and the model comparison).
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.features import RANDOM_STATE


def train_logistic_regression(X_train_tfidf, y_train) -> LogisticRegression:
    """Baseline model (notebook 02) -- also the final, best-performing model
    (see reports/final_report.md). class_weight='balanced' handles the
    62.7/21.2/16.1 class imbalance directly."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_tfidf, y_train)
    return model


def train_naive_bayes(X_train_tfidf, y_train) -> MultinomialNB:
    """Multinomial Naive Bayes (notebook 03) -- a classic, fast text
    classification baseline. Does NOT support class_weight, so imbalance is
    handled differently here (i.e., not at all) -- worth noting when
    comparing results."""
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model


def train_xgboost(X_train_tfidf, y_train):
    """XGBoost (notebook 03). Needs numeric class labels rather than
    strings, so the target is label-encoded first; TF-IDF matrices are
    sparse and XGBoost handles sparse input natively.

    Returns (model, label_encoder) -- use label_encoder.inverse_transform()
    on predictions to get back string labels.
    """
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_tfidf, y_train_enc)
    return model, label_encoder


def train_all_models(X_train_tfidf, y_train):
    """Convenience wrapper: train all three models used in the project.

    Returns a dict of {name: fitted_model}. XGBoost's entry is the
    (model, label_encoder) tuple returned by train_xgboost -- predictions
    need label_encoder.inverse_transform() applied.
    """
    return {
        "Logistic Regression": train_logistic_regression(X_train_tfidf, y_train),
        "Naive Bayes": train_naive_bayes(X_train_tfidf, y_train),
        "XGBoost": train_xgboost(X_train_tfidf, y_train),
    }
