"""
Evaluation: classification report, macro-F1 model comparison, confusion
matrix, and misclassification inspection.

Corresponds to notebook 02_text_preprocessing_and_baseline.ipynb (section 5-6)
and notebook 03_model_comparison_and_evaluation.ipynb (sections 5-6).
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

CLASS_ORDER = ("negative", "neutral", "positive")


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def full_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred)


def confusion(y_true, y_pred, labels: tuple[str, ...] = CLASS_ORDER):
    return confusion_matrix(y_true, y_pred, labels=list(labels))


def compare_models(y_true, predictions: dict) -> pd.DataFrame:
    """Model comparison table on macro-F1 and accuracy.

    Macro-F1 (unweighted average across the three classes) matters more
    than accuracy here: with a 62.7/21.2/16.1 class split, a model can post
    a deceptively high accuracy by favoring the majority "negative" class
    while doing poorly on "neutral" and "positive" -- macro-F1 penalizes
    that. `predictions` is a {model_name: y_pred array} dict.
    """
    rows = [
        {
            "Model": name,
            "Macro F1": macro_f1(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
        }
        for name, y_pred in predictions.items()
    ]
    return pd.DataFrame(rows).set_index("Model").round(3)


def inspect_misclassifications(X_test, y_true, y_pred) -> pd.DataFrame:
    """Rows where the model got it wrong, flagging the most interesting
    errors (negative predicted as positive, or vice versa -- confusion
    with "neutral" is less surprising and less useful to read)."""
    results_df = pd.DataFrame({"text": X_test.values, "actual": y_true.values, "predicted": y_pred})
    misclassified = results_df[results_df["actual"] != results_df["predicted"]]

    interesting = misclassified[
        ((misclassified["actual"] == "negative") & (misclassified["predicted"] == "positive"))
        | ((misclassified["actual"] == "positive") & (misclassified["predicted"] == "negative"))
    ]
    return interesting
