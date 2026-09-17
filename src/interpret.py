"""
Model interpretation: most predictive words per sentiment class, using the
final Logistic Regression model's coefficients (the most directly
interpretable of the three models compared in this project).

Corresponds to notebook 03_model_comparison_and_evaluation.ipynb, section 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def top_words_per_class(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    top_n: int = 15,
) -> dict[str, pd.DataFrame]:
    """For each sentiment class, the top_n TF-IDF terms with the largest
    positive coefficient in the (one-vs-rest) Logistic Regression model --
    i.e. the words that push a prediction most strongly toward that class.

    Returns {class_label: DataFrame(term, coefficient)}, sorted descending
    by coefficient.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    result = {}

    for class_label in model.classes_:
        class_idx = list(model.classes_).index(class_label)
        coefs = model.coef_[class_idx]
        top_idx = np.argsort(coefs)[-top_n:][::-1]
        result[class_label] = pd.DataFrame(
            {"term": feature_names[top_idx], "coefficient": coefs[top_idx]}
        )

    return result
