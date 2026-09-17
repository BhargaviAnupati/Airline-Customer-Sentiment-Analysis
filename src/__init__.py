"""
Airline Customer Sentiment Analysis
====================================

Reusable, script-friendly versions of the logic developed in the project
notebooks (see ../notebooks). Import these modules from a notebook or a
Python script instead of copy-pasting cells:

    from src.data_loader import load_clean_data
    from src.features import make_train_test_split, vectorize_text
    from src.train import train_logistic_regression, train_naive_bayes, train_xgboost
    from src.evaluate import compare_models, inspect_misclassifications
    from src.interpret import top_words_per_class

Every function here mirrors what actually ran in notebooks 01-03 -- same
random seed (42), same train/test split, same TF-IDF settings -- so results
computed through src/ match the reported notebook and report results.
"""
