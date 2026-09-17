# Airline Customer Sentiment Analysis

Classifying the sentiment of ~14,600 tweets about 6 US airlines as negative,
neutral, or positive, and diagnosing what actually drives negative customer
experiences. Includes full EDA (sentiment by airline, negative-reason
breakdown, word clouds), TF-IDF text vectorization, a 3-way model comparison
(Logistic Regression, Naive Bayes, XGBoost), and word-level interpretation of
the winning model.

📄 **[Read the full report →](reports/final_report.md)**

## Results at a glance

Final model: **Logistic Regression** (`class_weight='balanced'`, TF-IDF features, 5,000 terms, unigrams + bigrams)

| Model | Macro F1 | Accuracy |
|---|---|---|
| **Logistic Regression (final)** | **0.704** | 0.750 |
| XGBoost | 0.616 | 0.737 |
| Naive Bayes | 0.577 | 0.729 |

*Macro F1 (unweighted average across the 3 classes) matters more than accuracy here — with a 62.7% / 21.2% / 16.1% negative/neutral/positive split, a model can post a deceptively high accuracy by favoring the majority "negative" class. On this metric, the simplest model wins clearly. See the [final report](reports/final_report.md#4-results) for the full per-class breakdown.*

![Confusion matrix — final Logistic Regression model](reports/confusion_matrix_logistic_regression.png)

## Project structure

```
airline-customer-sentiment-analysis/
├── README.md                  <- you are here
├── requirements.txt
├── LICENSE
├── data/
│   ├── README.md              <- dataset source, schema, how to download
│   ├── raw/                   <- (gitignored) Tweets.csv lands here
│   └── processed/             <- (gitignored) optional cached artifacts
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_text_preprocessing_and_baseline.ipynb
│   └── 03_model_comparison_and_evaluation.ipynb
├── src/
│   ├── data_loader.py         <- load + clean tweet text
│   ├── features.py            <- train/test split, TF-IDF vectorization
│   ├── train.py                <- Logistic Regression / Naive Bayes / XGBoost training
│   ├── evaluate.py            <- classification report, macro-F1 comparison, misclassification inspection
│   └── interpret.py           <- most predictive words per class (from Logistic Regression coefficients)
├── models/                    <- (gitignored) trained model + vectorizer artifacts land here
└── reports/
    ├── final_report.md        <- full write-up: EDA, methodology, results, word-level interpretation, limitations
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_xgboost.png
    ├── top_predictive_words_per_class.png
    └── figures/                <- EDA charts referenced in the report
```

## Dataset

[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
— 14,640 tweets from February 2015 about 6 US airlines (Virgin America,
United, Southwest, Delta, US Airways, American), each labeled negative,
neutral, or positive. Full schema and download instructions in
[`data/README.md`](data/README.md).

## How to run this project

```bash
git clone <this-repo-url>
cd airline-customer-sentiment-analysis
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Download the data (see data/README.md)
kaggle datasets download -d crowdflower/twitter-airline-sentiment -p data/raw/
cd data/raw && unzip twitter-airline-sentiment.zip && cd ../..

# Option A — walk through the analysis notebook by notebook:
jupyter lab notebooks/

# Option B — use the reusable pipeline directly in Python:
python -c "
from src.data_loader import load_clean_data
from src.features import make_train_test_split, vectorize_text
from src.train import train_logistic_regression
from src.evaluate import compare_models

df = load_clean_data()
X_train, X_test, y_train, y_test = make_train_test_split(df)
X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)

model = train_logistic_regression(X_train_tfidf, y_train)
preds = model.predict(X_test_tfidf)

print(compare_models(y_test, {'Logistic Regression': preds}))
"
```

Notebooks are numbered and meant to be run in order — each one recreates the
same seeded train/test split (`random_state=42`) and TF-IDF settings so
results line up across notebooks 02–03.

## Methodology summary

1. **EDA** (`01`): load and inspect, clean tweet text (strip @mentions/URLs/HTML entities/punctuation, lowercase), sentiment distribution, sentiment by airline, negative-reason breakdown, tweet length and word clouds by sentiment.
2. **Text preprocessing + baseline** (`02`): stratified 80/20 split preserving class balance; TF-IDF vectorization (5,000 features, unigrams + bigrams, fit on train only); Logistic Regression baseline with `class_weight='balanced'`; misclassification inspection.
3. **Model comparison** (`03`): Naive Bayes and XGBoost benchmarked against the Logistic Regression baseline on the same TF-IDF features; model comparison table; most predictive words per class from the winning model's coefficients.

Full narrative, results tables, and discussion of limitations: **[reports/final_report.md](reports/final_report.md)**.

## Tech stack

Python · pandas · scikit-learn · XGBoost · TF-IDF · matplotlib / seaborn · WordCloud · Jupyter

## License

[MIT](LICENSE)
