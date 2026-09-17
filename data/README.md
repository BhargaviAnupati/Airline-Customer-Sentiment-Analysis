# Data

## Source

**Twitter US Airline Sentiment** — originally collected by Crowdflower (via
the "February Twitter" tweet-sentiment self-driving-car-style survey),
commonly hosted on Kaggle:
<https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment>

The dataset is `Tweets.csv`, one row per tweet. It isn't committed to this
repo — `data/raw/` and `data/processed/` are gitignored except for a
`.gitkeep` placeholder, so the folder structure is preserved but the data
itself isn't tracked.

## How to (re)download it

Download `Tweets.csv` from the Kaggle dataset page above (Kaggle account
required) and place it in `data/raw/`:

```bash
kaggle datasets download -d crowdflower/twitter-airline-sentiment -p data/raw/
cd data/raw && unzip twitter-airline-sentiment.zip && cd ../..
```

## Raw shape

- **14,640 rows × 15 columns** — one row per tweet, February 2015, mentioning one of 6 US airlines.
- **Airlines covered:** Virgin America, United, Southwest, Delta, US Airways, American.
- **Target:** `airline_sentiment` — `negative` (62.7%), `neutral` (21.2%), `positive` (16.1%) — a 3-class, imbalanced problem.
- **Missing values:** mostly in optional/metadata columns not used as model features — `negativereason_gold` (14,608 missing), `airline_sentiment_gold` (14,600 missing), `tweet_coord` (13,621 missing), `negativereason` (5,462 missing — only populated for negative tweets), `user_timezone` (4,820 missing), `tweet_location` (4,733 missing). The only two columns used for modeling, `text` and `airline_sentiment`, have zero missing values.

## Column reference (columns used in this project)

| Column | Type | Description |
|---|---|---|
| `tweet_id` | Integer | Unique tweet id |
| `airline_sentiment` | Categorical (target) | `negative` / `neutral` / `positive` |
| `airline_sentiment_confidence` | Float | Crowdsourced labeler confidence in the sentiment label |
| `negativereason` | Categorical | Why the tweet was negative (only populated for negative tweets) — see the negative-reason breakdown in `reports/final_report.md` |
| `airline` | Categorical | Which of the 6 airlines the tweet is about |
| `text` | Text | Raw tweet text (the model input, after cleaning — see `src/data_loader.clean_tweet`) |
| `retweet_count` | Integer | Not used as a model feature |
| `tweet_created` | Datetime | Not used as a model feature in this version |

Columns not used for modeling (`name`, `tweet_coord`, `tweet_location`,
`user_timezone`, the `*_gold` columns) are metadata retained from the
original crowdsourcing export.

## Processed data

`src/data_loader.load_clean_data()` loads the raw CSV and adds a
`text_clean` column (mentions/URLs/HTML entities/punctuation stripped,
lowercased), dropping the rare rows where cleaning leaves an empty string.
`src/features.make_train_test_split()` then performs a stratified 80/20
split (11,712 train / 2,928 test), and `src/features.vectorize_text()` fits
a TF-IDF vectorizer (5,000 features, unigrams + bigrams) on the training
text only. Nothing is cached to `data/processed/` by default since
re-running the pipeline is fast.
