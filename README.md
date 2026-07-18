# Sentiment Analysis Project — IMDB 50K Movie Reviews

A production-quality, modular binary sentiment classifier built on the
**IMDB Dataset of 50K Movie Reviews**, managed with **uv**, with full EDA,
a config-driven classical-ML pipeline, and an optional Streamlit app.

## Why the IMDB 50K dataset

- It's a clean, balanced (25K/25K) binary-labeled dataset — no class-imbalance
  handling is required, so the project can focus on preprocessing, feature
  engineering, and model comparison.
- 50K reviews is large enough to train genuinely useful classical models
  (TF-IDF + linear models routinely reach ~89-90% accuracy) but small enough
  to train fully on a laptop or free-tier Colab in minutes.
- It supports the project's "dev / medium / full" subset requirement well:
  a 2K-row dev subset iterates in seconds, a 10K medium subset is
  representative for model comparison, and the full 50K set is used for the
  final trained artifact.

## Project layout

```
configs/                 config.yaml, params.yaml, logging_config.yaml
data/raw/                 downloaded Kaggle CSV lands here
data/processed/           train.csv / val.csv / test.csv (generated)
docs/                     extra documentation
logs/                     rotating run logs
notebooks/                optional exploratory notebooks
artifacts/
  models/                 vectorizer.joblib, model_*.joblib, best_model.joblib
  metrics/                validation_metrics.json, test_metrics.json, comparison CSVs
  plots/                  EDA plots + confusion matrices + ROC curves + comparison chart
scripts/                  CLI entry points (download, prepare, eda, train, eval, infer)
tests/                    pytest suite (unit + end-to-end integration)
app/                      Streamlit interactive demo
src/sentiment_analysis_project/
  config/                 ConfigurationManager (YAML -> typed entities)
  constants/               project-wide constants & paths
  entity/                  frozen dataclass config objects
  utils/                   logger, YAML/joblib/JSON helpers
  data/                    Kaggle downloader, loader, stratified splitter
  eda/                     full EDA report generator
  preprocessing/           Strategy pattern: minimal / standard / aggressive cleaners
  features/                Strategy pattern: tfidf / count / tfidf_char vectorizers
  models/                   Registry + Factory pattern for classical models
  train/                    Template Method training/evaluation workflow
  eval/                     metrics computation & model comparison table
  infer/                    single-text & batch inference API
  visualize/                confusion matrix / ROC / comparison plots
```

## Design patterns used

| Pattern | Where | Why |
|---|---|---|
| **Factory** | `models/factory.py` | Builds any configured sklearn estimator from `params.yaml` without training code ever hardcoding a class name. |
| **Registry** | `models/registry.py` | New models are added with a one-line `@ModelRegistry.register(...)` decorator — no other file changes. |
| **Strategy** | `preprocessing/cleaners.py`, `features/vectorizers.py` | Swap cleaning/vectorization behavior purely via config (`minimal/standard/aggressive`, `tfidf/count/tfidf_char`). |
| **Template Method** | `train/trainer.py` | `BaseTrainingPipeline.run()` fixes the training/eval control flow; `ClassicalMLTrainingPipeline` fills in the steps. A future deep-learning pipeline can subclass the same base. |

## Setup with uv

```bash
# 1. Install uv (if you don't have it): https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install project dependencies (creates .venv and uv.lock)
uv sync

# 3. Include the optional Streamlit app dependency
uv sync --extra app

# 4. Configure Kaggle credentials
cp .env.example .env
# edit .env with your KAGGLE_USERNAME / KAGGLE_KEY, or place ~/.kaggle/kaggle.json
```

> **Note on `uv.lock`:** it is generated automatically the first time you run
> `uv sync` and should be committed to version control for reproducibility.
> It isn't hand-written here since it must be produced by resolving the
> exact dependency graph on your machine/CI.

## Running the pipeline

```bash
# Download the full dataset via the Kaggle API
uv run python scripts/download_data.py

# Prepare a subset + stratified train/val/test split
uv run python scripts/prepare_data.py --subset dev      # 2,000 rows — fast smoke test
uv run python scripts/prepare_data.py --subset medium    # 10,000 rows — fast experimentation
uv run python scripts/prepare_data.py --subset full      # all 50,000 rows — final model

# Run the full EDA suite (plots saved to artifacts/plots/eda/)
uv run python scripts/run_eda.py

# Train every model configured in configs/params.yaml (models.active_models)
uv run python scripts/run_train.py

# Evaluate all persisted models on the held-out test split
uv run python scripts/run_eval.py

# Single-text inference
uv run python scripts/run_infer.py --text "This movie was absolutely wonderful!"

# Batch inference on a CSV of reviews
uv run python scripts/run_infer.py --input-csv data/raw/IMDB_Dataset.csv --output-csv artifacts/predictions.csv

# Interactive Streamlit app
uv run streamlit run app/streamlit_app.py

# Or run the whole thing end-to-end in one command
uv run python main.py --subset dev
```

Equivalent `make` shortcuts (`make setup`, `make download-data`, `make prepare-dev`,
`make eda`, `make train`, `make eval`, `make infer`, `make run-app`, `make test`)
are defined in the `Makefile`.

## Training on a subset vs. the full dataset

The `dev` / `medium` / `full` subsets are defined in `configs/config.yaml`
under `subsets:` and are drawn as **stratified** samples (equal proportion of
positive/negative reviews) so model comparisons stay fair at any size:

- **dev (2,000 rows)** — use while iterating on preprocessing or config
  changes; a full train+eval cycle takes well under a minute even on Colab's
  free CPU tier.
- **medium (10,000 rows)** — a good default for comparing all four models
  and picking hyperparameters; still fast (a minute or two) but far more
  representative than `dev`.
- **full (50,000 rows)** — use for the final artifact you'd actually ship;
  training all four classical models still typically finishes in a few
  minutes on a laptop CPU since TF-IDF + linear/NB models are cheap.

Switch subsets by re-running `scripts/prepare_data.py --subset <name>` — it
overwrites `data/processed/{train,val,test}.csv`, and every downstream stage
(EDA, train, eval) automatically picks up the new split.

## EDA outputs

Running `uv run python scripts/run_eda.py` writes to `artifacts/plots/eda/`:

- `label_distribution.png` — class balance bar chart
- `length_wordcount_distribution.png` — character-length & word-count histograms
- `classwise_length_comparison.png` — word-count boxplot by sentiment
- `top_words.png`, `top_ngrams.png` — most frequent words/n-grams (stopwords removed)
- `wordcloud_positive.png`, `wordcloud_negative.png` — optional word clouds per class
- `eda_summary.json` — every computed statistic (missing values, duplicates,
  split balance, sample reviews, etc.) in machine-readable form
- `EDA_REPORT.md` — a human-readable Markdown summary linking every plot

## Models & evaluation

Four classical models are trained and compared out of the box (all
swappable/extendable via `configs/params.yaml` — no code changes needed):
Logistic Regression, Linear SVM (calibrated for probability output),
Multinomial Naive Bayes, and Random Forest, plus an optional lightweight
`sgd_linear` bonus model. Metrics: accuracy, precision, recall, F1,
confusion matrix, and ROC-AUC, with a full model-comparison table and
automatic best-model selection (by F1, configurable in `params.yaml`).

## Tests

```bash
uv run pytest -v
```

The suite runs entirely on an in-memory synthetic dataset (see
`tests/conftest.py`) so it needs no Kaggle download and completes in seconds,
while still exercising the real preprocessing, vectorization, model
factory/registry, training, and evaluation code paths end-to-end.
