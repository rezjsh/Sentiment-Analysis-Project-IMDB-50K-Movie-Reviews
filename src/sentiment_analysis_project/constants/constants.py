"""Project-wide constants: default file locations for the two YAML config files
and the canonical set of label names used throughout the pipeline.
"""

from pathlib import Path

# Repo root is 4 levels up from this file:
# src/sentiment_analysis_project/constants/constants.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_FILE_PATH = PROJECT_ROOT / "configs" / "config.yaml"
PARAMS_FILE_PATH = PROJECT_ROOT / "configs" / "params.yaml"

LABEL_MAP = {"negative": 0, "positive": 1}
INVERSE_LABEL_MAP = {0: "negative", 1: "positive"}

SUPPORTED_MODELS = (
    "logistic_regression",
    "linear_svm",
    "multinomial_nb",
    "random_forest",
    "sgd_linear",
)

SUPPORTED_PREPROCESSING_STRATEGIES = ("minimal", "standard", "aggressive")
SUPPORTED_VECTORIZER_STRATEGIES = ("tfidf", "count", "tfidf_char")
