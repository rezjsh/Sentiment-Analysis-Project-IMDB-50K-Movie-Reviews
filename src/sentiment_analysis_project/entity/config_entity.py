"""Typed configuration entities. Each pipeline stage receives one of these
frozen dataclasses instead of a loose dict, so downstream code gets IDE
autocomplete and type checking instead of magic-string dictionary keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DataIngestionConfig:
    kaggle_dataset_slug: str
    raw_data_dir: Path
    raw_file_path: Path
    raw_csv_name: str


@dataclass(frozen=True)
class SubsetConfig:
    name: str
    n_samples: Optional[int]
    description: str


@dataclass(frozen=True)
class DataSplitConfig:
    processed_data_dir: Path
    train_file_path: Path
    val_file_path: Path
    test_file_path: Path
    test_size: float
    val_size: float
    random_state: int


@dataclass(frozen=True)
class EDAConfig:
    output_dir: Path
    top_n_words: int
    top_n_ngrams: int
    ngram_range: tuple
    generate_wordcloud: bool
    sample_reviews_to_show: int


@dataclass(frozen=True)
class PreprocessingConfig:
    strategy: str
    lowercase: bool
    remove_html: bool
    remove_punctuation: bool
    remove_numbers: bool
    normalize_whitespace: bool
    remove_stopwords: bool
    lemmatize: bool


@dataclass(frozen=True)
class FeatureConfig:
    strategy: str
    max_features: int
    ngram_range: tuple
    min_df: int
    max_df: float
    sublinear_tf: bool


@dataclass(frozen=True)
class ModelTrainerConfig:
    artifacts_dir: Path
    metrics_dir: Path
    target_column: str
    text_column: str
    positive_label: str
    negative_label: str
    cv_folds: int
    active_models: list = field(default_factory=list)
    model_params: dict = field(default_factory=dict)
    random_state: int = 42
    primary_metric: str = "f1"
