"""Loads the raw IMDB CSV, validates its schema, and can draw reproducible
named subsets (dev / medium / full) for fast Colab-friendly experimentation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentiment_analysis_project.entity.config_entity import SubsetConfig
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"review", "sentiment"}


def load_raw_dataset(raw_file_path: Path) -> pd.DataFrame:
    """Load the raw Kaggle CSV and perform basic schema validation."""
    if not raw_file_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_file_path}. "
            f"Run the Kaggle downloader first: `uv run python scripts/download_data.py`"
        )

    df = pd.read_csv(raw_file_path)
    missing = REQUIRED_COLUMNS - set(c.lower() for c in df.columns)
    df.columns = [c.lower().strip() for c in df.columns]

    if missing:
        raise ValueError(
            f"Raw dataset is missing expected column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    n_before = len(df)
    df = df.dropna(subset=["review", "sentiment"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    n_after = len(df)
    if n_after < n_before:
        logger.info(
            "Dropped %d rows during load (missing values / duplicate reviews).",
            n_before - n_after,
        )

    logger.info("Loaded raw dataset: %d rows, columns=%s", len(df), list(df.columns))
    return df


def take_subset(df: pd.DataFrame, subset: SubsetConfig, random_state: int = 42) -> pd.DataFrame:
    """Return a reproducible, class-stratified subset of `df` of size
    `subset.n_samples` (or the full frame if n_samples is None / >= len(df)).
    """
    if subset.n_samples is None or subset.n_samples >= len(df):
        logger.info("Using full dataset (%d rows) for subset '%s'.", len(df), subset.name)
        return df.reset_index(drop=True)

    frac_per_class = subset.n_samples / len(df)
    sampled = (
        df.groupby("sentiment", group_keys=False)
        .apply(lambda g: g.sample(frac=frac_per_class, random_state=random_state))
        .reset_index(drop=True)
    )
    logger.info(
        "Subset '%s': sampled %d/%d rows (stratified by sentiment).",
        subset.name,
        len(sampled),
        len(df),
    )
    return sampled
