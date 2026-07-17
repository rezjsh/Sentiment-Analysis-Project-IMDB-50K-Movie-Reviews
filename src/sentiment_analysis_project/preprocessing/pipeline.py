"""Applies the configured cleaning strategy to a DataFrame's text column."""

from __future__ import annotations

import pandas as pd

from sentiment_analysis_project.entity.config_entity import PreprocessingConfig
from sentiment_analysis_project.preprocessing.cleaners import PreprocessingStrategyFactory
from sentiment_analysis_project.utils.common import timeit
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)


@timeit
def preprocess_dataframe(
    df: pd.DataFrame,
    config: PreprocessingConfig,
    text_column: str = "review",
    output_column: str = "clean_review",
) -> pd.DataFrame:
    """Return a copy of `df` with an added `output_column` of cleaned text."""
    cleaner = PreprocessingStrategyFactory.create(config)
    df = df.copy()
    df[output_column] = cleaner.clean_batch(df[text_column].astype(str).tolist())

    empty_after_clean = (df[output_column].str.strip() == "").sum()
    if empty_after_clean:
        logger.warning(
            "%d rows became empty strings after cleaning; consider a less aggressive strategy.",
            empty_after_clean,
        )

    return df
