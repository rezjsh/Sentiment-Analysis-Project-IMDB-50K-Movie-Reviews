"""Stratified train / validation / test splitting and persistence to
data/processed/*.csv so that every pipeline stage (EDA, training, evaluation)
reads from the exact same split.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment_analysis_project.entity.config_entity import DataSplitConfig
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)


def split_dataset(df: pd.DataFrame, config: DataSplitConfig) -> dict[str, pd.DataFrame]:
    """Split into train/val/test, stratified on `sentiment`, and write to disk."""
    train_val, test = train_test_split(
        df,
        test_size=config.test_size,
        stratify=df["sentiment"],
        random_state=config.random_state,
    )

    # val_size is expressed as a fraction of the ORIGINAL dataset; convert it
    # to a fraction of the remaining train_val split.
    relative_val_size = config.val_size / (1 - config.test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val["sentiment"],
        random_state=config.random_state,
    )

    splits = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

    config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    splits["train"].to_csv(config.train_file_path, index=False)
    splits["val"].to_csv(config.val_file_path, index=False)
    splits["test"].to_csv(config.test_file_path, index=False)

    for name, part in splits.items():
        logger.info(
            "%s split: %d rows | class balance: %s",
            name,
            len(part),
            part["sentiment"].value_counts(normalize=True).round(3).to_dict(),
        )

    return splits


def load_splits(config: DataSplitConfig) -> dict[str, pd.DataFrame]:
    """Load previously-persisted splits from disk."""
    return {
        "train": pd.read_csv(config.train_file_path),
        "val": pd.read_csv(config.val_file_path),
        "test": pd.read_csv(config.test_file_path),
    }
