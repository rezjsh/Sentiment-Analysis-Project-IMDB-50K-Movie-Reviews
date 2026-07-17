#!/usr/bin/env python
"""CLI: turn the raw Kaggle CSV into a reproducible train/val/test split.

Usage:
    uv run python scripts/prepare_data.py --subset dev
    uv run python scripts/prepare_data.py --subset medium
    uv run python scripts/prepare_data.py --subset full
"""

from __future__ import annotations

import argparse

from sentiment_analysis_project.config.configuration import ConfigurationManager
from sentiment_analysis_project.data.loader import load_raw_dataset, take_subset
from sentiment_analysis_project.data.splitter import split_dataset
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a train/val/test split from the raw dataset.")
    parser.add_argument(
        "--subset",
        choices=["dev", "medium", "full"],
        default="medium",
        help="Named subset size to use (see configs/config.yaml -> subsets).",
    )
    args = parser.parse_args()

    cm = ConfigurationManager()
    ingestion_cfg = cm.get_data_ingestion_config()
    subset_cfg = cm.get_subset_config(args.subset)
    split_cfg = cm.get_data_split_config()

    df = load_raw_dataset(ingestion_cfg.raw_file_path)
    df = take_subset(df, subset_cfg, random_state=split_cfg.random_state)
    split_dataset(df, split_cfg)

    logger.info(
        "Data preparation complete for subset='%s' (%s). Splits saved under %s",
        args.subset,
        subset_cfg.description,
        split_cfg.processed_data_dir,
    )


if __name__ == "__main__":
    main()
