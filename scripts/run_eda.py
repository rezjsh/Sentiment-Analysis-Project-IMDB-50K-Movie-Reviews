#!/usr/bin/env python
"""CLI: run the full EDA report over the prepared train/val/test splits
(falls back to the raw dataset directly if splits haven't been created yet).

Usage:
    uv run python scripts/run_eda.py
"""

from __future__ import annotations

from sentiment_analysis_project.config.configuration import ConfigurationManager
from sentiment_analysis_project.data.loader import load_raw_dataset
from sentiment_analysis_project.data.splitter import load_splits
from sentiment_analysis_project.eda.eda import EDAReport
from sentiment_analysis_project.preprocessing.pipeline import preprocess_dataframe
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cm = ConfigurationManager()
    ingestion_cfg = cm.get_data_ingestion_config()
    split_cfg = cm.get_data_split_config()
    eda_cfg = cm.get_eda_config()
    pp_cfg = cm.get_preprocessing_config()

    try:
        splits = load_splits(split_cfg)
        raw_df = splits["train"]
        logger.info("Running EDA over the prepared training split (%d rows).", len(raw_df))
    except FileNotFoundError:
        logger.warning("No prepared splits found — run scripts/prepare_data.py first for full split inspection. "
                        "Falling back to the raw dataset for this report.")
        raw_df = load_raw_dataset(ingestion_cfg.raw_file_path)
        splits = None

    clean_df = preprocess_dataframe(raw_df, pp_cfg, text_column="review")

    report = EDAReport(eda_cfg)
    report.run_full_report(raw_df=raw_df, clean_df=clean_df, splits=splits)
    logger.info("EDA artifacts saved under: %s", eda_cfg.output_dir)


if __name__ == "__main__":
    main()
