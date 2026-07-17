#!/usr/bin/env python
"""CLI: download the IMDB 50K dataset via the Kaggle API.

Usage:
    uv run python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse

from sentiment_analysis_project.config.configuration import ConfigurationManager
from sentiment_analysis_project.data.kaggle_downloader import KaggleDatasetDownloader
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the IMDB 50K Movie Reviews dataset.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the raw file already exists.")
    args = parser.parse_args()

    cm = ConfigurationManager()
    downloader = KaggleDatasetDownloader(cm.get_data_ingestion_config())
    path = downloader.download(force=args.force)
    logger.info("Done. Raw dataset available at: %s", path)


if __name__ == "__main__":
    main()
