"""Downloads the IMDB Dataset of 50K Movie Reviews from Kaggle via the
official Kaggle API.

Requires Kaggle credentials to be available either as:
  - `~/.kaggle/kaggle.json`, or
  - the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables (see .env.example).

Usage (from repo root):
    uv run python -m sentiment_analysis_project.data.kaggle_downloader
or via the packaged CLI script:
    uv run python scripts/download_data.py
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

from sentiment_analysis_project.entity.config_entity import DataIngestionConfig
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)


class KaggleDatasetDownloader:
    """Wraps the Kaggle API so the rest of the pipeline never has to know
    where the data physically comes from (Strategy-friendly seam: a second
    downloader, e.g. for a local zip or an S3 bucket, could implement the
    same `.download()` interface).
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _credentials_available(self) -> bool:
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        return kaggle_json.exists() or (
            os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
        )

    def download(self, force: bool = False) -> Path:
        """Downloads and extracts the dataset. Returns the path to the raw CSV."""
        raw_csv_path = self.config.raw_file_path

        if raw_csv_path.exists() and not force:
            logger.info("Raw dataset already present at %s — skipping download.", raw_csv_path)
            return raw_csv_path

        if not self._credentials_available():
            raise EnvironmentError(
                "Kaggle credentials not found. Place kaggle.json in ~/.kaggle/ "
                "or set KAGGLE_USERNAME and KAGGLE_KEY environment variables. "
                "See .env.example and README.md for setup instructions."
            )

        # Imported lazily: importing kaggle eagerly at module load time raises
        # if credentials are missing, even for unrelated commands.
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

        api = KaggleApi()
        api.authenticate()

        self.config.raw_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Kaggle dataset '%s' ...", self.config.kaggle_dataset_slug)
        api.dataset_download_files(
            self.config.kaggle_dataset_slug,
            path=str(self.config.raw_data_dir),
            unzip=False,
            quiet=False,
        )

        zip_candidates = list(self.config.raw_data_dir.glob("*.zip"))
        if not zip_candidates:
            raise FileNotFoundError("Kaggle download did not produce a .zip archive as expected.")

        zip_path = zip_candidates[0]
        logger.info("Extracting %s ...", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.config.raw_data_dir)
        zip_path.unlink(missing_ok=True)

        extracted_csv = self.config.raw_data_dir / self.config.raw_csv_name
        if extracted_csv.exists() and extracted_csv != raw_csv_path:
            shutil.move(str(extracted_csv), str(raw_csv_path))

        if not raw_csv_path.exists():
            raise FileNotFoundError(
                f"Expected raw CSV at {raw_csv_path} after extraction but it was not found. "
                f"Check config.yaml's `kaggle.raw_csv_name` matches the file inside the Kaggle zip."
            )

        logger.info("Dataset ready at %s", raw_csv_path)
        return raw_csv_path


if __name__ == "__main__":
    from sentiment_analysis_project.config.configuration import ConfigurationManager

    cm = ConfigurationManager()
    downloader = KaggleDatasetDownloader(cm.get_data_ingestion_config())
    downloader.download()
