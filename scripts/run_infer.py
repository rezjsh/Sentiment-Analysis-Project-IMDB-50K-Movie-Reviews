#!/usr/bin/env python
"""CLI: run inference with the persisted champion model.

Usage:
    # single text
    uv run python scripts/run_infer.py --text "This movie was fantastic!"

    # batch CSV (must contain a 'review' column, or pass --text-column)
    uv run python scripts/run_infer.py --input-csv path/to/reviews.csv --output-csv path/to/predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentiment_analysis_project.config.configuration import ConfigurationManager
from sentiment_analysis_project.infer.predictor import SentimentPredictor
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sentiment inference.")
    parser.add_argument("--text", type=str, help="A single review text to classify.")
    parser.add_argument("--input-csv", type=str, help="Path to a CSV file of reviews for batch prediction.")
    parser.add_argument("--output-csv", type=str, help="Where to save batch predictions (CSV).")
    parser.add_argument("--text-column", type=str, default="review", help="Text column name in --input-csv.")
    parser.add_argument("--model-name", type=str, default=None, help="Use a specific model instead of the champion.")
    args = parser.parse_args()

    if not args.text and not args.input_csv:
        parser.error("Provide either --text or --input-csv.")

    cm = ConfigurationManager()
    mt_cfg = cm.get_model_trainer_config()
    pp_cfg = cm.get_preprocessing_config()

    predictor = SentimentPredictor(
        model_trainer_config=mt_cfg,
        preprocessing_config=pp_cfg,
        use_best=args.model_name is None,
        model_name=args.model_name,
    )

    if args.text:
        result = predictor.predict_text(args.text)
        logger.info("Prediction: %s", result)
        print(result)

    if args.input_csv:
        output_csv = Path(args.output_csv) if args.output_csv else Path("artifacts/predictions.csv")
        df = predictor.predict_csv(Path(args.input_csv), text_column=args.text_column, output_csv=output_csv)
        logger.info("Batch prediction complete: %d rows scored.", len(df))
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
