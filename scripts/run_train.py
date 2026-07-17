#!/usr/bin/env python
"""CLI: train every configured model, evaluate on the validation split,
and persist the champion model + comparison metrics.

Usage:
    uv run python scripts/run_train.py
"""

from __future__ import annotations

from sentiment_analysis_project.config.configuration import ConfigurationManager
from sentiment_analysis_project.data.splitter import load_splits
from sentiment_analysis_project.eval.evaluator import ModelEvaluator
from sentiment_analysis_project.train.trainer import ClassicalMLTrainingPipeline
from sentiment_analysis_project.utils.logger import get_logger
from sentiment_analysis_project.visualize.plots import plot_model_comparison

logger = get_logger(__name__)


def main() -> None:
    cm = ConfigurationManager()
    split_cfg = cm.get_data_split_config()
    mt_cfg = cm.get_model_trainer_config()
    pp_cfg = cm.get_preprocessing_config()
    ft_cfg = cm.get_feature_config()

    try:
        splits = load_splits(split_cfg)
    except FileNotFoundError as exc:
        raise SystemExit(
            "No train/val/test splits found. Run `uv run python scripts/prepare_data.py --subset <dev|medium|full>` first."
        ) from exc

    pipeline = ClassicalMLTrainingPipeline(
        model_trainer_config=mt_cfg,
        preprocessing_config=pp_cfg,
        feature_config=ft_cfg,
    )
    outcome = pipeline.run(splits["train"], splits["val"])

    comparison_df = ModelEvaluator.comparison_table(outcome["results"])
    logger.info("\n%s", comparison_df.to_string(index=False))
    plot_model_comparison(comparison_df, mt_cfg.metrics_dir.parent / "plots")

    comparison_df.to_csv(mt_cfg.metrics_dir / "model_comparison.csv", index=False)
    logger.info("Champion model: %s", outcome["best_model"])
    logger.info("Training complete. Artifacts saved under %s", mt_cfg.artifacts_dir)


if __name__ == "__main__":
    main()
