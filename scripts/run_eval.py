#!/usr/bin/env python
"""CLI: evaluate all persisted models on the held-out TEST split, producing
confusion-matrix / ROC-curve plots and a final comparison table.

Usage:
    uv run python scripts/run_eval.py
"""

from __future__ import annotations

from sentiment_analysis_project.config.configuration import ConfigurationManager
from sentiment_analysis_project.constants.constants import LABEL_MAP
from sentiment_analysis_project.data.splitter import load_splits
from sentiment_analysis_project.eval.evaluator import ModelEvaluator
from sentiment_analysis_project.preprocessing.pipeline import preprocess_dataframe
from sentiment_analysis_project.utils.common import load_json, load_object, save_json
from sentiment_analysis_project.utils.logger import get_logger
from sentiment_analysis_project.visualize.plots import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curve,
)

logger = get_logger(__name__)


def main() -> None:
    cm = ConfigurationManager()
    split_cfg = cm.get_data_split_config()
    mt_cfg = cm.get_model_trainer_config()
    pp_cfg = cm.get_preprocessing_config()

    splits = load_splits(split_cfg)
    test_df = preprocess_dataframe(splits["test"], pp_cfg, text_column=mt_cfg.text_column)

    vectorizer = load_object(mt_cfg.artifacts_dir / "vectorizer.joblib")
    X_test = vectorizer.transform(test_df["clean_review"])
    y_test = test_df[mt_cfg.target_column].map(LABEL_MAP).values

    evaluator = ModelEvaluator(positive_label=mt_cfg.positive_label)
    plots_dir = mt_cfg.metrics_dir.parent / "plots"

    results = {}
    for name in mt_cfg.active_models:
        model_path = mt_cfg.artifacts_dir / f"model_{name}.joblib"
        if not model_path.exists():
            logger.warning("No saved model found for '%s' at %s — skipping.", name, model_path)
            continue
        model = load_object(model_path)
        metrics = evaluator.evaluate(model, X_test, y_test)
        results[name] = metrics
        plot_confusion_matrix(metrics["confusion_matrix"], name, plots_dir)
        plot_roc_curve(model, X_test, y_test, name, plots_dir)
        logger.info("Test metrics for '%s': %s", name, metrics["summary"])

    comparison_df = ModelEvaluator.comparison_table(results)
    plot_model_comparison(comparison_df, plots_dir, metric=mt_cfg.primary_metric)

    best_info = load_json(mt_cfg.artifacts_dir / "best_model_name.json")
    logger.info("Champion model (selected on validation set): %s", best_info["best_model"])
    logger.info("\nTest-set comparison:\n%s", comparison_df.to_string(index=False))

    save_json(
        mt_cfg.metrics_dir / "test_metrics.json",
        {name: r["summary"] for name, r in results.items()},
    )
    comparison_df.to_csv(mt_cfg.metrics_dir / "test_comparison.csv", index=False)
    logger.info("Final evaluation artifacts saved under %s", mt_cfg.metrics_dir)


if __name__ == "__main__":
    main()
