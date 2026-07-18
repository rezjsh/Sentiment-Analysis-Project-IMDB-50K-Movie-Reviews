"""Post-training visualizations: confusion matrices, ROC curves, and the
multi-model comparison chart. Saved as PNGs under artifacts/plots/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")


def plot_confusion_matrix(cm: list[list[int]], model_name: str, output_dir: Path, labels=("negative", "positive")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    path = output_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    logger.info("Saved confusion matrix plot: %s", path)
    return path


def plot_roc_curve(model, X_val, y_val, model_name: str, output_dir: Path) -> Path | None:
    if not (hasattr(model, "predict_proba") or hasattr(model, "decision_function")):
        logger.warning("Model '%s' has no scoring method; skipping ROC curve.", model_name)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_val, y_val, ax=ax, name=model_name)
    ax.set_title(f"ROC Curve — {model_name}")
    path = output_dir / f"roc_curve_{model_name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    logger.info("Saved ROC curve plot: %s", path)
    return path


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: Path, metric: str = "f1") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=comparison_df, x="model", y=metric, hue="model", palette="crest", legend=False, ax=ax)
    ax.set_title(f"Model Comparison by {metric.upper()}")
    ax.set_ylim(0, 1)
    for i, v in enumerate(comparison_df[metric]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    path = output_dir / "model_comparison.png"
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    logger.info("Saved model comparison plot: %s", path)
    return path
