"""Computes the full evaluation metric suite for a fitted binary classifier:
accuracy, precision, recall, F1, confusion matrix, and ROC-AUC (when the
model can produce probability/decision scores). Also builds a
multi-model comparison table used for best-model selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    def __init__(self, positive_label: str = "positive"):
        self.positive_label = positive_label

    def _get_scores(self, model, X) -> np.ndarray | None:
        """Return positive-class probability/decision scores if the model supports it."""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        return None

    def evaluate(self, model, X, y_true) -> dict[str, Any]:
        y_pred = model.predict(X)
        scores = self._get_scores(model, X)

        summary = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred)), 4),
            "recall": round(float(recall_score(y_true, y_pred)), 4),
            "f1": round(float(f1_score(y_true, y_pred)), 4),
        }
        if scores is not None:
            try:
                summary["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 4)
            except ValueError:
                summary["roc_auc"] = None
        else:
            summary["roc_auc"] = None

        cm = confusion_matrix(y_true, y_pred).tolist()

        return {
            "summary": summary,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "scores": scores,
        }

    @staticmethod
    def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
        """Build a tidy DataFrame comparing every model's summary metrics,
        sorted with the best F1 first.
        """
        rows = []
        for name, r in results.items():
            row = {"model": name, **r["summary"]}
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
        return df
