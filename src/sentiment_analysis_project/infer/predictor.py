"""Inference-time API: load the persisted vectorizer + champion model and
serve single-text or batch (CSV / list) predictions with confidence scores.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentiment_analysis_project.constants.constants import INVERSE_LABEL_MAP
from sentiment_analysis_project.entity.config_entity import ModelTrainerConfig, PreprocessingConfig
from sentiment_analysis_project.preprocessing.cleaners import PreprocessingStrategyFactory
from sentiment_analysis_project.utils.common import load_json, load_object
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentPredictor:
    """Loads artifacts once and reuses them across many predictions —
    avoids re-loading the vectorizer/model from disk on every call.
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        preprocessing_config: PreprocessingConfig,
        use_best: bool = True,
        model_name: str | None = None,
    ):
        self.mt_config = model_trainer_config
        self.cleaner = PreprocessingStrategyFactory.create(preprocessing_config)

        self.vectorizer = load_object(model_trainer_config.artifacts_dir / "vectorizer.joblib")

        if use_best and model_name is None:
            best_info = load_json(model_trainer_config.artifacts_dir / "best_model_name.json")
            model_name = best_info["best_model"]
            logger.info("Loading champion model: %s", model_name)
            self.model = load_object(model_trainer_config.artifacts_dir / "best_model.joblib")
        else:
            model_name = model_name or "best"
            self.model = load_object(model_trainer_config.artifacts_dir / f"model_{model_name}.joblib")

        self.model_name = model_name

    def _predict_scores(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        if hasattr(self.model, "decision_function"):
            # squash decision scores into a pseudo-confidence via a logistic function
            import numpy as np

            raw = self.model.decision_function(X)
            return 1 / (1 + np.exp(-raw))
        return None

    def predict_text(self, text: str) -> dict:
        cleaned = self.cleaner.clean(text)
        X = self.vectorizer.transform([cleaned])
        pred = int(self.model.predict(X)[0])
        scores = self._predict_scores(X)
        confidence = float(scores[0]) if scores is not None else None

        return {
            "text": text,
            "predicted_label": INVERSE_LABEL_MAP[pred],
            "confidence": round(confidence, 4) if confidence is not None else None,
            "model_used": self.model_name,
        }

    def predict_batch(self, texts: list[str]) -> pd.DataFrame:
        cleaned = self.cleaner.clean_batch(texts)
        X = self.vectorizer.transform(cleaned)
        preds = self.model.predict(X)
        scores = self._predict_scores(X)

        df = pd.DataFrame(
            {
                "text": texts,
                "predicted_label": [INVERSE_LABEL_MAP[int(p)] for p in preds],
                "confidence": [round(float(s), 4) for s in scores] if scores is not None else None,
            }
        )
        return df

    def predict_csv(self, input_csv: Path, text_column: str = "review", output_csv: Path | None = None) -> pd.DataFrame:
        df = pd.read_csv(input_csv)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in {input_csv}. Found: {list(df.columns)}")

        results = self.predict_batch(df[text_column].astype(str).tolist())
        merged = pd.concat([df.reset_index(drop=True), results[["predicted_label", "confidence"]]], axis=1)

        if output_csv is not None:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(output_csv, index=False)
            logger.info("Batch predictions written to %s", output_csv)

        return merged
