"""Template Method pattern for the training + evaluation workflow.

`BaseTrainingPipeline.run()` defines the fixed skeleton of steps every
training run goes through. `ClassicalMLTrainingPipeline` fills in each step
for scikit-learn models; a future deep-learning pipeline could subclass
`BaseTrainingPipeline` and override the same hooks without touching the
overall control flow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from sentiment_analysis_project.constants.constants import LABEL_MAP
from sentiment_analysis_project.entity.config_entity import (
    FeatureConfig,
    ModelTrainerConfig,
    PreprocessingConfig,
)
from sentiment_analysis_project.eval.evaluator import ModelEvaluator
from sentiment_analysis_project.features.vectorizers import VectorizerStrategyFactory
from sentiment_analysis_project.models.factory import ModelFactory
from sentiment_analysis_project.preprocessing.pipeline import preprocess_dataframe
from sentiment_analysis_project.utils.common import save_json, save_object, timeit
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)


class BaseTrainingPipeline(ABC):
    """Defines the invariant sequence of a training run:
    prepare data -> build features -> train candidate models ->
    evaluate them -> select the best -> persist artifacts.
    """

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        logger.info("=== Training pipeline started ===")
        train_df, val_df = self.prepare_data(train_df, val_df)
        X_train, X_val, y_train, y_val = self.build_features(train_df, val_df)
        fitted_models = self.train_models(X_train, y_train)
        results = self.evaluate_models(fitted_models, X_val, y_val)
        best_name, best_pipeline = self.select_best(fitted_models, results)
        self.persist(fitted_models, best_name, results)
        logger.info("=== Training pipeline finished. Best model: %s ===", best_name)
        return {
            "results": results,
            "best_model": best_name,
        }

    @abstractmethod
    def prepare_data(self, train_df, val_df): ...

    @abstractmethod
    def build_features(self, train_df, val_df): ...

    @abstractmethod
    def train_models(self, X_train, y_train) -> dict[str, Any]: ...

    @abstractmethod
    def evaluate_models(self, fitted_models, X_val, y_val) -> dict[str, dict]: ...

    @abstractmethod
    def select_best(self, fitted_models, results): ...

    @abstractmethod
    def persist(self, fitted_models, best_name, results) -> None: ...


class ClassicalMLTrainingPipeline(BaseTrainingPipeline):
    """Concrete Template-Method implementation for classical scikit-learn
    models (Logistic Regression, Linear SVM, Multinomial NB, Random Forest, ...).
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        preprocessing_config: PreprocessingConfig,
        feature_config: FeatureConfig,
    ):
        self.mt_config = model_trainer_config
        self.pp_config = preprocessing_config
        self.ft_config = feature_config
        self.model_factory = ModelFactory(model_trainer_config)
        self.evaluator = ModelEvaluator(positive_label=model_trainer_config.positive_label)
        self._vectorizer = None

    def prepare_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        train_df = preprocess_dataframe(train_df, self.pp_config, text_column=self.mt_config.text_column)
        val_df = preprocess_dataframe(val_df, self.pp_config, text_column=self.mt_config.text_column)
        return train_df, val_df

    def build_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        self._vectorizer = VectorizerStrategyFactory.create(self.ft_config)
        X_train = self._vectorizer.fit_transform(train_df["clean_review"])
        X_val = self._vectorizer.transform(val_df["clean_review"])

        y_train = train_df[self.mt_config.target_column].map(LABEL_MAP).values
        y_val = val_df[self.mt_config.target_column].map(LABEL_MAP).values
        return X_train, X_val, y_train, y_val

    @timeit
    def train_models(self, X_train, y_train) -> dict[str, Any]:
        fitted = {}
        for name in self.mt_config.active_models:
            logger.info("Training model: %s", name)
            model = self.model_factory.create(name)
            model.fit(X_train, y_train)
            fitted[name] = model
        return fitted

    def evaluate_models(self, fitted_models: dict, X_val, y_val) -> dict[str, dict]:
        results = {}
        for name, model in fitted_models.items():
            metrics = self.evaluator.evaluate(model, X_val, y_val)
            results[name] = metrics
            logger.info("Model '%s' validation metrics: %s", name, metrics["summary"])
        return results

    def select_best(self, fitted_models: dict, results: dict):
        metric = self.mt_config.primary_metric
        best_name = max(results, key=lambda n: results[n]["summary"][metric])
        return best_name, fitted_models[best_name]

    def persist(self, fitted_models: dict, best_name: str, results: dict) -> None:
        self.mt_config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.mt_config.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Persist the shared vectorizer once.
        save_object(self.mt_config.artifacts_dir / "vectorizer.joblib", self._vectorizer)

        for name, model in fitted_models.items():
            save_object(self.mt_config.artifacts_dir / f"model_{name}.joblib", model)

        # Convenience copies pointing at the champion model for inference code.
        save_object(self.mt_config.artifacts_dir / "best_model.joblib", fitted_models[best_name])
        save_json(
            self.mt_config.artifacts_dir / "best_model_name.json",
            {"best_model": best_name, "primary_metric": self.mt_config.primary_metric},
        )

        summary_table = {name: r["summary"] for name, r in results.items()}
        save_json(self.mt_config.metrics_dir / "validation_metrics.json", summary_table)
        logger.info("All model artifacts and metrics saved under %s", self.mt_config.artifacts_dir)


def build_inference_pipeline(vectorizer, model) -> Pipeline:
    """Convenience wrapper combining a fitted vectorizer + model into a single
    scikit-learn Pipeline object for simpler downstream inference code."""
    return Pipeline([("vectorizer", vectorizer), ("model", model)])
