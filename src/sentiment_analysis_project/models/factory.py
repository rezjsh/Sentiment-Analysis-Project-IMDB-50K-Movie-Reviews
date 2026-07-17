"""Factory pattern: creates fully-configured, ready-to-fit model instances
purely from configuration — training code never hardcodes `LogisticRegression(...)`
or any other estimator directly.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV

from sentiment_analysis_project.entity.config_entity import ModelTrainerConfig
from sentiment_analysis_project.models.registry import ModelRegistry
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)

# Models whose native decision_function output has no probability calibration,
# so we wrap them for ROC-AUC / confidence-score support.
_NEEDS_CALIBRATION = {"linear_svm"}


class ModelFactory:
    """Builds one or many models by name using ModelTrainerConfig.model_params."""

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    # Estimators that do NOT accept a random_state kwarg.
    _NO_RANDOM_STATE = {"multinomial_nb"}

    def create(self, model_name: str):
        params = dict(self.config.model_params.get(model_name, {}))
        if model_name not in self._NO_RANDOM_STATE:
            params.setdefault("random_state", self.config.random_state)

        constructor = ModelRegistry.get(model_name)
        estimator = constructor(**params)

        if model_name in _NEEDS_CALIBRATION:
            logger.info(
                "Wrapping '%s' with CalibratedClassifierCV to enable predict_proba.", model_name
            )
            estimator = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)

        logger.info("Created model '%s' with params=%s", model_name, params)
        return estimator

    def create_all(self) -> dict:
        return {name: self.create(name) for name in self.config.active_models}
