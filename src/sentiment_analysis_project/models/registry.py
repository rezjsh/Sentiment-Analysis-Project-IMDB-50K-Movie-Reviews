"""Registry pattern: a central, extensible mapping of model-name -> constructor.

To add a brand-new classical model to the whole project, you only need to:
  1. Register it here with `@ModelRegistry.register("my_model")`.
  2. Add a `my_model:` block of hyperparameters to configs/params.yaml.
  3. Add "my_model" to `models.active_models` in configs/params.yaml.
No other file needs to change — this is what makes the model factory
"config-driven" as required by the project spec.
"""

from __future__ import annotations

from typing import Callable, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


class ModelRegistry:
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(constructor: Callable):
            cls._registry[name] = constructor
            return constructor

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        if name not in cls._registry:
            raise ValueError(
                f"Model '{name}' is not registered. Available models: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def available_models(cls) -> list[str]:
        return list(cls._registry.keys())


@ModelRegistry.register("logistic_regression")
def _build_logistic_regression(**params):
    return LogisticRegression(**params)


@ModelRegistry.register("linear_svm")
def _build_linear_svm(**params):
    # LinearSVC has no predict_proba; the trainer wraps it with CalibratedClassifierCV
    # when probability estimates are required (see train/trainer.py).
    return LinearSVC(**params)


@ModelRegistry.register("multinomial_nb")
def _build_multinomial_nb(**params):
    return MultinomialNB(**params)


@ModelRegistry.register("random_forest")
def _build_random_forest(**params):
    return RandomForestClassifier(**params)


@ModelRegistry.register("sgd_linear")
def _build_sgd_linear(**params):
    return SGDClassifier(**params)
