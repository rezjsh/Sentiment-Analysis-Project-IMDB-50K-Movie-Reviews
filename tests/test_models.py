import pytest

from sentiment_analysis_project.entity.config_entity import ModelTrainerConfig
from sentiment_analysis_project.models.factory import ModelFactory
from sentiment_analysis_project.models.registry import ModelRegistry


def _make_trainer_config(tmp_path) -> ModelTrainerConfig:
    return ModelTrainerConfig(
        artifacts_dir=tmp_path / "artifacts",
        metrics_dir=tmp_path / "metrics",
        target_column="sentiment",
        text_column="review",
        positive_label="positive",
        negative_label="negative",
        cv_folds=3,
        active_models=["logistic_regression", "multinomial_nb"],
        model_params={
            "logistic_regression": {"C": 1.0, "max_iter": 500},
            "multinomial_nb": {"alpha": 1.0},
        },
        random_state=42,
        primary_metric="f1",
    )


def test_registry_lists_all_expected_models():
    expected = {"logistic_regression", "linear_svm", "multinomial_nb", "random_forest", "sgd_linear"}
    assert expected.issubset(set(ModelRegistry.available_models()))


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        ModelRegistry.get("not_a_real_model")


def test_factory_creates_configured_models(tmp_path):
    config = _make_trainer_config(tmp_path)
    factory = ModelFactory(config)

    lr = factory.create("logistic_regression")
    assert lr.C == 1.0
    assert lr.max_iter == 500

    nb = factory.create("multinomial_nb")
    assert nb.alpha == 1.0
    assert not hasattr(nb, "random_state")  # MultinomialNB has no random_state


def test_factory_wraps_linear_svm_for_calibration(tmp_path):
    config = _make_trainer_config(tmp_path)
    config.model_params["linear_svm"] = {"C": 1.0, "max_iter": 1000}
    config.active_models.append("linear_svm")

    factory = ModelFactory(config)
    svm = factory.create("linear_svm")
    assert hasattr(svm, "predict_proba")  # CalibratedClassifierCV adds this
