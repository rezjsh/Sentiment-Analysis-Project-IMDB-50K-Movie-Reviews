import numpy as np
from sklearn.linear_model import LogisticRegression

from sentiment_analysis_project.eval.evaluator import ModelEvaluator


def test_evaluator_computes_expected_metric_keys():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression().fit(X, y)
    evaluator = ModelEvaluator()
    result = evaluator.evaluate(model, X, y)

    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in result["summary"]
    assert len(result["confusion_matrix"]) == 2
    assert result["summary"]["accuracy"] > 0.5  # should beat random guessing


def test_comparison_table_sorted_by_f1():
    fake_results = {
        "model_a": {"summary": {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1": 0.80, "roc_auc": 0.85}},
        "model_b": {"summary": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.90, "roc_auc": 0.95}},
    }
    df = ModelEvaluator.comparison_table(fake_results)
    assert df.iloc[0]["model"] == "model_b"
