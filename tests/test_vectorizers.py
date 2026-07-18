import pytest

from sentiment_analysis_project.entity.config_entity import FeatureConfig
from sentiment_analysis_project.features.vectorizers import VectorizerStrategyFactory


def _make_config(strategy: str) -> FeatureConfig:
    return FeatureConfig(
        strategy=strategy,
        max_features=500,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
    )


@pytest.mark.parametrize("strategy", ["tfidf", "count", "tfidf_char"])
def test_vectorizer_strategies_fit_transform(strategy):
    texts = ["great movie loved it", "terrible movie hated it", "an okay movie overall"]
    vec = VectorizerStrategyFactory.create(_make_config(strategy))
    X = vec.fit_transform(texts)
    assert X.shape[0] == 3
    assert X.shape[1] > 0


def test_unknown_vectorizer_strategy_raises():
    with pytest.raises(ValueError):
        VectorizerStrategyFactory.create(_make_config("not_real"))
