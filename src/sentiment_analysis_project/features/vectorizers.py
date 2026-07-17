"""Strategy pattern for turning cleaned text into numeric features.

Adding a new vectorization strategy means adding one class + one registry
entry — no other code changes.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sentiment_analysis_project.entity.config_entity import FeatureConfig
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)


class VectorizerStrategyFactory:
    """Builds a scikit-learn vectorizer instance from a FeatureConfig."""

    @staticmethod
    def create(config: FeatureConfig):
        strategy = config.strategy

        if strategy == "tfidf":
            vectorizer = TfidfVectorizer(
                max_features=config.max_features,
                ngram_range=config.ngram_range,
                min_df=config.min_df,
                max_df=config.max_df,
                sublinear_tf=config.sublinear_tf,
                analyzer="word",
            )
        elif strategy == "count":
            vectorizer = CountVectorizer(
                max_features=config.max_features,
                ngram_range=config.ngram_range,
                min_df=config.min_df,
                max_df=config.max_df,
                analyzer="word",
            )
        elif strategy == "tfidf_char":
            vectorizer = TfidfVectorizer(
                max_features=config.max_features,
                ngram_range=(3, 5),
                min_df=config.min_df,
                max_df=config.max_df,
                sublinear_tf=config.sublinear_tf,
                analyzer="char_wb",
            )
        else:
            raise ValueError(
                f"Unknown feature strategy '{strategy}'. "
                f"Available: tfidf, count, tfidf_char"
            )

        logger.info("Using vectorizer strategy: %s", strategy)
        return vectorizer
