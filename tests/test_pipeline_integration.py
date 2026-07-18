"""End-to-end smoke test: raw dataframe -> split -> preprocess -> vectorize
-> train -> evaluate -> predict, all using the tiny synthetic dataset so it
runs in a couple of seconds. This is the test that would catch a broken
import chain or a wiring mistake between modules.
"""

from __future__ import annotations

from sentiment_analysis_project.entity.config_entity import (
    DataSplitConfig,
    FeatureConfig,
    ModelTrainerConfig,
    PreprocessingConfig,
)
from sentiment_analysis_project.data.splitter import split_dataset
from sentiment_analysis_project.train.trainer import ClassicalMLTrainingPipeline


def test_full_pipeline_runs_end_to_end(tmp_path, synthetic_df):
    split_cfg = DataSplitConfig(
        processed_data_dir=tmp_path / "processed",
        train_file_path=tmp_path / "processed" / "train.csv",
        val_file_path=tmp_path / "processed" / "val.csv",
        test_file_path=tmp_path / "processed" / "test.csv",
        test_size=0.2,
        val_size=0.2,
        random_state=42,
    )
    splits = split_dataset(synthetic_df, split_cfg)
    assert set(splits.keys()) == {"train", "val", "test"}
    assert len(splits["train"]) > 0

    pp_cfg = PreprocessingConfig(
        strategy="standard",
        lowercase=True,
        remove_html=True,
        remove_punctuation=True,
        remove_numbers=False,
        normalize_whitespace=True,
        remove_stopwords=True,
        lemmatize=False,
    )
    ft_cfg = FeatureConfig(
        strategy="tfidf",
        max_features=500,
        ngram_range=(1, 1),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
    )
    mt_cfg = ModelTrainerConfig(
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

    pipeline = ClassicalMLTrainingPipeline(mt_cfg, pp_cfg, ft_cfg)
    outcome = pipeline.run(splits["train"], splits["val"])

    assert outcome["best_model"] in mt_cfg.active_models
    assert (mt_cfg.artifacts_dir / "best_model.joblib").exists()
    assert (mt_cfg.artifacts_dir / "vectorizer.joblib").exists()
    assert (mt_cfg.metrics_dir / "validation_metrics.json").exists()

    for name in mt_cfg.active_models:
        assert 0.0 <= outcome["results"][name]["summary"]["f1"] <= 1.0
