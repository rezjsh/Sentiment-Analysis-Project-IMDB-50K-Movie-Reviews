"""ConfigurationManager: the single place that reads configs/config.yaml and
configs/params.yaml and turns them into the typed entities defined in
entity/config_entity.py. Every pipeline stage asks this class for its config
instead of parsing YAML itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sentiment_analysis_project.constants.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    PROJECT_ROOT,
)
from sentiment_analysis_project.entity.config_entity import (
    DataIngestionConfig,
    DataSplitConfig,
    EDAConfig,
    FeatureConfig,
    ModelTrainerConfig,
    PreprocessingConfig,
    SubsetConfig,
)
from sentiment_analysis_project.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_path: Path = CONFIG_FILE_PATH,
        params_path: Path = PARAMS_FILE_PATH,
    ) -> None:
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([PROJECT_ROOT / self.config.artifacts_root])

    def _abs(self, rel_path: str) -> Path:
        return PROJECT_ROOT / rel_path

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config.data_ingestion
        create_directories([self._abs(cfg.raw_data_dir)])
        return DataIngestionConfig(
            kaggle_dataset_slug=self.config.kaggle.dataset_slug,
            raw_data_dir=self._abs(cfg.raw_data_dir),
            raw_file_path=self._abs(cfg.raw_file_path),
            raw_csv_name=self.config.kaggle.raw_csv_name,
        )

    def get_subset_config(self, subset_name: str) -> SubsetConfig:
        if subset_name not in self.config.subsets:
            raise ValueError(
                f"Unknown subset '{subset_name}'. "
                f"Available subsets: {list(self.config.subsets.keys())}"
            )
        subset = self.config.subsets[subset_name]
        return SubsetConfig(
            name=subset_name,
            n_samples=subset.n_samples,
            description=subset.description,
        )

    def get_data_split_config(self) -> DataSplitConfig:
        cfg = self.config.data_ingestion
        create_directories([self._abs(cfg.processed_data_dir)])
        return DataSplitConfig(
            processed_data_dir=self._abs(cfg.processed_data_dir),
            train_file_path=self._abs(cfg.train_file_path),
            val_file_path=self._abs(cfg.val_file_path),
            test_file_path=self._abs(cfg.test_file_path),
            test_size=self.params.test_size,
            val_size=self.params.val_size,
            random_state=self.params.random_state,
        )

    def get_eda_config(self) -> EDAConfig:
        cfg = self.config.eda
        create_directories([self._abs(cfg.output_dir)])
        return EDAConfig(
            output_dir=self._abs(cfg.output_dir),
            top_n_words=cfg.top_n_words,
            top_n_ngrams=cfg.top_n_ngrams,
            ngram_range=tuple(cfg.ngram_range),
            generate_wordcloud=cfg.generate_wordcloud,
            sample_reviews_to_show=cfg.sample_reviews_to_show,
        )

    def get_preprocessing_config(self) -> PreprocessingConfig:
        p = self.params.preprocessing
        return PreprocessingConfig(
            strategy=p.strategy,
            lowercase=p.lowercase,
            remove_html=p.remove_html,
            remove_punctuation=p.remove_punctuation,
            remove_numbers=p.remove_numbers,
            normalize_whitespace=p.normalize_whitespace,
            remove_stopwords=p.remove_stopwords,
            lemmatize=p.lemmatize,
        )

    def get_feature_config(self) -> FeatureConfig:
        f = self.params.features
        return FeatureConfig(
            strategy=f.strategy,
            max_features=f.max_features,
            ngram_range=tuple(f.ngram_range),
            min_df=f.min_df,
            max_df=f.max_df,
            sublinear_tf=f.sublinear_tf,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        mt = self.config.model_trainer
        create_directories([self._abs(mt.artifacts_dir), self._abs(mt.metrics_dir)])
        model_params = {name: dict(self.params.models[name]) for name in self.params.models.active_models}
        return ModelTrainerConfig(
            artifacts_dir=self._abs(mt.artifacts_dir),
            metrics_dir=self._abs(mt.metrics_dir),
            target_column=mt.target_column,
            text_column=mt.text_column,
            positive_label=mt.positive_label,
            negative_label=mt.negative_label,
            cv_folds=mt.cv_folds,
            active_models=list(self.params.models.active_models),
            model_params=model_params,
            random_state=self.params.random_state,
            primary_metric=self.params.selection.primary_metric,
        )
