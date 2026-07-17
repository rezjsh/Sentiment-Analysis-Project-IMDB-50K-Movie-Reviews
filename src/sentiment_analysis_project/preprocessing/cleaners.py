"""Strategy pattern for text preprocessing.

Each concrete strategy implements `.clean(text) -> str`. The
`PreprocessingStrategyFactory` builds the right strategy from a
`PreprocessingConfig`, so training code never branches on config values
itself — it just asks the factory for "the configured strategy" and calls it.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sentiment_analysis_project.entity.config_entity import PreprocessingConfig
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)

_HTML_TAG_RE = re.compile(r"<.*?>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_NUMBER_RE = re.compile(r"\d+")
_WHITESPACE_RE = re.compile(r"\s+")


def _ensure_nltk_resources() -> None:
    """Download the small NLTK corpora needed for stopwords/lemmatization,
    but only once, and quietly if they're already present."""
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, pkg in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", pkg)
            nltk.download(pkg, quiet=True)


class BaseTextCleaner(ABC):
    """Abstract base for all preprocessing strategies."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    @abstractmethod
    def clean(self, text: str) -> str:
        raise NotImplementedError

    def clean_batch(self, texts: list[str]) -> list[str]:
        return [self.clean(t) for t in texts]

    # --- shared building blocks, reused by subclasses ---
    def _strip_html_and_urls(self, text: str) -> str:
        text = _HTML_TAG_RE.sub(" ", text)
        text = _URL_RE.sub(" ", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        return _WHITESPACE_RE.sub(" ", text).strip()


class MinimalCleaner(BaseTextCleaner):
    """Lowercase + HTML/URL removal + whitespace normalization only.
    Preserves punctuation and stopwords — useful as a fast baseline or when
    a downstream model (e.g. a transformer) prefers close-to-raw text.
    """

    def clean(self, text: str) -> str:
        text = str(text)
        if self.config.remove_html:
            text = self._strip_html_and_urls(text)
        if self.config.lowercase:
            text = text.lower()
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)
        return text


class StandardCleaner(BaseTextCleaner):
    """The default, recommended strategy for classical ML models:
    HTML/URL removal, lowercasing, punctuation stripping, optional stopword
    removal and lemmatization.
    """

    def __init__(self, config: PreprocessingConfig):
        super().__init__(config)
        if config.remove_stopwords or config.lemmatize:
            _ensure_nltk_resources()
        self._stopwords = set(stopwords.words("english")) if config.remove_stopwords else set()
        self._lemmatizer = WordNetLemmatizer() if config.lemmatize else None

    def clean(self, text: str) -> str:
        text = str(text)
        if self.config.remove_html:
            text = self._strip_html_and_urls(text)
        if self.config.lowercase:
            text = text.lower()
        if self.config.remove_numbers:
            text = _NUMBER_RE.sub(" ", text)
        if self.config.remove_punctuation:
            text = _PUNCT_RE.sub(" ", text)
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)

        if self._stopwords or self._lemmatizer:
            tokens = text.split()
            if self._stopwords:
                tokens = [t for t in tokens if t not in self._stopwords]
            if self._lemmatizer:
                tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
            text = " ".join(tokens)

        return text


class AggressiveCleaner(StandardCleaner):
    """Everything `StandardCleaner` does, plus: forces number removal and
    drops very short tokens (length < 3), which can help reduce noise for
    high-bias linear models on very large vocabularies.
    """

    def clean(self, text: str) -> str:
        text = str(text)
        if self.config.remove_html:
            text = self._strip_html_and_urls(text)
        if self.config.lowercase:
            text = text.lower()
        text = _NUMBER_RE.sub(" ", text)  # always strip numbers, regardless of config
        if self.config.remove_punctuation:
            text = _PUNCT_RE.sub(" ", text)
        text = self._normalize_whitespace(text)

        tokens = text.split()
        tokens = [t for t in tokens if len(t) >= 3]
        if self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        if self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)


class PreprocessingStrategyFactory:
    """Factory that builds the configured cleaning strategy by name."""

    _REGISTRY = {
        "minimal": MinimalCleaner,
        "standard": StandardCleaner,
        "aggressive": AggressiveCleaner,
    }

    @classmethod
    def create(cls, config: PreprocessingConfig) -> BaseTextCleaner:
        strategy_cls = cls._REGISTRY.get(config.strategy)
        if strategy_cls is None:
            raise ValueError(
                f"Unknown preprocessing strategy '{config.strategy}'. "
                f"Available: {list(cls._REGISTRY.keys())}"
            )
        logger.info("Using preprocessing strategy: %s", config.strategy)
        return strategy_cls(config)
