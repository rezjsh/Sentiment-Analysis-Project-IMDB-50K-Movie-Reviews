from sentiment_analysis_project.entity.config_entity import PreprocessingConfig
from sentiment_analysis_project.preprocessing.cleaners import PreprocessingStrategyFactory


def _make_config(strategy: str) -> PreprocessingConfig:
    return PreprocessingConfig(
        strategy=strategy,
        lowercase=True,
        remove_html=True,
        remove_punctuation=True,
        remove_numbers=False,
        normalize_whitespace=True,
        remove_stopwords=True,
        lemmatize=True,
    )


def test_minimal_cleaner_lowercases_and_strips_html():
    cleaner = PreprocessingStrategyFactory.create(_make_config("minimal"))
    out = cleaner.clean("<br/>This Movie WAS Great!!")
    assert out == "this movie was great!!"


def test_standard_cleaner_removes_punctuation_and_stopwords():
    cleaner = PreprocessingStrategyFactory.create(_make_config("standard"))
    out = cleaner.clean("<p>This movie was NOT good at all.</p>")
    assert "<p>" not in out
    assert "." not in out
    assert out == out.lower()


def test_aggressive_cleaner_drops_short_tokens():
    cleaner = PreprocessingStrategyFactory.create(_make_config("aggressive"))
    out = cleaner.clean("I am go to a it 123 movie")
    tokens = out.split()
    assert all(len(t) >= 3 for t in tokens)


def test_unknown_strategy_raises():
    import pytest

    bad_config = _make_config("does_not_exist")
    with pytest.raises(ValueError):
        PreprocessingStrategyFactory.create(bad_config)
