"""Shared pytest fixtures. Uses a small synthetic IMDB-like dataset so the
full test suite runs in seconds without needing the real Kaggle download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

POSITIVE_SAMPLES = [
    "This movie was absolutely fantastic, I loved every minute of it!",
    "A wonderful, heartwarming film with brilliant performances.",
    "Great direction and an amazing soundtrack, highly recommended.",
    "One of the best films I have seen this year, truly delightful.",
    "Superb acting and a gripping story from start to finish.",
    "I really enjoyed this movie, it was funny and touching.",
    "An excellent movie with a powerful and inspiring message.",
    "Beautifully shot and wonderfully acted, a real masterpiece.",
    "The plot was engaging and the characters were lovable.",
    "A joyous, uplifting experience that I would happily watch again.",
] * 3

NEGATIVE_SAMPLES = [
    "This movie was terrible, I wanted my two hours back.",
    "A boring, poorly written film with awful acting.",
    "Dreadful direction and a forgettable soundtrack, do not recommend.",
    "One of the worst films I have seen this year, truly disappointing.",
    "Weak acting and a confusing story from start to finish.",
    "I really disliked this movie, it was dull and tedious.",
    "A bad movie with a hollow and uninspired message.",
    "Poorly shot and badly acted, a real disappointment.",
    "The plot was incoherent and the characters were unlikeable.",
    "A miserable, depressing experience I would never watch again.",
] * 3


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    reviews = POSITIVE_SAMPLES + NEGATIVE_SAMPLES
    labels = ["positive"] * len(POSITIVE_SAMPLES) + ["negative"] * len(NEGATIVE_SAMPLES)
    return pd.DataFrame({"review": reviews, "sentiment": labels})
