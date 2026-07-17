"""Small, dependency-light utility helpers shared across the whole pipeline."""

from __future__ import annotations

import json
import time
import functools
from pathlib import Path
from typing import Any, Callable

import joblib
import yaml

from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigBox(dict):
    """A minimal dot-accessible dict (avoids an extra third-party dependency
    such as `python-box` while giving the same ergonomic ``cfg.key.subkey`` access).
    Nested dicts are recursively wrapped.
    """

    def __init__(self, data: dict | None = None):
        super().__init__(data or {})
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigBox(value)
            elif isinstance(value, list):
                self[key] = [ConfigBox(v) if isinstance(v, dict) else v for v in value]

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def read_yaml(path: Path) -> ConfigBox:
    """Read a YAML file and return its content as a dot-accessible ConfigBox object."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
        if content is None:
            content = {}
        logger.debug("YAML file loaded: %s", path)
        return ConfigBox(content)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"YAML config not found at {path}") from exc


def create_directories(paths: list[Path | str], verbose: bool = True) -> None:
    """Create a list of directories if they do not already exist."""
    for p in paths:
        p = Path(p)
        p.mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.debug("Directory ensured: %s", p)


def save_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug("JSON saved at: %s", path)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_object(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Object persisted at: %s", path)


def load_object(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No object found at {path}")
    return joblib.load(path)


def get_size(path: Path) -> str:
    """Return a human-readable size (KB) of a file."""
    size_kb = round(Path(path).stat().st_size / 1024)
    return f"~ {size_kb} KB"


def timeit(func: Callable) -> Callable:
    """Decorator that logs execution time of the wrapped function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("'%s' completed in %.2fs", func.__name__, elapsed)
        return result

    return wrapper
