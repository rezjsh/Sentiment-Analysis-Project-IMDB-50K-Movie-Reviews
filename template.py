"""Idempotent project-scaffolding script — an improved, corrected version of
the original template.py.

Key changes vs. the original template:
  - PROJECT_SLUG, directories, and files now match a real sentiment-analysis
    package (the original template mixed in credit-card-fraud/SMOTE naming
    left over from a different project).
  - Uses a Strategy/Factory/Template-Method/Registry-friendly module layout
    (config, constants, entity, utils, data, eda, preprocessing, features,
    models, train, eval, infer, visualize) instead of a generic
    "components/pipeline" layout, matching the design patterns actually used.
  - Adds uv-specific root files (pyproject.toml) instead of setup.py /
    requirements.txt, and drops the Dockerfile/MLflow config that aren't
    part of this project's scope.
  - Still safe to re-run: only creates missing files/directories and never
    overwrites existing content.

Run with:
    uv run python template.py
"""

from __future__ import annotations

import pathlib
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PROJECT_SLUG = "sentiment_analysis_project"

dirs_to_create = [
    "configs",
    "data/raw",
    "data/processed",
    "docs",
    "logs",
    "notebooks",
    "artifacts/models",
    "artifacts/plots/eda",
    "artifacts/metrics",
    "scripts",
    "tests",
    "app",
    f"src/{PROJECT_SLUG}",
    f"src/{PROJECT_SLUG}/config",
    f"src/{PROJECT_SLUG}/constants",
    f"src/{PROJECT_SLUG}/entity",
    f"src/{PROJECT_SLUG}/utils",
    f"src/{PROJECT_SLUG}/data",
    f"src/{PROJECT_SLUG}/eda",
    f"src/{PROJECT_SLUG}/preprocessing",
    f"src/{PROJECT_SLUG}/features",
    f"src/{PROJECT_SLUG}/models",
    f"src/{PROJECT_SLUG}/train",
    f"src/{PROJECT_SLUG}/eval",
    f"src/{PROJECT_SLUG}/infer",
    f"src/{PROJECT_SLUG}/visualize",
]

files_to_create = [
    # Config files
    "configs/config.yaml",
    "configs/params.yaml",
    "configs/logging_config.yaml",
    # Root files
    "README.md",
    "pyproject.toml",
    ".gitignore",
    "main.py",
    "Makefile",
    ".env.example",
    # Package init files
    f"src/{PROJECT_SLUG}/__init__.py",
    f"src/{PROJECT_SLUG}/config/__init__.py",
    f"src/{PROJECT_SLUG}/constants/__init__.py",
    f"src/{PROJECT_SLUG}/entity/__init__.py",
    f"src/{PROJECT_SLUG}/utils/__init__.py",
    f"src/{PROJECT_SLUG}/data/__init__.py",
    f"src/{PROJECT_SLUG}/eda/__init__.py",
    f"src/{PROJECT_SLUG}/preprocessing/__init__.py",
    f"src/{PROJECT_SLUG}/features/__init__.py",
    f"src/{PROJECT_SLUG}/models/__init__.py",
    f"src/{PROJECT_SLUG}/train/__init__.py",
    f"src/{PROJECT_SLUG}/eval/__init__.py",
    f"src/{PROJECT_SLUG}/infer/__init__.py",
    f"src/{PROJECT_SLUG}/visualize/__init__.py",
    # Core modules
    f"src/{PROJECT_SLUG}/constants/constants.py",
    f"src/{PROJECT_SLUG}/entity/config_entity.py",
    f"src/{PROJECT_SLUG}/config/configuration.py",
    f"src/{PROJECT_SLUG}/utils/common.py",
    f"src/{PROJECT_SLUG}/utils/logger.py",
    f"src/{PROJECT_SLUG}/data/kaggle_downloader.py",
    f"src/{PROJECT_SLUG}/data/loader.py",
    f"src/{PROJECT_SLUG}/data/splitter.py",
    f"src/{PROJECT_SLUG}/eda/eda.py",
    f"src/{PROJECT_SLUG}/preprocessing/cleaners.py",
    f"src/{PROJECT_SLUG}/preprocessing/pipeline.py",
    f"src/{PROJECT_SLUG}/features/vectorizers.py",
    f"src/{PROJECT_SLUG}/models/registry.py",
    f"src/{PROJECT_SLUG}/models/factory.py",
    f"src/{PROJECT_SLUG}/train/trainer.py",
    f"src/{PROJECT_SLUG}/eval/evaluator.py",
    f"src/{PROJECT_SLUG}/infer/predictor.py",
    f"src/{PROJECT_SLUG}/visualize/plots.py",
    # CLI scripts
    "scripts/download_data.py",
    "scripts/prepare_data.py",
    "scripts/run_eda.py",
    "scripts/run_train.py",
    "scripts/run_eval.py",
    "scripts/run_infer.py",
    # Tests
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/test_preprocessing.py",
    "tests/test_vectorizers.py",
    "tests/test_models.py",
    "tests/test_evaluator.py",
    "tests/test_pipeline_integration.py",
    # App
    "app/streamlit_app.py",
]

gitignore_content = """
# Standard Python ignores
__pycache__/
*.py[cod]
*.so

# Environments
.env
.venv
env/
venv/

# uv
.uv/

# Data and Logs
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
logs/
*.log

# Models/artifacts
artifacts/
!artifacts/.gitkeep

# IDE files
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints
"""


def main() -> None:
    logging.info("Starting project structure creation for package: %s", PROJECT_SLUG)

    for dir_path_str in dirs_to_create:
        path = pathlib.Path(dir_path_str)
        try:
            path.mkdir(parents=True, exist_ok=True)
            logging.info("Created directory (or verified exists): %s", path)
        except OSError as e:
            logging.error("Failed to create directory %s: %s", path, e)

    for file_path_str in files_to_create:
        file_path = pathlib.Path(file_path_str)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            try:
                file_path.touch()
                logging.info("Created empty file (fill in with project source): %s", file_path)
            except Exception as e:
                logging.error("Failed to create file %s: %s", file_path, e)
                continue

    gitignore_path = pathlib.Path(".gitignore")
    try:
        existing_lines = set()
        if gitignore_path.exists():
            with open(gitignore_path, "r", encoding="utf-8") as f:
                existing_lines = set(line.strip() for line in f.read().splitlines() if line.strip())

        new_lines = [line.strip() for line in gitignore_content.strip().splitlines() if line.strip()]
        lines_to_add = [line for line in new_lines if line not in existing_lines]

        if lines_to_add:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write("\n".join(lines_to_add))
            logging.info("Updated .gitignore with %d new lines.", len(lines_to_add))
        else:
            logging.info(".gitignore file is up-to-date.")
    except Exception as e:
        logging.error("Failed to handle .gitignore: %s", e)

    logging.info("Project structure creation process finished successfully.")


if __name__ == "__main__":
    main()
