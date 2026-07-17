#!/usr/bin/env python
"""Convenience entry point that runs the whole pipeline end-to-end:
download -> prepare (subset + split) -> EDA -> train -> evaluate.

Usage:
    uv run python main.py --subset dev
    uv run python main.py --subset medium
    uv run python main.py --subset full --skip-download
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full sentiment-analysis pipeline.")
    parser.add_argument("--subset", choices=["dev", "medium", "full"], default="dev")
    parser.add_argument("--skip-download", action="store_true", help="Skip the Kaggle download step.")
    parser.add_argument("--skip-eda", action="store_true")
    args = parser.parse_args()

    python = sys.executable

    if not args.skip_download:
        run([python, "scripts/download_data.py"])

    run([python, "scripts/prepare_data.py", "--subset", args.subset])

    if not args.skip_eda:
        run([python, "scripts/run_eda.py"])

    run([python, "scripts/run_train.py"])
    run([python, "scripts/run_eval.py"])

    print("\nPipeline complete. See artifacts/ for models, metrics, and plots.")


if __name__ == "__main__":
    main()
