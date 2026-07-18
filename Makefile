.PHONY: setup download-data prepare-dev prepare-medium prepare-full eda train eval infer run-app test clean

setup:
	uv sync

download-data:
	uv run python scripts/download_data.py

prepare-dev:
	uv run python scripts/prepare_data.py --subset dev

prepare-medium:
	uv run python scripts/prepare_data.py --subset medium

prepare-full:
	uv run python scripts/prepare_data.py --subset full

eda:
	uv run python scripts/run_eda.py

train:
	uv run python scripts/run_train.py

eval:
	uv run python scripts/run_eval.py

infer:
	uv run python scripts/run_infer.py --text "This movie was absolutely wonderful!"

run-app:
	uv run streamlit run app/streamlit_app.py

test:
	uv run pytest -v

clean:
	rm -rf artifacts/models/* artifacts/metrics/* artifacts/plots/* logs/*.log
