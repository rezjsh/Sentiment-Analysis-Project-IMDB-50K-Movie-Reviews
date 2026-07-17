"""Interactive Streamlit app for the IMDB sentiment classifier.

Run with:
    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import wordnet

# Make `src/` importable when this file is run directly by Streamlit.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sentiment_analysis_project.config.configuration import ConfigurationManager  # noqa: E402
from sentiment_analysis_project.infer.predictor import SentimentPredictor  # noqa: E402
from sentiment_analysis_project.utils.common import load_json  # noqa: E402

st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner="Loading model artifacts ...")
def get_predictor():
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    wordnet.ensure_loaded()
    cm = ConfigurationManager()
    mt_cfg = cm.get_model_trainer_config()
    pp_cfg = cm.get_preprocessing_config()
    predictor = SentimentPredictor(mt_cfg, pp_cfg, use_best=True)
    return predictor, mt_cfg


st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.caption("Classical ML pipeline trained on the IMDB Dataset of 50K Movie Reviews.")

try:
    predictor, mt_cfg = get_predictor()
    best_info = load_json(mt_cfg.artifacts_dir / "best_model_name.json")
    st.success(f"Champion model loaded: **{best_info['best_model']}**")
except FileNotFoundError:
    st.error(
        "No trained model artifacts found yet. Run the training pipeline first:\n\n"
        "```\nuv run python scripts/prepare_data.py --subset medium\n"
        "uv run python scripts/run_train.py\n```"
    )
    st.stop()

tab_single, tab_batch = st.tabs(["Single review", "Batch (CSV) prediction"])

with tab_single:
    text = st.text_area(
        "Paste a movie review:",
        height=180,
        placeholder="This film had incredible pacing and the performances were unforgettable...",
    )
    if st.button("Classify sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            result = predictor.predict_text(text)
            label = result["predicted_label"]
            conf = result["confidence"]
            emoji = "😊" if label == "positive" else "☹️"
            st.metric("Predicted sentiment", f"{emoji} {label.title()}")
            if conf is not None:
                st.progress(conf if label == "positive" else 1 - conf)
                st.caption(f"Confidence (positive-class probability): {conf:.2%}")

with tab_batch:
    uploaded = st.file_uploader("Upload a CSV with a 'review' column", type=["csv"])
    text_col = st.text_input("Text column name", value="review")
    if uploaded is not None and st.button("Run batch prediction"):
        df = pd.read_csv(uploaded)
        if text_col not in df.columns:
            st.error(f"Column '{text_col}' not found. Available columns: {list(df.columns)}")
        else:
            results = predictor.predict_batch(df[text_col].astype(str).tolist())
            merged = pd.concat([df.reset_index(drop=True), results[["predicted_label", "confidence"]]], axis=1)
            st.dataframe(merged, use_container_width=True)
            st.download_button(
                "Download predictions as CSV",
                merged.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
