"""Full Exploratory Data Analysis suite for the IMDB reviews dataset.

Covers: dataset overview, missing-value analysis, duplicate analysis, label
distribution, review length distribution, character/word count statistics,
top words, top n-grams, class-wise text length comparison, optional word
clouds, train/val/test split inspection, and sample review display. All
plots are exported as PNG files under `artifacts/plots/eda/` and a
machine-readable summary is written as JSON alongside a human-readable
Markdown report.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for servers/Colab/CI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from sentiment_analysis_project.entity.config_entity import EDAConfig
from sentiment_analysis_project.utils.common import save_json
from sentiment_analysis_project.utils.logger import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")

_HTML_TAG_RE = re.compile(r"<.*?>")
_WORD_RE = re.compile(r"\b[a-zA-Z]+\b")


class EDAReport:
    def __init__(self, config: EDAConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary: dict = {}

    def _savefig(self, fig, name: str) -> Path:
        path = self.config.output_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=140)
        plt.close(fig)
        logger.info("Saved EDA plot: %s", path)
        return path

    # ------------------------------------------------------------------ #
    # Individual analyses
    # ------------------------------------------------------------------ #
    def dataset_overview(self, df: pd.DataFrame) -> dict:
        overview = {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        }
        self.summary["dataset_overview"] = overview
        return overview

    def missing_value_analysis(self, df: pd.DataFrame) -> dict:
        missing_counts = df.isna().sum()
        missing_pct = (missing_counts / len(df) * 100).round(3)
        result = {
            "missing_counts": missing_counts.to_dict(),
            "missing_pct": missing_pct.to_dict(),
        }
        self.summary["missing_values"] = result
        return result

    def duplicate_analysis(self, df: pd.DataFrame, subset: str = "review") -> dict:
        n_dupes = int(df.duplicated(subset=[subset]).sum())
        result = {"duplicate_review_count": n_dupes, "duplicate_pct": round(n_dupes / len(df) * 100, 3)}
        self.summary["duplicates"] = result
        return result

    def label_distribution(self, df: pd.DataFrame, label_col: str = "sentiment") -> dict:
        counts = df[label_col].value_counts()
        pct = (counts / len(df) * 100).round(2)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="Set2", legend=False, ax=ax)
        ax.set_title("Label Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(v), ha="center", va="bottom")
        self._savefig(fig, "label_distribution")

        result = {"counts": counts.to_dict(), "pct": pct.to_dict()}
        self.summary["label_distribution"] = result
        return result

    def length_and_wordcount_stats(self, df: pd.DataFrame, text_col: str = "review") -> dict:
        char_len = df[text_col].astype(str).str.len()
        word_count = df[text_col].astype(str).str.split().str.len()

        stats = {
            "char_length": char_len.describe().round(2).to_dict(),
            "word_count": word_count.describe().round(2).to_dict(),
        }

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.histplot(char_len, bins=50, ax=axes[0], color="#4C72B0")
        axes[0].set_title("Review Character-Length Distribution")
        axes[0].set_xlabel("Characters")

        sns.histplot(word_count, bins=50, ax=axes[1], color="#55A868")
        axes[1].set_title("Review Word-Count Distribution")
        axes[1].set_xlabel("Words")
        self._savefig(fig, "length_wordcount_distribution")

        self.summary["length_wordcount_stats"] = stats
        return stats

    def class_wise_length_comparison(
        self, df: pd.DataFrame, text_col: str = "review", label_col: str = "sentiment"
    ) -> dict:
        df = df.copy()
        df["_word_count"] = df[text_col].astype(str).str.split().str.len()

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x=label_col, y="_word_count", hue=label_col, palette="Set3", legend=False, ax=ax)
        ax.set_title("Word Count by Sentiment Class")
        ax.set_ylabel("Word count")
        self._savefig(fig, "classwise_length_comparison")

        result = df.groupby(label_col)["_word_count"].describe().round(2).to_dict()
        self.summary["classwise_length_comparison"] = result
        return result

    def top_words(self, df: pd.DataFrame, text_col: str = "clean_review") -> dict:
        vec = CountVectorizer(stop_words="english", max_features=5000)
        X = vec.fit_transform(df[text_col].astype(str))
        freqs = X.sum(axis=0).A1
        vocab = vec.get_feature_names_out()
        top = sorted(zip(vocab, freqs), key=lambda x: -x[1])[: self.config.top_n_words]

        words, counts = zip(*top) if top else ([], [])
        fig, ax = plt.subplots(figsize=(7, max(4, len(words) * 0.28)))
        sns.barplot(x=list(counts), y=list(words), hue=list(words), palette="viridis", legend=False, ax=ax)
        ax.set_title(f"Top {len(words)} Words (stopwords removed)")
        ax.set_xlabel("Frequency")
        self._savefig(fig, "top_words")

        result = {w: int(c) for w, c in top}
        self.summary["top_words"] = result
        return result

    def top_ngrams(self, df: pd.DataFrame, text_col: str = "clean_review") -> dict:
        ngram_range = self.config.ngram_range
        vec = CountVectorizer(
            stop_words="english", ngram_range=ngram_range, max_features=5000
        )
        X = vec.fit_transform(df[text_col].astype(str))
        freqs = X.sum(axis=0).A1
        vocab = vec.get_feature_names_out()
        top = sorted(zip(vocab, freqs), key=lambda x: -x[1])[: self.config.top_n_ngrams]

        phrases, counts = zip(*top) if top else ([], [])
        fig, ax = plt.subplots(figsize=(7, max(4, len(phrases) * 0.3)))
        sns.barplot(x=list(counts), y=list(phrases), hue=list(phrases), palette="magma", legend=False, ax=ax)
        ax.set_title(f"Top {len(phrases)} N-grams {tuple(ngram_range)}")
        ax.set_xlabel("Frequency")
        self._savefig(fig, "top_ngrams")

        result = {p: int(c) for p, c in top}
        self.summary["top_ngrams"] = result
        return result

    def wordclouds(self, df: pd.DataFrame, text_col: str = "clean_review", label_col: str = "sentiment") -> None:
        if not self.config.generate_wordcloud:
            return
        try:
            from wordcloud import WordCloud
        except ImportError:
            logger.warning("wordcloud package not installed; skipping word cloud generation.")
            return

        for label in df[label_col].unique():
            text_blob = " ".join(df.loc[df[label_col] == label, text_col].astype(str).tolist())
            if not text_blob.strip():
                continue
            wc = WordCloud(width=900, height=500, background_color="white", max_words=150).generate(text_blob)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Word Cloud — {label}")
            self._savefig(fig, f"wordcloud_{label}")

    def split_inspection(self, splits: dict[str, pd.DataFrame], label_col: str = "sentiment") -> dict:
        result = {}
        for name, part in splits.items():
            result[name] = {
                "n_rows": int(len(part)),
                "label_balance": part[label_col].value_counts(normalize=True).round(3).to_dict(),
            }
        self.summary["split_inspection"] = result
        return result

    def sample_reviews(self, df: pd.DataFrame, text_col: str = "review", label_col: str = "sentiment") -> list[dict]:
        n = self.config.sample_reviews_to_show
        sample = df.sample(n=min(n, len(df)), random_state=42)
        rows = [
            {"sentiment": r[label_col], "review_snippet": str(r[text_col])[:300]}
            for _, r in sample.iterrows()
        ]
        self.summary["sample_reviews"] = rows
        return rows

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run_full_report(
        self,
        raw_df: pd.DataFrame,
        clean_df: pd.DataFrame | None = None,
        splits: dict[str, pd.DataFrame] | None = None,
    ) -> dict:
        logger.info("Running full EDA report ...")
        self.dataset_overview(raw_df)
        self.missing_value_analysis(raw_df)
        self.duplicate_analysis(raw_df)
        self.label_distribution(raw_df)
        self.length_and_wordcount_stats(raw_df)
        self.class_wise_length_comparison(raw_df)

        ngram_source = clean_df if clean_df is not None else raw_df
        text_col = "clean_review" if clean_df is not None else "review"
        self.top_words(ngram_source, text_col=text_col)
        self.top_ngrams(ngram_source, text_col=text_col)
        self.wordclouds(ngram_source, text_col=text_col)

        if splits is not None:
            self.split_inspection(splits)

        self.sample_reviews(raw_df)

        save_json(self.config.output_dir / "eda_summary.json", self.summary)
        self._write_markdown_report()
        logger.info("EDA report complete. Artifacts saved under %s", self.config.output_dir)
        return self.summary

    def _write_markdown_report(self) -> None:
        lines = ["# EDA Report — IMDB 50K Movie Reviews\n"]
        ov = self.summary.get("dataset_overview", {})
        lines.append(f"- **Rows:** {ov.get('n_rows')}  \n- **Columns:** {ov.get('columns')}\n")

        dup = self.summary.get("duplicates", {})
        lines.append(f"- **Duplicate reviews:** {dup.get('duplicate_review_count')} "
                     f"({dup.get('duplicate_pct')}%)\n")

        lbl = self.summary.get("label_distribution", {})
        lines.append(f"- **Label distribution:** {lbl.get('counts')}\n")

        lines.append("\n## Plots\n")
        for png in sorted(self.config.output_dir.glob("*.png")):
            lines.append(f"![{png.stem}]({png.name})\n")

        report_path = self.config.output_dir / "EDA_REPORT.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown EDA report written to %s", report_path)
