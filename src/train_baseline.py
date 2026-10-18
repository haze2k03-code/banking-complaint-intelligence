"""
Step 4: baseline complaint-routing classifier — TF-IDF + Logistic Regression.

Task: read the customer's narrative, predict which product team should handle it.
Business value: auto-routing complaints instead of manual triage.

Outputs:
    outputs/models/baseline_tfidf_lr.joblib
    outputs/metrics/baseline.json          (accuracy, macro/weighted F1, latency)
    outputs/metrics/baseline_report.csv    (per-class precision/recall/F1)
    outputs/metrics/top_features.json      (most predictive words per class)
    outputs/metrics/test_ids.csv           (held-out ids, reused by zero_shot.py)
    outputs/figures/confusion_matrix.png

Usage:
    python src/train_baseline.py
"""
import json
import sqlite3
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import DB_PATH, FIG_DIR, METRICS_DIR, MODEL_DIR, RANDOM_STATE, ensure_dirs
from text_utils import clean_text


def load_data() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            "SELECT complaint_id, narrative, product FROM complaints WHERE product != 'Other'", con
        )
    # Drop classes too small to split/stratify reliably.
    counts = df["product"].value_counts()
    return df[df["product"].isin(counts[counts >= 20].index)]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    stop_words="english",
                    ngram_range=(1, 2),  # "late fee", "credit report"
                    min_df=3,
                    max_df=0.9,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0),
            ),
        ]
    )


def top_features(pipe: Pipeline, n=10) -> dict:
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    clf = pipe.named_steps["clf"]
    return {
        cls: vocab[np.argsort(clf.coef_[i])[::-1][:n]].tolist()
        for i, cls in enumerate(clf.classes_)
    }


def main():
    ensure_dirs()
    df = load_data()
    print(f"{len(df):,} complaints, {df['product'].nunique()} classes")

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        df["narrative"], df["product"], df["complaint_id"],
        test_size=0.2, stratify=df["product"], random_state=RANDOM_STATE,
    )
    pd.DataFrame({"complaint_id": id_test}).to_csv(METRICS_DIR / "test_ids.csv", index=False)

    # Sanity floor: always predicting the most common class.
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    pipe = build_pipeline()
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_secs = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = pipe.predict(X_test)
    ms_per_doc = (time.perf_counter() - t0) * 1000 / len(X_test)

    metrics = {
        "model": "TF-IDF (1-2 grams) + Logistic Regression",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_test, pred, average="weighted"), 4),
        "dummy_accuracy": round(accuracy_score(y_test, dummy_pred), 4),
        "dummy_macro_f1": round(f1_score(y_test, dummy_pred, average="macro"), 4),
        "train_seconds": round(train_secs, 1),
        "ms_per_complaint": round(ms_per_doc, 3),
    }
    (METRICS_DIR / "baseline.json").write_text(json.dumps(metrics, indent=2))

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.round(3).to_csv(METRICS_DIR / "baseline_report.csv")
    (METRICS_DIR / "top_features.json").write_text(json.dumps(top_features(pipe), indent=2))
    joblib.dump(pipe, MODEL_DIR / "baseline_tfidf_lr.joblib")

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, normalize="true", values_format=".2f",
        cmap="Blues", ax=ax, xticks_rotation=45, colorbar=False,
    )
    ax.set_title("Baseline: confusion matrix (row-normalised = recall per class)")
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)

    print("\n" + classification_report(y_test, pred, zero_division=0))
    print(json.dumps(metrics, indent=2))
    print("\nMost predictive words per class:")
    for cls, words in top_features(pipe, 6).items():
        print(f"  {cls:35s} {', '.join(words)}")


if __name__ == "__main__":
    main()
