"""
Step 5: compare the baseline with a transformer (zero-shot, no training).

A zero-shot NLI model scores how well the complaint matches a sentence like
"This complaint is about a mortgage." for each product, and picks the best.
No labelled data needed — but it is much slower, and usually less accurate than
a supervised model trained on in-domain labels. Measuring that trade-off IS the point.

Both models are scored on the SAME random subset of the held-out test set.

Needs:  pip install transformers torch   (CPU is fine; first run downloads the model)
Usage:
    python src/zero_shot.py                       # 300 test complaints, bart-large-mnli
    python src/zero_shot.py --n 100               # quicker
    python src/zero_shot.py --model valhalla/distilbart-mnli-12-1   # smaller, faster model
"""
import argparse
import json
import sqlite3
import time

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from config import DB_PATH, METRICS_DIR, MODEL_DIR, RANDOM_STATE

# Natural-language descriptions work better than raw label names for NLI models.
LABEL_TEXT = {
    "Credit reporting": "errors on a credit report or credit score",
    "Debt collection": "debt collection or a debt collector",
    "Credit card": "a credit card or prepaid card",
    "Checking/savings account": "a checking or savings bank account",
    "Mortgage": "a mortgage or home loan",
    "Student loan": "a student loan",
    "Vehicle loan": "a car loan or vehicle lease",
    "Money transfer / virtual currency": "a money transfer, payment app or cryptocurrency",
    "Payday / personal loan": "a payday loan or personal loan",
}


def main(n: int, model_name: str, max_words: int):
    from transformers import pipeline  # imported here so other scripts don't need torch

    test_ids = pd.read_csv(METRICS_DIR / "test_ids.csv")["complaint_id"]
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query("SELECT complaint_id, narrative, product FROM complaints", con)
    df = df[df["complaint_id"].isin(test_ids)]
    df = df.sample(n=min(n, len(df)), random_state=RANDOM_STATE)

    labels = [l for l in LABEL_TEXT if l in set(df["product"])]
    text_to_label = {LABEL_TEXT[l]: l for l in labels}

    # Long narratives exceed the model's input limit; the start usually states the problem.
    texts = [" ".join(t.split()[:max_words]) for t in df["narrative"]]

    print(f"Loading {model_name} ...")
    clf = pipeline("zero-shot-classification", model=model_name, device=-1)

    print(f"Classifying {len(texts)} complaints (CPU, this is slow) ...")
    t0 = time.perf_counter()
    outputs = clf(texts, candidate_labels=list(text_to_label), hypothesis_template="This complaint is about {}.")
    zs_ms = (time.perf_counter() - t0) * 1000 / len(texts)
    zs_pred = [text_to_label[o["labels"][0]] for o in outputs]

    baseline = joblib.load(MODEL_DIR / "baseline_tfidf_lr.joblib")
    t0 = time.perf_counter()
    bl_pred = baseline.predict(df["narrative"])
    bl_ms = (time.perf_counter() - t0) * 1000 / len(texts)

    y = df["product"]
    result = {
        "n_complaints": len(df),
        "zero_shot_model": model_name,
        "zero_shot": {
            "accuracy": round(accuracy_score(y, zs_pred), 4),
            "macro_f1": round(f1_score(y, zs_pred, average="macro", zero_division=0), 4),
            "ms_per_complaint": round(zs_ms, 1),
        },
        "baseline_same_subset": {
            "accuracy": round(accuracy_score(y, bl_pred), 4),
            "macro_f1": round(f1_score(y, bl_pred, average="macro", zero_division=0), 4),
            "ms_per_complaint": round(bl_ms, 3),
        },
    }
    (METRICS_DIR / "zero_shot.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame({"complaint_id": df["complaint_id"], "true": y, "baseline": bl_pred, "zero_shot": zs_pred}).to_csv(
        METRICS_DIR / "zero_shot_predictions.csv", index=False
    )
    print(json.dumps(result, indent=2))
    speed = zs_ms / max(bl_ms, 1e-6)
    print(f"\nZero-shot is ~{speed:,.0f}x slower per complaint than the baseline.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--model", default="facebook/bart-large-mnli")
    p.add_argument("--max-words", type=int, default=300)
    a = p.parse_args()
    main(a.n, a.model, a.max_words)
