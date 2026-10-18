"""
Step 3: rule-based (regex) feature extraction from complaint narratives.

Creates SQLite table `complaint_features`:
    complaint_id, n_amounts, max_amount, total_amount,
    fraud_flag, fee_flag, legal_flag, masked_dates

Why regex before ML: it's cheap, explainable, and gives business-ready signals
(fraud mentions, money at stake, legal escalation risk) without any training.

Usage:
    python src/regex_extract.py
"""
import sqlite3

import pandas as pd

from config import DB_PATH, ensure_dirs
from text_utils import FEE_RE, FRAUD_RE, LEGAL_RE, MASKED_DATE_RE, parse_amounts

# Ignore absurd values (typos, account numbers typed with a $).
MAX_PLAUSIBLE_AMOUNT = 1_000_000


def extract(row_text: str) -> dict:
    amounts = [a for a in parse_amounts(row_text) if 0 < a <= MAX_PLAUSIBLE_AMOUNT]
    return {
        "n_amounts": len(amounts),
        "max_amount": max(amounts) if amounts else 0.0,
        "total_amount": sum(amounts),
        "fraud_flag": int(bool(FRAUD_RE.search(row_text))),
        "fee_flag": int(bool(FEE_RE.search(row_text))),
        "legal_flag": int(bool(LEGAL_RE.search(row_text))),
        "masked_dates": len(MASKED_DATE_RE.findall(row_text)),
    }


def main():
    ensure_dirs()
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query("SELECT complaint_id, product, narrative FROM complaints", con)
        feats = pd.DataFrame([extract(t) for t in df["narrative"].fillna("")])
        feats.insert(0, "complaint_id", df["complaint_id"].values)
        feats.to_sql("complaint_features", con, if_exists="replace", index=False)

    merged = feats.assign(product=df["product"].values)
    print(f"Extracted features for {len(feats):,} complaints -> table complaint_features\n")
    print("Share of complaints with each signal:")
    print((feats[["fraud_flag", "fee_flag", "legal_flag"]].mean() * 100).round(1).to_string())
    print(f"mentions a $ amount: {(feats['n_amounts'] > 0).mean() * 100:.1f}%")

    print("\nMedian largest amount mentioned, by product (complaints with an amount):")
    with_amt = merged[merged["max_amount"] > 0]
    print(with_amt.groupby("product")["max_amount"].median().round(0).sort_values(ascending=False).to_string())

    print("\nExample fraud-flagged snippet:")
    sample = df[feats["fraud_flag"] == 1]["narrative"].head(1)
    for text in sample:
        m = FRAUD_RE.search(text)
        start = max(0, m.start() - 80)
        print("  ..." + text[start : m.end() + 80].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
