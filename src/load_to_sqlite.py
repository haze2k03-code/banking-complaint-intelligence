"""
Step 1: clean the sample and load it into SQLite.

Cleaning decisions (worth explaining in an interview):
  * CFPB renamed product categories over the years (e.g. three different names
    for credit reporting). We map them to 9 stable `product` groups so trends
    and the classifier aren't split across duplicate labels.
  * Yes/No columns become 1/0 integers so SQL can average them into rates.
  * A `days_to_company` column measures how fast CFPB forwarded the complaint.

Usage:
    python src/load_to_sqlite.py
    python src/load_to_sqlite.py --csv data/some_other_file.csv
"""
import argparse
import sqlite3

import pandas as pd

from config import DB_PATH, SAMPLE_CSV, ensure_dirs

# Order matters: first matching keyword wins.
PRODUCT_RULES = [
    ("credit reporting", "Credit reporting"),
    ("debt collection", "Debt collection"),
    ("credit card", "Credit card"),
    ("prepaid card", "Credit card"),
    ("checking or savings", "Checking/savings account"),
    ("bank account", "Checking/savings account"),
    ("mortgage", "Mortgage"),
    ("student loan", "Student loan"),
    ("vehicle loan", "Vehicle loan"),
    ("consumer loan", "Vehicle loan"),
    ("money transfer", "Money transfer / virtual currency"),
    ("virtual currency", "Money transfer / virtual currency"),
    ("payday loan", "Payday / personal loan"),
    ("personal loan", "Payday / personal loan"),
]


def map_product(raw: str) -> str:
    raw = str(raw).lower()
    for keyword, group in PRODUCT_RULES:
        if keyword in raw:
            return group
    return "Other"


def yes_no_to_int(series: pd.Series) -> pd.Series:
    return series.map({"Yes": 1, "No": 0})


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["product"] = df["product_raw"].map(map_product)
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df["date_sent_to_company"] = pd.to_datetime(df["date_sent_to_company"], errors="coerce")
    df = df.dropna(subset=["date_received", "narrative", "company"])

    df["days_to_company"] = (df["date_sent_to_company"] - df["date_received"]).dt.days
    df["year_month"] = df["date_received"].dt.strftime("%Y-%m")
    df["timely"] = yes_no_to_int(df["timely_response"])
    df["disputed"] = yes_no_to_int(df["consumer_disputed"])  # mostly null after 2017
    df["got_monetary_relief"] = (df["company_response"] == "Closed with monetary relief").astype(int)
    df["narrative_len"] = df["narrative"].str.split().str.len()

    # SQLite has no date type; ISO strings sort and compare correctly.
    df["date_received"] = df["date_received"].dt.strftime("%Y-%m-%d")
    df["date_sent_to_company"] = df["date_sent_to_company"].dt.strftime("%Y-%m-%d")
    df["complaint_id"] = pd.to_numeric(df["complaint_id"], errors="coerce").astype("Int64")
    return df.drop_duplicates(subset="complaint_id")


def load(df: pd.DataFrame):
    with sqlite3.connect(DB_PATH) as con:
        df.to_sql("complaints", con, if_exists="replace", index=False)
        con.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_product ON complaints(product);
            CREATE INDEX IF NOT EXISTS idx_company ON complaints(company);
            CREATE INDEX IF NOT EXISTS idx_month   ON complaints(year_month);
            """
        )
    print(f"Loaded {len(df):,} rows into {DB_PATH} (table: complaints)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(SAMPLE_CSV))
    args = p.parse_args()

    ensure_dirs()
    raw = pd.read_csv(args.csv, dtype=str)
    df = clean(raw)
    print("\nComplaints per product group:")
    print(df["product"].value_counts().to_string())
    load(df)
