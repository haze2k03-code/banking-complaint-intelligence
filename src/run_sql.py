"""
Step 2: run every query in sql/analysis.sql, print results, save each as a CSV.

Usage:
    python src/run_sql.py            # run all queries
    python src/run_sql.py 04         # run queries whose name contains "04"
"""
import re
import sqlite3
import sys

import pandas as pd

from config import DB_PATH, SQL_FILE, SQL_OUT_DIR, ensure_dirs


def parse_queries(text: str) -> dict:
    """Split the file on '-- name: xyz' markers -> {name: sql}."""
    parts = re.split(r"^--\s*name:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    # parts = [preamble, name1, sql1, name2, sql2, ...]
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def main(filter_text=None):
    ensure_dirs()
    queries = parse_queries(SQL_FILE.read_text(encoding="utf-8"))
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    with sqlite3.connect(DB_PATH) as con:
        for name, sql in queries.items():
            if filter_text and filter_text not in name:
                continue
            print(f"\n=== {name} ===")
            try:
                df = pd.read_sql_query(sql, con)
            except Exception as e:  # e.g. complaint_features table not built yet
                print(f"  skipped: {e}")
                continue
            print(df.head(15).to_string(index=False))
            df.to_csv(SQL_OUT_DIR / f"{name}.csv", index=False)

    print(f"\nResults saved to {SQL_OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
