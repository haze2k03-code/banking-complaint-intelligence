"""
Step 0: download the CFPB Consumer Complaint Database and build a manageable sample.

The full file is several GB, so we stream it in chunks and keep only:
  - complaints that have a written narrative (needed for NLP)
  - complaints received on/after --since
  - a random fraction of rows per chunk, then cap at --n rows

Usage:
    python src/download_data.py                  # download + sample 80,000 rows
    python src/download_data.py --skip-download  # reuse data/complaints.csv.zip
    python src/download_data.py --n 50000 --since 2023-01-01
"""
import argparse
import sys
import urllib.request

import pandas as pd

from config import CFPB_URL, COLUMN_MAP, RANDOM_STATE, RAW_ZIP, SAMPLE_CSV, ensure_dirs


def download(url=CFPB_URL, dest=RAW_ZIP):
    print(f"Downloading {url}\n  -> {dest}  (large file, this can take a while)")

    def progress(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 / total)
            sys.stdout.write(f"\r  {pct:5.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print("\nDownload complete.")


def build_sample(n=80_000, since="2022-01-01", keep_frac=0.15, chunksize=200_000):
    kept = []
    total_seen = 0
    reader = pd.read_csv(
        RAW_ZIP,
        compression="zip",
        chunksize=chunksize,
        usecols=list(COLUMN_MAP.keys()),
        dtype=str,
        low_memory=False,
    )
    for i, chunk in enumerate(reader):
        total_seen += len(chunk)
        chunk = chunk.rename(columns=COLUMN_MAP)
        chunk = chunk[chunk["narrative"].notna() & (chunk["narrative"].str.len() > 50)]
        chunk = chunk[chunk["date_received"] >= since]  # ISO dates compare correctly as strings
        if chunk.empty:
            continue
        kept.append(chunk.sample(frac=keep_frac, random_state=RANDOM_STATE + i))
        print(f"  chunk {i:3d}: seen {total_seen:,} rows, kept so far {sum(map(len, kept)):,}")

    if not kept:
        raise SystemExit("No rows matched the filters. Try an earlier --since date.")

    df = pd.concat(kept, ignore_index=True)
    if len(df) > n:
        df = df.sample(n=n, random_state=RANDOM_STATE)
    df.to_csv(SAMPLE_CSV, index=False)
    print(f"\nSaved {len(df):,} complaints -> {SAMPLE_CSV}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=80_000, help="rows to keep in the sample")
    p.add_argument("--since", default="2022-01-01", help="keep complaints received on/after this date")
    p.add_argument("--keep-frac", type=float, default=0.15, help="fraction kept from each chunk")
    p.add_argument("--skip-download", action="store_true", help="reuse an existing zip")
    args = p.parse_args()

    ensure_dirs()
    if not args.skip_download or not RAW_ZIP.exists():
        download()
    build_sample(n=args.n, since=args.since, keep_frac=args.keep_frac)
