"""Shared paths and settings for the whole project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_ZIP = DATA_DIR / "complaints.csv.zip"
SAMPLE_CSV = DATA_DIR / "complaints_sample.csv"
DB_PATH = DATA_DIR / "complaints.db"

OUTPUT_DIR = ROOT / "outputs"
SQL_OUT_DIR = OUTPUT_DIR / "sql"
MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIG_DIR = OUTPUT_DIR / "figures"

SQL_FILE = ROOT / "sql" / "analysis.sql"

CFPB_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

RANDOM_STATE = 42

# Original CFPB column name -> clean snake_case name
COLUMN_MAP = {
    "Date received": "date_received",
    "Product": "product_raw",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
    "Company public response": "company_public_response",
    "Company": "company",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Consumer consent provided?": "consent_provided",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent_to_company",
    "Company response to consumer": "company_response",
    "Timely response?": "timely_response",
    "Consumer disputed?": "consumer_disputed",
    "Complaint ID": "complaint_id",
}


def ensure_dirs():
    for d in (DATA_DIR, SQL_OUT_DIR, MODEL_DIR, METRICS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
