# Banking Complaint Intelligence

Analysing consumer banking complaints to find where customers are being let down, flag fraud and legal-escalation risk, and automatically route each complaint to the right product team.

**Data:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) — public complaints about US banks, lenders, card issuers and credit bureaus, each with the customer's own written narrative.

**Stack:** Python (pandas, scikit-learn), SQL (SQLite: CTEs, window functions), regex, Hugging Face transformers, Streamlit

<!-- After running: add a dashboard screenshot here -->
<!-- ![Dashboard](outputs/figures/dashboard.png) -->

## Business questions

1. Which products and companies generate the most complaints, and is volume growing?
2. Which companies miss response deadlines or rarely give customers money back compared with peers?
3. How many complaints involve fraud, fees or threats of legal action, and how much money is at stake?
4. Can we route a new complaint to the right product team automatically, and is a transformer worth its cost over a simple model?

## Key findings

> Replace with your real results (see `reports/insights.md` for the full brief).

- <Finding 1>
- <Finding 2>
- <Finding 3>

## Results: complaint routing

| Model | Accuracy | Macro F1 | ms / complaint |
|---|---|---|---|
| Majority-class guess | | | – |
| TF-IDF + Logistic Regression | | | |
| Zero-shot transformer | | | |

## Pipeline

```
download_data.py   → stream the multi-GB CSV, keep a random sample with narratives
load_to_sqlite.py  → clean, map 18 raw product names to 9 stable groups, load SQLite
run_sql.py         → 10 business queries in sql/analysis.sql, results to outputs/sql/
regex_extract.py   → $ amounts, fraud / fee / legal-threat flags → complaint_features table
train_baseline.py  → TF-IDF + Logistic Regression router, metrics, confusion matrix
zero_shot.py       → transformer (no training) vs baseline on the same test complaints
dashboard/app.py   → Streamlit dashboard
```

## Run it

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py          # ~1-2 GB download, then samples 80,000 complaints
python src/load_to_sqlite.py
python src/run_sql.py
python src/regex_extract.py
python src/train_baseline.py
python src/zero_shot.py --n 300      # optional, slow on CPU
streamlit run dashboard/app.py
```

Quick test without the big download (fake data, results meaningless):

```bash
python tests/make_fake_data.py
python src/load_to_sqlite.py --csv data/fake_complaints.csv
```

## Project structure

```
├── src/            pipeline scripts
├── sql/            analysis.sql (all business queries)
├── dashboard/      Streamlit app
├── reports/        insights.md (one-page brief for a business audience)
├── outputs/        SQL results, metrics, figures (models are gitignored)
└── tests/          fake-data generator for smoke tests
```

## Design decisions

- **Product grouping:** CFPB renamed several product categories over time (credit reporting alone has three names). Mapping them to 9 groups stops trends and model labels from splitting across duplicates.
- **Macro F1, not just accuracy:** credit reporting dominates the data, so a model can score high accuracy by ignoring small products. Macro F1 weights every product team equally.
- **Majority-class baseline:** shows how much the model actually adds over guessing.
- **Regex before ML:** cheap, explainable signals a business user can trust and audit.

## Limitations

- Complaints are not a representative sample of all customers, and volume scales with company size.
- Only complaints with a published narrative are included.
- Product labels are chosen by consumers and are sometimes wrong, which limits achievable accuracy.
- Keyword flags indicate *mentions*, not confirmed fraud.
