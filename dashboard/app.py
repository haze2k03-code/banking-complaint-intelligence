"""
Streamlit dashboard for the Banking Complaint Intelligence project.

Run locally:
    streamlit run dashboard/app.py
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import DB_PATH, FIG_DIR, METRICS_DIR, MODEL_DIR  # noqa: E402

st.set_page_config(page_title="Complaint Intelligence", layout="wide")


@st.cache_data
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(sql, con, params=params)


def load_json(name: str):
    path = METRICS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def has_table(name: str) -> bool:
    return not query("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)).empty


if not DB_PATH.exists():
    st.error("No database found. Run `python src/load_to_sqlite.py` first.")
    st.stop()

st.title("Banking Complaint Intelligence")
st.caption("CFPB Consumer Complaint Database · SQL analysis, regex signals and NLP complaint routing")

# ---------- Filters ----------
products = query("SELECT DISTINCT product FROM complaints ORDER BY product")["product"].tolist()
chosen = st.sidebar.multiselect("Products", products, default=products)
if not chosen:
    st.warning("Select at least one product.")
    st.stop()
placeholders = ",".join("?" * len(chosen))
where = f"WHERE product IN ({placeholders})"
params = tuple(chosen)

# ---------- KPI row ----------
kpi = query(
    f"""SELECT COUNT(*) AS n, AVG(timely) AS timely, AVG(got_monetary_relief) AS relief,
               COUNT(DISTINCT company) AS companies
        FROM complaints {where}""",
    params,
).iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Complaints", f"{int(kpi.n):,}")
c2.metric("Companies", f"{int(kpi.companies):,}")
c3.metric("Timely responses", f"{kpi.timely * 100:.1f}%")
c4.metric("Monetary relief", f"{kpi.relief * 100:.2f}%")

tab_trend, tab_company, tab_signals, tab_model = st.tabs(
    ["Trends", "Companies", "Regex signals", "Routing model"]
)

# ---------- Trends ----------
with tab_trend:
    monthly = query(
        f"""SELECT year_month, product, COUNT(*) AS complaints
            FROM complaints {where} GROUP BY year_month, product ORDER BY year_month""",
        params,
    )
    pivot = monthly.pivot(index="year_month", columns="product", values="complaints").fillna(0)
    st.subheader("Monthly complaints by product")
    st.line_chart(pivot)

    st.subheader("Volume by product")
    vol = query(
        f"SELECT product, COUNT(*) AS complaints FROM complaints {where} GROUP BY product ORDER BY complaints DESC",
        params,
    )
    st.bar_chart(vol.set_index("product"), horizontal=True)

# ---------- Companies ----------
with tab_company:
    min_n = st.slider("Minimum complaints per company", 10, 500, 100, step=10)
    comp = query(
        f"""SELECT company, COUNT(*) AS complaints,
                   ROUND(100.0 * (1 - AVG(timely)), 2) AS pct_untimely,
                   ROUND(100.0 * AVG(got_monetary_relief), 2) AS pct_monetary_relief
            FROM complaints {where}
            GROUP BY company HAVING COUNT(*) >= ?
            ORDER BY complaints DESC LIMIT 25""",
        params + (min_n,),
    )
    st.subheader("Top companies by complaint volume")
    st.dataframe(comp, use_container_width=True, hide_index=True)

# ---------- Regex signals ----------
with tab_signals:
    if has_table("complaint_features"):
        sig = query(
            f"""SELECT c.product,
                       ROUND(100.0 * AVG(f.fraud_flag), 1) AS pct_fraud_mention,
                       ROUND(100.0 * AVG(f.fee_flag), 1)   AS pct_fee_mention,
                       ROUND(100.0 * AVG(f.legal_flag), 1) AS pct_legal_threat,
                       ROUND(100.0 * AVG(f.n_amounts > 0), 1) AS pct_mentions_amount
                FROM complaints c JOIN complaint_features f USING (complaint_id)
                {where.replace('product', 'c.product')}
                GROUP BY c.product ORDER BY pct_fraud_mention DESC""",
            params,
        )
        st.subheader("Share of complaints mentioning each signal (%)")
        st.bar_chart(sig.set_index("product")[["pct_fraud_mention", "pct_fee_mention", "pct_legal_threat"]])
        st.dataframe(sig, use_container_width=True, hide_index=True)
        st.caption("Signals come from regular expressions, not ML: fraud/unauthorized wording, fee wording, and legal-action wording.")
    else:
        st.info("Run `python src/regex_extract.py` to build the regex features.")

# ---------- Model ----------
with tab_model:
    base = load_json("baseline.json")
    zs = load_json("zero_shot.json")
    if base:
        st.subheader("Complaint routing: which product team should handle this?")
        rows = [
            {"Model": "Majority-class guess", "Accuracy": base["dummy_accuracy"], "Macro F1": base["dummy_macro_f1"], "ms / complaint": None},
            {"Model": base["model"], "Accuracy": base["accuracy"], "Macro F1": base["macro_f1"], "ms / complaint": base["ms_per_complaint"]},
        ]
        if zs:
            rows.append({
                "Model": f"Zero-shot transformer ({zs['zero_shot_model']}), n={zs['n_complaints']}",
                "Accuracy": zs["zero_shot"]["accuracy"], "Macro F1": zs["zero_shot"]["macro_f1"],
                "ms / complaint": zs["zero_shot"]["ms_per_complaint"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if zs:
            st.caption(
                f"On the same {zs['n_complaints']} complaints the baseline scored "
                f"accuracy {zs['baseline_same_subset']['accuracy']} / macro F1 {zs['baseline_same_subset']['macro_f1']}."
            )
        cm = FIG_DIR / "confusion_matrix.png"
        if cm.exists():
            st.image(str(cm), caption="Baseline confusion matrix")

        model_path = MODEL_DIR / "baseline_tfidf_lr.joblib"
        if model_path.exists():
            st.subheader("Try it")
            text = st.text_area("Paste a complaint", "I was charged an overdraft fee twice on my checking account.")
            if text.strip():
                import joblib

                pipe = joblib.load(model_path)
                proba = pd.Series(pipe.predict_proba([text])[0], index=pipe.classes_).sort_values(ascending=False)
                st.write(f"**Route to:** {proba.index[0]}  ({proba.iloc[0] * 100:.0f}% confidence)")
                st.bar_chart(proba.head(5))
    else:
        st.info("Run `python src/train_baseline.py` to train the routing model.")
