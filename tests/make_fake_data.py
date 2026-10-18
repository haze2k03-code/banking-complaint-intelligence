"""
SMOKE-TEST DATA ONLY. Generates a small fake file in the CFPB column format so the
pipeline can be tested without the multi-GB download.

The numbers this produces are meaningless. Never report them anywhere.

Usage:
    python tests/make_fake_data.py              # writes data/fake_complaints.csv
    python src/load_to_sqlite.py --csv data/fake_complaints.csv
"""
import random
from pathlib import Path

import pandas as pd

random.seed(0)
OUT = Path(__file__).resolve().parent.parent / "data" / "fake_complaints.csv"

PRODUCTS = {
    "Credit reporting or other personal consumer reports": [
        "There is an account on my credit report that is not mine. I disputed it with the credit bureau on XX/XX/XXXX but they verified it anyway and my credit score dropped.",
        "My credit report shows a late payment that is inaccurate. The reporting agency has not corrected the information after my dispute.",
        "Someone opened accounts in my name, this is identity theft. The inquiries on my credit file are unauthorized.",
    ],
    "Debt collection": [
        "A debt collector keeps calling me about a debt of {$850.00} that I do not owe. They threatened to sue me and take me to court.",
        "The collection agency never sent validation of the debt and continues to call my workplace every day.",
        "I paid this debt in full but the collector is still attempting to collect {$1,200.00} and reporting it.",
    ],
    "Credit card": [
        "My credit card was charged an annual fee of {$95.00} that I was told would be waived. The card company refused to refund it.",
        "There were fraudulent charges on my credit card totaling {$2,340.00}. I did not authorize these purchases and the issuer denied my claim.",
        "The interest rate on my card was increased without notice and late fees were added to my statement.",
    ],
    "Checking or savings account": [
        "The bank charged me overdraft fees of {$35.00} three times in one day on my checking account.",
        "My savings account was frozen without explanation and I cannot access my deposit of {$4,500.00}.",
        "Someone withdrew money from my checking account without authorization. The bank says the transaction is not fraud.",
    ],
    "Mortgage": [
        "My mortgage servicer misapplied my escrow payment and now claims I owe {$3,100.00}. I applied for a loan modification months ago.",
        "The lender is moving forward with foreclosure on my home even though I submitted every document for loss mitigation.",
        "My mortgage payment increased because of an escrow shortage that the servicer cannot explain.",
    ],
    "Student loan": [
        "My student loan servicer placed my loans in forbearance without asking me, and interest capitalized.",
        "I qualify for income driven repayment but the servicer keeps losing my paperwork and my federal loan is now delinquent.",
    ],
    "Vehicle loan or lease": [
        "The auto lender repossessed my car even though I made my payment of {$420.00} on time.",
        "My vehicle lease end charges include {$1,800.00} in excess wear fees that I dispute.",
    ],
    "Money transfer, virtual currency, or money service": [
        "I sent {$600.00} through a payment app to a seller who turned out to be a scam. The app refused to reverse the transfer.",
        "My cryptocurrency wallet was drained and the exchange will not respond to my support tickets.",
    ],
    "Payday loan, title loan, personal loan, or advance loan": [
        "The payday lender charged an interest rate far above what was disclosed and debits my account every two weeks.",
        "I took a personal loan of {$2,000.00} and the lender is adding fees I never agreed to.",
    ],
}
WEIGHTS = [0.40, 0.14, 0.11, 0.09, 0.08, 0.05, 0.04, 0.05, 0.04]
FILLER = [
    "I have contacted them multiple times.", "This has caused me a lot of stress.",
    "Please help me resolve this issue.", "I want this corrected immediately.",
    "I called customer service on XX/XX/XXXX.", "They told me to wait 30 days.", "",
]
COMPANIES = ["Alpha Bank", "Beta Financial", "Gamma Credit Bureau", "Delta Servicing",
             "Epsilon Collections", "Zeta Card Services", "Eta Payments", "Theta Lending"]
RESPONSES = ["Closed with explanation"] * 8 + ["Closed with monetary relief", "Closed with non-monetary relief", "In progress"]
STATES = ["CA", "TX", "FL", "NY", "GA", "IL", "PA", "OH", "NC", "NJ"]


def main(n=3000):
    rows = []
    products = list(PRODUCTS)
    for i in range(n):
        prod = random.choices(products, WEIGHTS)[0]
        base = random.choice(PRODUCTS[prod])
        narrative = " ".join([base] + random.sample(FILLER, 2))
        day = pd.Timestamp("2023-01-01") + pd.Timedelta(days=random.randint(0, 600))
        rows.append({
            "Date received": day.strftime("%Y-%m-%d"),
            "Product": prod,
            "Sub-product": "",
            "Issue": random.choice(["Incorrect information", "Attempts to collect debt", "Fees or interest",
                                    "Problem with a purchase", "Trouble during payment process"]),
            "Sub-issue": "",
            "Consumer complaint narrative": narrative,
            "Company public response": "",
            "Company": random.choice(COMPANIES),
            "State": random.choice(STATES),
            "ZIP code": "",
            "Tags": "",
            "Consumer consent provided?": "Consent provided",
            "Submitted via": "Web",
            "Date sent to company": (day + pd.Timedelta(days=random.randint(0, 3))).strftime("%Y-%m-%d"),
            "Company response to consumer": random.choice(RESPONSES),
            "Timely response?": "Yes" if random.random() > 0.03 else "No",
            "Consumer disputed?": "",
            "Complaint ID": str(1_000_000 + i),
        })
    from_cols = list(rows[0])
    df = pd.DataFrame(rows, columns=from_cols)
    # Rename to snake_case like download_data.py does, so load_to_sqlite.py accepts it directly.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from config import COLUMN_MAP
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.rename(columns=COLUMN_MAP).to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} FAKE complaints -> {OUT}")


if __name__ == "__main__":
    main()
