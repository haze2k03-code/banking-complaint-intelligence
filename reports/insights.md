# Insights brief: consumer banking complaints

**Audience:** Head of Customer Risk, a retail bank (hypothetical)
**Data:** CFPB Consumer Complaint Database, random sample of N complaints with narratives received from <DATE> to <DATE>

> Fill every `<...>` from your own outputs (`outputs/sql/*.csv`, `outputs/metrics/*.json`).
> Keep each finding to: what we saw → why it matters → what to do.

## Key findings

**1. <Product> drives <X>% of all complaints.**
<One line from 01_volume_by_product.csv. Why it matters: where complaint-handling capacity should go.>

**2. Complaint volume <grew / fell> <X>% between <month> and <month>.**
<From 02_monthly_trend_mom_growth.csv. Mention any spike and a plausible reason, labelled as a hypothesis.>

**3. <Company> misses response deadlines on <X>% of complaints vs <Y>% overall.**
<From 03_companies_worst_timeliness.csv. Regulatory and reputational risk.>

**4. <X>% of <product> complaints mention fraud or unauthorized activity; median amount at stake ≈ $<X>.**
<From 10_fraud_signal_by_product.csv and regex_extract.py output. Ties to fraud operations.>

**5. Automatic routing is feasible: the TF-IDF model routes complaints to the right product team with macro F1 <X> (vs <Y> for a majority-class guess).**
<From baseline.json. Mention the weakest class from baseline_report.csv and the confusion matrix.>

## Model comparison

| Model | Accuracy | Macro F1 | Time per complaint |
|---|---|---|---|
| Majority-class guess | | | – |
| TF-IDF + Logistic Regression | | | ~<X> ms |
| Zero-shot transformer (<model>) | | | ~<X> ms |

<Two sentences: which one you'd deploy and why (accuracy vs speed/cost vs no labels needed).>

## Recommendations

1. <Action tied to finding 1 or 3>
2. <Action tied to finding 4>
3. <Pilot auto-routing with human review for low-confidence predictions>

## Limitations

- CFPB complaints are not a representative sample of all customers; volume reflects company size.
- Only complaints whose consumers agreed to publish a narrative are included.
- Product labels are chosen by the consumer and can be wrong, which caps model accuracy.
- Regex flags are keyword-based: "fraud" in a sentence does not prove fraud happened.
