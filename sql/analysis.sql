-- Banking Complaint Intelligence: business analysis queries (SQLite)
-- Each query starts with "-- name:" so src/run_sql.py can run and save it separately.
-- Business question first, technique in brackets.


-- name: 01_volume_by_product
-- Q: Which products generate the most complaints, and how often do customers get money back?
-- [GROUP BY, aggregates, share of total via window function]
SELECT
    product,
    COUNT(*)                                             AS complaints,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)   AS pct_of_total,
    ROUND(100.0 * AVG(got_monetary_relief), 2)           AS pct_monetary_relief,
    ROUND(100.0 * AVG(timely), 1)                        AS pct_timely
FROM complaints
GROUP BY product
ORDER BY complaints DESC;


-- name: 02_monthly_trend_mom_growth
-- Q: Is complaint volume growing? Month-over-month change.
-- [CTE, LAG window function]
WITH monthly AS (
    SELECT year_month, COUNT(*) AS complaints
    FROM complaints
    GROUP BY year_month
)
SELECT
    year_month,
    complaints,
    LAG(complaints) OVER (ORDER BY year_month)                     AS prev_month,
    ROUND(100.0 * (complaints - LAG(complaints) OVER (ORDER BY year_month))
          / LAG(complaints) OVER (ORDER BY year_month), 1)         AS mom_growth_pct
FROM monthly
ORDER BY year_month;


-- name: 03_companies_worst_timeliness
-- Q: Among companies with meaningful volume, who misses response deadlines most?
-- [HAVING to filter small companies, CASE WHEN]
SELECT
    company,
    COUNT(*)                                                   AS complaints,
    SUM(CASE WHEN timely = 0 THEN 1 ELSE 0 END)                AS untimely,
    ROUND(100.0 * SUM(CASE WHEN timely = 0 THEN 1 ELSE 0 END)
          / COUNT(*), 2)                                       AS pct_untimely
FROM complaints
GROUP BY company
HAVING COUNT(*) >= 100
ORDER BY pct_untimely DESC, complaints DESC
LIMIT 15;


-- name: 04_top3_companies_per_product
-- Q: Which 3 companies dominate complaints within each product?
-- [CTE + ROW_NUMBER() OVER (PARTITION BY ...)]  classic "top N per group"
WITH ranked AS (
    SELECT
        product,
        company,
        COUNT(*) AS complaints,
        ROW_NUMBER() OVER (PARTITION BY product ORDER BY COUNT(*) DESC) AS rn
    FROM complaints
    GROUP BY product, company
)
SELECT product, rn AS rank, company, complaints
FROM ranked
WHERE rn <= 3
ORDER BY product, rn;


-- name: 05_response_mix_by_product
-- Q: How do companies resolve complaints for each product?
-- [conditional aggregation = pivot with CASE WHEN]
SELECT
    product,
    COUNT(*) AS complaints,
    ROUND(100.0 * SUM(CASE WHEN company_response = 'Closed with explanation'        THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_explanation,
    ROUND(100.0 * SUM(CASE WHEN company_response = 'Closed with monetary relief'    THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_monetary,
    ROUND(100.0 * SUM(CASE WHEN company_response = 'Closed with non-monetary relief' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_non_monetary,
    ROUND(100.0 * SUM(CASE WHEN company_response = 'In progress'                    THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_in_progress
FROM complaints
GROUP BY product
ORDER BY complaints DESC;


-- name: 06_top_issues_per_product
-- Q: What are customers actually complaining about in each product?
-- [DENSE_RANK keeps ties]
WITH issue_counts AS (
    SELECT product, issue, COUNT(*) AS complaints
    FROM complaints
    GROUP BY product, issue
),
ranked AS (
    SELECT *, DENSE_RANK() OVER (PARTITION BY product ORDER BY complaints DESC) AS rnk
    FROM issue_counts
)
SELECT product, rnk, issue, complaints
FROM ranked
WHERE rnk <= 3
ORDER BY product, rnk;


-- name: 07_top_states
-- Q: Which states file the most complaints, and what share comes from each?
-- [aggregate + window share; per-capita would need census data joined in]
SELECT
    state,
    COUNT(*)                                             AS complaints,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)   AS pct_of_total
FROM complaints
WHERE state IS NOT NULL
GROUP BY state
ORDER BY complaints DESC
LIMIT 10;


-- name: 08_cumulative_by_product_year
-- Q: Running total of complaints per product across months.
-- [SUM() OVER with ORDER BY = running total]
WITH monthly AS (
    SELECT product, year_month, COUNT(*) AS complaints
    FROM complaints
    GROUP BY product, year_month
)
SELECT
    product,
    year_month,
    complaints,
    SUM(complaints) OVER (PARTITION BY product ORDER BY year_month) AS running_total
FROM monthly
ORDER BY product, year_month;


-- name: 09_companies_below_peer_relief
-- Q: Which companies have a monetary-relief rate below their product's average
--    (i.e. customers are less likely to get money back than at peers)?
-- [correlated comparison via CTEs and JOIN]
WITH company_rates AS (
    SELECT product, company, COUNT(*) AS complaints, AVG(got_monetary_relief) AS relief_rate
    FROM complaints
    GROUP BY product, company
    HAVING COUNT(*) >= 50
),
product_rates AS (
    SELECT product, AVG(got_monetary_relief) AS product_avg
    FROM complaints
    GROUP BY product
)
SELECT
    c.product,
    c.company,
    c.complaints,
    ROUND(100.0 * c.relief_rate, 2) AS company_relief_pct,
    ROUND(100.0 * p.product_avg, 2) AS product_avg_pct
FROM company_rates c
JOIN product_rates p USING (product)
WHERE c.relief_rate < p.product_avg
ORDER BY c.complaints DESC
LIMIT 20;


-- name: 10_fraud_signal_by_product
-- Q: How many complaints mention fraud / unauthorized activity, and what money is at stake?
-- [JOIN with regex-extracted features table; run src/regex_extract.py first]
SELECT
    c.product,
    COUNT(*)                                              AS complaints,
    SUM(f.fraud_flag)                                     AS fraud_mentions,
    ROUND(100.0 * AVG(f.fraud_flag), 1)                   AS pct_fraud,
    ROUND(AVG(CASE WHEN f.max_amount > 0 THEN f.max_amount END), 0) AS avg_amount_mentioned
FROM complaints c
JOIN complaint_features f ON f.complaint_id = c.complaint_id
GROUP BY c.product
ORDER BY pct_fraud DESC;
