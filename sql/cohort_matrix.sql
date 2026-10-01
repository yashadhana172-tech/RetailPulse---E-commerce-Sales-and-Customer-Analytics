-- RetailPulse Monthly Cohort Retention Matrix
-- Assigns acquisition cohorts based on the customer's first observed qualifying purchase
-- in the full dataset, then calculates active retention across subsequent calendar months.
-- Unobserved future periods are explicitly excluded or NULL.

WITH customer_first_purchase AS (
    -- Customer's first observed purchase across the entire dataset
    SELECT
        customer_id,
        MIN(date) AS first_order_date,
        STRFTIME(MIN(date), '%Y-%m') AS cohort_month
    FROM fct_transactions
    WHERE is_qualifying_sale AND has_customer_id
    GROUP BY customer_id
),
cohort_sizes AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT customer_id) AS cohort_size
    FROM customer_first_purchase
    GROUP BY cohort_month
),
customer_monthly_activity AS (
    SELECT DISTINCT
        t.customer_id,
        c.cohort_month,
        STRFTIME(t.date, '%Y-%m') AS activity_month,
        -- Month difference index (period 0 = cohort month)
        (
            (CAST(STRFTIME(t.date, '%Y') AS INTEGER) - CAST(SUBSTR(c.cohort_month, 1, 4) AS INTEGER)) * 12 +
            (CAST(STRFTIME(t.date, '%m') AS INTEGER) - CAST(SUBSTR(c.cohort_month, 6, 2) AS INTEGER))
        ) AS month_number
    FROM fct_transactions t
    INNER JOIN customer_first_purchase c ON t.customer_id = c.customer_id
    WHERE t.is_qualifying_sale AND t.has_customer_id
),
retention_counts AS (
    SELECT
        cohort_month,
        month_number,
        COUNT(DISTINCT customer_id) AS active_customers
    FROM customer_monthly_activity
    GROUP BY cohort_month, month_number
)
SELECT
    r.cohort_month,
    s.cohort_size,
    r.month_number,
    r.active_customers,
    CASE
        WHEN s.cohort_size > 0 THEN (CAST(r.active_customers AS DOUBLE) / s.cohort_size) * 100.0
        ELSE 0.0
    END AS retention_rate_pct
FROM retention_counts r
INNER JOIN cohort_sizes s ON r.cohort_month = s.cohort_month
ORDER BY r.cohort_month ASC, r.month_number ASC;
