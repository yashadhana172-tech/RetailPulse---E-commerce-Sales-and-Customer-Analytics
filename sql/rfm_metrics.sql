-- RetailPulse Customer RFM Metrics Base Query
-- Calculates Recency (days since last purchase relative to analysis date),
-- Frequency (distinct qualifying orders), and Monetary Value (total qualifying sales revenue).
-- Parameters: $analysis_date (DATE), $start_date (DATE), $end_date (DATE), $country (VARCHAR or 'ALL')

WITH filtered_orders AS (
    SELECT
        t.customer_id,
        t.invoice_no,
        t.date,
        SUM(t.gross_amount) AS order_gross
    FROM fct_transactions t
    WHERE t.is_qualifying_sale
      AND t.has_customer_id
      AND t.date >= $start_date
      AND t.date <= $end_date
      AND ($country = 'ALL' OR t.country = $country)
    GROUP BY t.customer_id, t.invoice_no, t.date
)
SELECT
    customer_id,
    -- Recency: difference between explicit analysis date and customer's latest order date
    DATE_DIFF('day', MAX(date), $analysis_date) AS recency_days,
    -- Frequency: count of distinct qualifying sales orders
    COUNT(DISTINCT invoice_no) AS frequency,
    -- Monetary: sum of qualifying sales revenue
    SUM(order_gross) AS monetary_value,
    MIN(date) AS first_order_date,
    MAX(date) AS last_order_date
FROM filtered_orders
GROUP BY customer_id;
