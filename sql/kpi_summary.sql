-- RetailPulse Executive KPI Query
-- Demonstrates parameterized date filtering, CTE aggregation, and zero-denominator safeguards.
-- Parameters: $start_date (DATE), $end_date (DATE), $country (VARCHAR or 'ALL')

WITH filtered_tx AS (
    SELECT
        t.*
    FROM fct_transactions t
    WHERE t.date >= $start_date
      AND t.date <= $end_date
      AND ($country = 'ALL' OR t.country = $country)
),
sales_metrics AS (
    SELECT
        COALESCE(SUM(CASE WHEN is_qualifying_sale THEN gross_amount ELSE 0.0 END), 0.0) AS gross_sales,
        COALESCE(SUM(CASE WHEN is_return THEN return_amount ELSE 0.0 END), 0.0) AS return_value,
        COALESCE(SUM(CASE WHEN is_qualifying_sale THEN quantity ELSE 0 END), 0) AS units_sold,
        COALESCE(SUM(CASE WHEN is_return THEN ABS(quantity) ELSE 0 END), 0) AS returned_units,
        COUNT(DISTINCT CASE WHEN is_qualifying_sale THEN invoice_no END) AS completed_sales_orders,
        COUNT(DISTINCT CASE WHEN is_qualifying_sale AND has_customer_id THEN customer_id END) AS purchasing_customers
    FROM filtered_tx
),
repeat_customers AS (
    SELECT
        customer_id,
        COUNT(DISTINCT invoice_no) AS order_count
    FROM filtered_tx
    WHERE is_qualifying_sale AND has_customer_id
    GROUP BY customer_id
    HAVING COUNT(DISTINCT invoice_no) >= 2
)
SELECT
    m.gross_sales,
    m.return_value,
    (m.gross_sales - m.return_value) AS net_revenue,
    m.completed_sales_orders,
    CASE
        WHEN m.completed_sales_orders > 0 THEN m.gross_sales / m.completed_sales_orders
        ELSE 0.0
    END AS average_order_value,
    m.purchasing_customers,
    m.units_sold,
    m.returned_units,
    (SELECT COUNT(*) FROM repeat_customers) AS repeat_customers_count,
    CASE
        WHEN m.purchasing_customers > 0 THEN
            (CAST((SELECT COUNT(*) FROM repeat_customers) AS DOUBLE) / m.purchasing_customers) * 100.0
        ELSE 0.0
    END AS repeat_purchase_rate_pct
FROM sales_metrics m;
