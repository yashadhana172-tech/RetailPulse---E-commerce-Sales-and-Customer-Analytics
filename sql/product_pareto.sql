-- RetailPulse Product Pareto & Concentration Analysis
-- Demonstrates CTEs, Joins, Window Functions (SUM() OVER (...), ROW_NUMBER() OVER (...))
-- Parameters: $start_date (DATE), $end_date (DATE), $country (VARCHAR or 'ALL')

WITH product_aggregates AS (
    SELECT
        t.stock_code,
        p.description,
        COALESCE(SUM(CASE WHEN t.is_qualifying_sale THEN t.gross_amount ELSE 0.0 END), 0.0) AS gross_revenue,
        COALESCE(SUM(CASE WHEN t.is_return THEN t.return_amount ELSE 0.0 END), 0.0) AS return_value,
        COALESCE(SUM(CASE WHEN t.is_qualifying_sale THEN t.quantity ELSE 0 END), 0) AS units_sold,
        COALESCE(SUM(CASE WHEN t.is_return THEN ABS(t.quantity) ELSE 0 END), 0) AS units_returned,
        COUNT(DISTINCT CASE WHEN t.is_qualifying_sale THEN t.invoice_no END) AS order_appearances
    FROM fct_transactions t
    LEFT JOIN dim_products p ON t.stock_code = p.stock_code
    WHERE t.date >= $start_date
      AND t.date <= $end_date
      AND ($country = 'ALL' OR t.country = $country)
    GROUP BY t.stock_code, p.description
    HAVING SUM(CASE WHEN t.is_qualifying_sale THEN t.gross_amount ELSE 0.0 END) > 0
),
ranked_products AS (
    SELECT
        stock_code,
        description,
        gross_revenue,
        return_value,
        (gross_revenue - return_value) AS net_revenue,
        units_sold,
        units_returned,
        order_appearances,
        CASE
            WHEN units_sold > 0 THEN (CAST(units_returned AS DOUBLE) / units_sold) * 100.0
            ELSE 0.0
        END AS return_rate_units_pct,
        CASE
            WHEN gross_revenue > 0 THEN (return_value / gross_revenue) * 100.0
            ELSE 0.0
        END AS return_rate_value_pct,
        ROW_NUMBER() OVER (ORDER BY gross_revenue DESC) AS product_rank,
        COUNT(*) OVER () AS total_products_count,
        SUM(gross_revenue) OVER () AS total_catalog_revenue,
        SUM(gross_revenue) OVER (
            ORDER BY gross_revenue DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_revenue
    FROM product_aggregates
)
SELECT
    product_rank,
    stock_code,
    description,
    gross_revenue,
    return_value,
    net_revenue,
    units_sold,
    units_returned,
    return_rate_units_pct,
    return_rate_value_pct,
    order_appearances,
    (CAST(product_rank AS DOUBLE) / total_products_count) * 100.0 AS cumulative_product_pct,
    (cumulative_revenue / total_catalog_revenue) * 100.0 AS cumulative_revenue_pct,
    CASE
        WHEN (cumulative_revenue / total_catalog_revenue) <= 0.80 THEN 'Top 80% (Core)'
        ELSE 'Long Tail (Bottom 20%)'
    END AS pareto_class
FROM ranked_products
ORDER BY product_rank ASC;
