# RetailPulse - SQL Query Showcase

This document highlights advanced SQL techniques implemented in DuckDB across the RetailPulse application, including Common Table Expressions (CTEs), Window Functions, Multi-Table Joins, and Parameterized Filtering.

---

## 1. Product Pareto & Concentration Analysis (Window Functions & Running Totals)
**File**: `sql/product_pareto.sql`

Demonstrates:
- Multiple CTEs for staged aggregation.
- `LEFT JOIN` between transactional fact table and product dimensions.
- Window functions: `ROW_NUMBER() OVER (...)`, `COUNT(*) OVER ()`, `SUM() OVER ()`, and unbounded running totals.

```sql
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
```

---

## 2. Customer Acquisition Cohort Retention Matrix
**File**: `sql/cohort_matrix.sql`

Demonstrates:
- First-purchase acquisition timestamp extraction across the full dataset.
- Date arithmetic computing calendar month offsets (`month_number = (Y_act - Y_acq)*12 + (M_act - M_acq)`).
- Two-stage aggregation calculating cohort size and repeat activity rates.

```sql
WITH customer_first_purchase AS (
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
```

---

## 3. Executive KPI Aggregation & Repeat Customer Tracking
**File**: `sql/kpi_summary.sql`

Demonstrates:
- Parameterized date bounding.
- Conditional aggregation with `CASE WHEN`.
- Zero-denominator division protection.
- Isolated repeat customer count via `HAVING COUNT(DISTINCT invoice_no) >= 2`.

```sql
WITH filtered_tx AS (
    SELECT t.*
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
    SELECT customer_id
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
```
