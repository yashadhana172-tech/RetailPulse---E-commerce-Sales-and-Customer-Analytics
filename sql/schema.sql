-- RetailPulse Analytical Schema Definition (DuckDB)
--
-- Dimensional Model Grain and Keys:
--
-- 1. fct_transactions
--    - Grain: One row per transaction line item in an invoice or return slip.
--    - Primary Key: transaction_id
--    - Foreign Keys: invoice_no -> fct_orders, stock_code -> dim_products, customer_id -> dim_customers, date -> dim_dates
--
-- 2. fct_orders
--    - Grain: One row per unique invoice (sales order or return notice).
--    - Primary Key: invoice_no
--    - Foreign Keys: customer_id -> dim_customers, date -> dim_dates
--
-- 3. dim_customers
--    - Grain: One row per distinct known customer (excluding guest checkout).
--    - Primary Key: customer_id
--
-- 4. dim_products
--    - Grain: One row per unique catalog item stock code.
--    - Primary Key: stock_code
--
-- 5. dim_dates
--    - Grain: One row per calendar date spanning the dataset coverage.
--    - Primary Key: date_id

CREATE TABLE IF NOT EXISTS dim_dates (
    date_id DATE PRIMARY KEY,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    year_month VARCHAR(7) NOT NULL,
    quarter INTEGER NOT NULL,
    week_start DATE NOT NULL,
    day_name VARCHAR(15) NOT NULL,
    is_weekend BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_products (
    stock_code VARCHAR(30) PRIMARY KEY,
    description VARCHAR(255) NOT NULL,
    typical_unit_price DOUBLE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_customers (
    customer_id VARCHAR(30) PRIMARY KEY,
    country VARCHAR(100) NOT NULL,
    first_purchase_date TIMESTAMP,
    last_purchase_date TIMESTAMP,
    lifetime_orders INTEGER NOT NULL,
    lifetime_gross_spend DOUBLE NOT NULL
);

CREATE TABLE IF NOT EXISTS fct_orders (
    invoice_no VARCHAR(30) PRIMARY KEY,
    customer_id VARCHAR(30),
    country VARCHAR(100) NOT NULL,
    order_timestamp TIMESTAMP NOT NULL,
    order_date DATE NOT NULL,
    year_month VARCHAR(7) NOT NULL,
    is_cancellation BOOLEAN NOT NULL,
    is_qualifying_sale BOOLEAN NOT NULL,
    line_count INTEGER NOT NULL,
    total_quantity INTEGER NOT NULL,
    gross_sales DOUBLE NOT NULL,
    return_value DOUBLE NOT NULL,
    net_revenue DOUBLE NOT NULL
);

CREATE TABLE IF NOT EXISTS fct_transactions (
    transaction_id VARCHAR(100) PRIMARY KEY,
    invoice_no VARCHAR(30) NOT NULL,
    stock_code VARCHAR(30) NOT NULL,
    description VARCHAR(255) NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DOUBLE NOT NULL,
    line_amount DOUBLE NOT NULL,
    gross_amount DOUBLE NOT NULL,
    return_amount DOUBLE NOT NULL,
    net_amount DOUBLE NOT NULL,
    invoice_date TIMESTAMP NOT NULL,
    date DATE NOT NULL,
    year_month VARCHAR(7) NOT NULL,
    week_start DATE NOT NULL,
    customer_id VARCHAR(30),
    country VARCHAR(100) NOT NULL,
    is_cancellation BOOLEAN NOT NULL,
    is_return BOOLEAN NOT NULL,
    is_qualifying_sale BOOLEAN NOT NULL,
    has_customer_id BOOLEAN NOT NULL
);
