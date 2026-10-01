# RetailPulse - Data Dictionary & KPI Specification

## 1. Raw Dataset Schema (UCI Online Retail)
The target real-data format mirrors the official UCI Machine Learning Repository Online Retail dataset.

| Column | Data Type | Nullable | Description | Handling / Cleaning Rules |
| :--- | :--- | :--- | :--- | :--- |
| `InvoiceNo` | String / Object | No | 6-digit order identifier. Cancellations prefixed with `'C'`. | Preserved as string. Stripped of whitespace. Used to identify cancellations. |
| `StockCode` | String / Object | No | 5-6 digit product SKU code. | Stripped and upper-cased. Identifies catalog products. |
| `Description` | String / Object | Yes | Product name. | Imputed as `"(No description)"` if missing or whitespace. |
| `Quantity` | Integer | No | Transaction quantity of merchandise. | Negative indicates returns/cancellations. Zero is quarantined. |
| `InvoiceDate` | Timestamp | No | Transaction datetime. | Parsed to datetime. Unparseable strings are quarantined. |
| `UnitPrice` | Float / Numeric | No | Product price per unit in GBP (£). | Must be > 0.0. Values <= 0.0 (adjustments, bad debts) are quarantined. |
| `CustomerID` | String / Object | Yes | 5-digit customer identifier. | Preserved as string (e.g., `'17850.0'` -> `'17850'`). Missing IDs kept for sales, excluded from customer metrics. |
| `Country` | String | No | Customer residency country. | Defaults to `'Unspecified'` if missing. |

---

## 2. Dimensional Analytical Model Schema

### Table: `fct_transactions`
- **Grain**: One record per transaction line item in an invoice or return notice.
- **Primary Key**: `transaction_id` (Surrogate string key).
- **Foreign Keys**: `invoice_no` -> `fct_orders(invoice_no)`, `stock_code` -> `dim_products(stock_code)`, `customer_id` -> `dim_customers(customer_id)`, `date` -> `dim_dates(date_id)`.

| Column | Type | Definition |
| :--- | :--- | :--- |
| `transaction_id` | VARCHAR | Unique surrogate line identifier (`TX_{invoice_no}_{stock_code}_{index}`). |
| `invoice_no` | VARCHAR | Sales order or cancellation invoice number. |
| `stock_code` | VARCHAR | Standardized product SKU. |
| `description` | VARCHAR | Product description. |
| `quantity` | INTEGER | Dispatched quantity (positive for sales, negative for returns). |
| `unit_price` | DOUBLE | Unit selling price in GBP (£). |
| `line_amount` | DOUBLE | Signed line value (`quantity * unit_price`). |
| `gross_amount` | DOUBLE | Positive sales revenue (`max(0, quantity) * unit_price`). |
| `return_amount` | DOUBLE | Positive return magnitude (`abs(min(0, quantity)) * unit_price` if return else 0.0). |
| `net_amount` | DOUBLE | Net transaction revenue (`gross_amount - return_amount`). |
| `invoice_date` | TIMESTAMP | Exact transaction timestamp. |
| `date` | DATE | Calendar date of transaction. |
| `year_month` | VARCHAR | Calendar month (`YYYY-MM`). |
| `week_start` | DATE | Monday date of transaction week. |
| `customer_id` | VARCHAR | Identified customer string or NULL for guest checkouts. |
| `country` | VARCHAR | Country of billing/shipping. |
| `is_cancellation`| BOOLEAN | True if `invoice_no` begins with `'C'`. |
| `is_return` | BOOLEAN | True if `is_cancellation` or `quantity < 0`. |
| `is_qualifying_sale`| BOOLEAN | True if positive sale (`not is_return and quantity > 0 and unit_price > 0`). |
| `has_customer_id`| BOOLEAN | True if customer was identified. |

### Table: `fct_orders`
- **Grain**: One record per unique order invoice.
- **Primary Key**: `invoice_no`.

| Column | Type | Definition |
| :--- | :--- | :--- |
| `invoice_no` | VARCHAR | Unique invoice identifier. |
| `customer_id` | VARCHAR | Associated customer ID or NULL. |
| `country` | VARCHAR | Shipping country. |
| `order_timestamp`| TIMESTAMP | First timestamp recorded for invoice. |
| `order_date` | DATE | Date of invoice. |
| `is_cancellation`| BOOLEAN | Whether invoice represents a return/cancellation. |
| `is_qualifying_sale`| BOOLEAN| Whether invoice represents a valid positive sales order. |
| `line_count` | INTEGER | Number of distinct item lines in the invoice. |
| `total_quantity`| INTEGER | Net total quantity across all lines in invoice. |
| `gross_sales` | DOUBLE | Gross merchandise value (£) of positive sales lines. |
| `return_value` | DOUBLE | Monetary value (£) of returned items. |
| `net_revenue` | DOUBLE | Retained revenue (£) (`gross_sales - return_value`). |

### Table: `dim_customers`
- **Grain**: One record per registered, known customer.
- **Primary Key**: `customer_id`.

| Column | Type | Definition |
| :--- | :--- | :--- |
| `customer_id` | VARCHAR | Unique customer account string. |
| `country` | VARCHAR | Primary customer country. |
| `first_purchase_date`| TIMESTAMP | First observed qualifying transaction timestamp in dataset. |
| `last_purchase_date`| TIMESTAMP | Most recent observed qualifying transaction timestamp. |
| `lifetime_orders`| INTEGER | Count of distinct qualifying sales orders placed. |
| `lifetime_gross_spend`| DOUBLE | Cumulative gross spend (£) on qualifying purchases. |

### Table: `dim_products`
- **Grain**: One record per catalog SKU code.
- **Primary Key**: `stock_code`.

| Column | Type | Definition |
| :--- | :--- | :--- |
| `stock_code` | VARCHAR | Unique SKU code. |
| `description` | VARCHAR | Most frequent descriptive name. |
| `typical_unit_price`| DOUBLE | Modal selling price. |

### Table: `dim_dates`
- **Grain**: One record per calendar day.
- **Primary Key**: `date_id` (DATE).

| Column | Type | Definition |
| :--- | :--- | :--- |
| `date_id` | DATE | Calendar date. |
| `year` | INTEGER | 4-digit calendar year. |
| `month` | INTEGER | Calendar month number (1-12). |
| `year_month` | VARCHAR | Format `YYYY-MM`. |
| `quarter` | INTEGER | Calendar quarter (1-4). |
| `week_start` | DATE | Monday commencing the week. |
| `day_name` | VARCHAR | Day of the week (Monday - Sunday). |
| `is_weekend` | BOOLEAN | True if Saturday or Sunday. |

---

## 3. Standardized Business KPI Definitions

| KPI Name | Formula | Unit | Business Interpretation |
| :--- | :--- | :--- | :--- |
| **Gross Sales** | `SUM(Quantity * UnitPrice)` where `Quantity > 0` and `UnitPrice > 0` and `not is_return` | GBP (£) | Total gross demand before returns and deductions. |
| **Return Value** | `SUM(ABS(Quantity) * UnitPrice)` where `is_return = TRUE` | GBP (£) | Monetary volume of refunds and cancellations processed. |
| **Net Revenue** | `Gross Sales - Return Value` | GBP (£) | Core retained commercial income. |
| **Completed Sales Orders** | `COUNT(DISTINCT InvoiceNo)` where `is_qualifying_sale = TRUE` | Orders | Volume of commercial transactions executed. |
| **Average Order Value (AOV)** | `Gross Sales / Completed Sales Orders` (0.0 if orders == 0) | GBP (£) / Order | Average transaction size across completed orders. |
| **Purchasing Customers** | `COUNT(DISTINCT CustomerID)` where `is_qualifying_sale AND CustomerID IS NOT NULL` | Customers | Number of identified customers active in window. |
| **Repeat Purchase Rate** | `(Customers with >= 2 qualifying orders / Purchasing Customers) * 100` | % | Loyalty metric measuring repurchase frequency. |
| **Units Sold** | `SUM(Quantity)` where `is_qualifying_sale = TRUE` | Units | Operational fulfillment dispatch volume. |
| **Returned Units** | `SUM(ABS(Quantity))` where `is_return = TRUE` | Units | Operational reverse-logistics volume. |
| **Unit Return Rate** | `(Returned Units / Units Sold) * 100` (0.0 if Units Sold == 0) | % | Product quality and return velocity indicator. |
| **Value Return Rate** | `(Return Value / Gross Sales) * 100` (0.0 if Gross Sales == 0) | % | Commercial revenue leakage to refunds. |

---

## 4. Analytical Boundary Rules & Exclusions
1. **Zero Margin / COGS Claims**: Cost of Goods Sold is NOT provided in the UCI Online Retail dataset. Gross profit, markup, and net margin are never reported to prevent unsubstantiated speculation.
2. **Missing Customer IDs**: Unregistered guest orders are included in company sales, revenue, and product volumes, but excluded from customer-level RFM and cohort retention matrices.
3. **Cancellation Double-Counting**: Cancellations (prefixed with `'C'`) and negative quantity records are processed as return entries; they never reduce the positive gross sales count twice.
