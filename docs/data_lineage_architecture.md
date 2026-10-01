# RetailPulse - Data Lineage & Technical Architecture

## 1. System Architecture Overview

RetailPulse uses a modular data engineering and analytics architecture designed for full reproducibility, local execution (zero API keys or external services), and sub-second analytical query performance using DuckDB.

```mermaid
flowchart TD
    subgraph Ingestion ["1. Ingestion Layer"]
        A1[UCI Online Retail Dataset / Custom Upload] --> B1[Schema Validation Engine]
        A2[Deterministic Synthetic Generator Seed=42] --> B1
        B1 -->|Invalid Columns / Types| C1[Upload Error Reject]
        B1 -->|Valid Columns| D1[Raw DataFrame In-Memory]
    end

    subgraph Cleaning ["2. Cleaning & Quarantine Pipeline"]
        D1 --> E1[CustomerID String Standardization]
        E1 --> E2[Exact Duplicate Detection & Deduplication]
        E2 --> E3[Quarantine Filter: Bad Prices / Bad Dates / Zero Qty]
        E3 -->|Quarantined Records| Q1[Quarantine Store & Audit Log]
        E3 -->|Clean Records| F1[Clean Analytical DataFrame]
    end

    subgraph Modeling ["3. DuckDB Dimensional Model Engine"]
        F1 --> G1[(fct_transactions)]
        F1 --> G2[(fct_orders)]
        F1 --> G3[(dim_customers)]
        F1 --> G4[(dim_products)]
        F1 --> G5[(dim_dates)]
    end

    subgraph Analytics ["4. Parameterized Analytical SQL Engines"]
        G1 & G2 & G3 & G4 & G5 --> H1[KPI Summary Engine with Period Comparison]
        G1 & G4 --> H2[Product Pareto & Concentration Engine]
        G1 & G3 --> H3[Customer RFM Segmentation Engine]
        G1 & G3 --> H4[Acquisition Cohort Retention Matrix Engine]
        G1 --> H5[Weekly Time-Series Aggregation & statsmodels Forecast]
        H1 & H2 & H3 --> H6[Deterministic Business Insights Engine]
    end

    subgraph Presentation ["5. Streamlit Interactive Dashboard"]
        H1 --> P1[Page A: Executive Overview]
        H2 --> P2[Page B: Sales and Products]
        H3 --> P3[Page C: Customer Segmentation]
        H4 --> P4[Page D: Cohort Retention]
        H5 --> P5[Page E: Sales Forecast]
        Q1 --> P6[Page F: Data Quality & Audit]
    end
```

---

## 2. Ingestion & Transformation Details

### A. Raw Ingestion (`src/ingestion.py`)
- Standardizes incoming column headers case-insensitively to matching target columns:
  `InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`.
- Rejects files missing required fields with descriptive error notifications.
- Generates a 24-month deterministic synthetic demo dataset with repeat customers, seasonal peaks, cancellations, and intentional data flaws.

### B. Cleaning Pipeline (`src/cleaning.py`)
1. **CustomerID Formatting**: Float numbers like `17850.0` are stripped and preserved as strings (`'17850'`). Empty strings or `'nan'` strings are converted to `None`.
2. **Duplicate Deduplication**: Identifies exact duplicate rows across all original fields. Logs the count and retains the first occurrence to avoid artificial transaction inflation.
3. **Quarantine Routing**: Records with `UnitPrice <= 0`, invalid dates, or `Quantity == 0` are routed into `quarantine_df` with an explicit human-readable reason string.
4. **Cancellation Isolation**: Invoices starting with `'C'` and negative quantities are categorized as returns. Positive sales rows are restricted to `not is_return and quantity > 0 and unit_price > 0`.
5. **Guest Checkout Isolation**: Rows where `customer_id IS NULL` are retained for top-line revenue, volume, and product analysis, but flagged with `has_customer_id = False` to prevent distorting customer retention and RFM metrics.

---

## 3. Dimensional Model & SQL Execution

### DuckDB In-Memory Integration (`src/database.py`)
- Tables are defined via `sql/schema.sql`.
- DuckDB executes all aggregation logic in-memory with C++ vectorized speed.
- Filter parameters (`$start_date`, `$end_date`, `$country`) are passed safely via DuckDB's parameterized interface, preventing SQL injection vulnerabilities.

---

## 4. Analytical Modules
- **KPI Metrics (`src/metrics.py`)**: Computes Gross Sales, Return Value, Net Revenue, Orders, AOV, Purchasing Customers, Repeat Purchase Rate, and Units. Evaluates period-over-period delta against preceding equal-length window.
- **RFM Segmentation (`src/segmentation.py`)**: Quantile scoring with rank-based tie-breaking. Reference date defaults to `end_date + 1 day` (never today's date).
- **Cohort Retention (`src/cohorts.py`)**: Acquisition cohort determined from customer's first observed purchase across the full dataset. Future periods are preserved as `NaN` (blank).
- **Forecasting (`src/forecasting.py`)**: Aggregates weekly series, detects partial boundary weeks, performs chronological holdout (last 4 weeks), benchmarks Exponential Smoothing against Naive Moving Average, and computes MAE/WAPE.
- **Business Insights (`src/insights.py`)**: Deterministic 5-part rule-based insight generation (Observation, Comparison, Cautious Interpretation, Action, Limitation).
