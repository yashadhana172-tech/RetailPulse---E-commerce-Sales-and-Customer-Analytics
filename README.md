# RetailPulse: E-commerce Sales & Customer Analytics

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.0+-yellow.svg)](https://duckdb.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://docs.pytest.org/)

**RetailPulse** is an interactive, reproducible analytics platform and analytics engineering portfolio project. It ingests, cleans, models, and visualizes multi-year transactional e-commerce ledgers to extract rigorous commercial insights across executive KPIs, catalog Pareto concentration, RFM customer segmentation, monthly cohort retention, and short-term statistical demand forecasting.

The application operates completely locally without requiring paid third-party services, cloud warehouses, or external API keys. It comes out-of-the-box with a deterministic 24-month synthetic demo dataset and provides full support for custom CSV and Excel uploads following the official **UCI Online Retail** schema.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ installed.

### 2. Installation
Clone or navigate to the repository directory and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Application
Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```
*The application will automatically load the deterministic demo dataset (`data/sample/online_retail_sample.csv`). If it does not exist, it generates it instantly on first boot.*

### 4. Run Automated Test Suite
Execute the full automated test suite containing 16 unit and integration test fixtures:

```bash
python -m pytest tests/ -v
```

---

## 🎯 Target Real-Data Target: UCI Online Retail Dataset

RetailPulse is engineered to ingest and analyze the official **Online Retail** dataset from the UCI Machine Learning Repository:
- **Repository URL**: [https://archive.ics.uci.edu/dataset/352/online+retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Direct Download**: [https://archive.ics.uci.edu/static/public/352/online+retail.zip](https://archive.ics.uci.edu/static/public/352/online+retail.zip)
- **Citation**: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
- **Donor**: Dr Daqing Chen, Director of Public Analytics, School of Engineering, London South Bank University.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Currency**: British Pound (£ / GBP).

To download the real dataset directly into `data/raw/`:
```bash
python scripts/download_real_data.py
```
You can then upload `Online Retail.xlsx` or its CSV export directly through the application sidebar upload interface.

---

## 🛠️ Tech Stack & Key Technologies

| Technology | Purpose in Project |
| :--- | :--- |
| **Python** | Core analytical and scripting language. |
| **Pandas** | Raw tabular ingestion, strict schema validation, and type normalization. |
| **DuckDB** | In-memory vectorized SQL OLAP engine; runs dimensional queries, CTEs, and window functions. |
| **Streamlit** | Multi-page reactive web user interface. |
| **Plotly** | High-contrast interactive time series, heatmaps, and Pareto distribution visualizations. |
| **statsmodels** | Time-series forecasting (Exponential Smoothing / Holt-Winters) with chronological holdouts. |
| **pytest** | Automated test suite validating data integrity, financial math, and edge cases. |

---

## 📂 Repository Structure

```text
├── app.py                     # Main multi-page Streamlit application controller
├── src/
│   ├── ingestion.py           # UCI schema validator & deterministic synthetic data generator
│   ├── cleaning.py            # Deduplication, string CustomerID formatting, quarantine pipeline
│   ├── database.py            # DuckDB in-memory manager & parameterized query interface
│   ├── metrics.py             # Unified KPI calculations & period-over-period comparison
│   ├── segmentation.py        # RFM scoring with rank-based tie handling & business taxonomy
│   ├── cohorts.py             # Monthly acquisition cohort matrix with unobserved period masking
│   ├── forecasting.py         # Weekly aggregation, holdout validation (WAPE/MAE), statsmodels
│   └── insights.py            # Deterministic, rule-based 5-part business insight generator
├── sql/
│   ├── schema.sql             # Analytical dimensional star schema DDL (DuckDB)
│   ├── kpi_summary.sql        # Parameterized KPI aggregation query
│   ├── product_pareto.sql     # Window functions (SUM OVER, ROW_NUMBER OVER) for 80/20 curves
│   ├── cohort_matrix.sql      # Multi-stage CTEs for monthly cohort retention
│   └── rfm_metrics.sql        # Base RFM calculation query
├── tests/
│   ├── test_stage1.py         # Hand-checkable ledger fixtures, deduplication, & KPI tests
│   ├── test_stage2_3.py       # RFM tied values, cohort NaN masking, WAPE zero handling
│   └── test_end_to_end.py     # Schema validation, SQL vs Pandas parity, country filters
├── scripts/
│   ├── generate_demo_data.py  # Script to generate sample CSV deterministically
│   └── download_real_data.py  # Official UCI dataset download and extractor script
├── docs/
│   ├── data_dictionary.md     # Full schema definitions, table grains, and KPI formulas
│   ├── data_lineage_architecture.md # Architecture flow and data lineage documentation
│   ├── sql_showcase.md        # Detailed breakdown of CTEs, window functions, and joins
│   └── case_study.md          # Comprehensive portfolio business case study
├── notebooks/
│   └── exploratory_analysis.ipynb # End-to-end interactive EDA and pipeline walk-through
├── data/
│   └── sample/
│       └── online_retail_sample.csv # 17,000+ line deterministic synthetic dataset
├── requirements.txt           # Pinned dependency versions
├── .gitignore                 # Excludes local databases, IDE, and large caches
└── README.md                  # Project overview and documentation
```

---

## 📊 Dashboard Pages & Capabilities

### A. Executive Overview
- **Standardized KPI Cards**: Net Revenue, Gross Sales, Return Value, Completed Sales Orders, Average Order Value (AOV), Purchasing Customers, Repeat Purchase Rate, and Units Sold.
- **Period-over-Period Comparison**: Automatically benchmarks current window against the immediately preceding equal-length historical window.
- **Time-Series Trends**: Interactive weekly and monthly revenue trajectories.
- **Top Commercial Drivers**: Top 10 revenue-generating SKUs and international markets.
- **Automated Business Insights**: Deterministic, rule-based cards with numerical observation, comparison window, cautious interpretation, business recommendation, and analytical limitations.

### B. Sales and Products
- **Pareto Concentration Curve (80/20 Rule)**: Visualizes cumulative % of gross sales against cumulative % of catalog items using SQL window functions (`SUM() OVER (...)`, `ROW_NUMBER() OVER (...)`).
- **Return Rate Analytics**: Computes unit and value return rates with explicit stated denominators:
  $$\text{Unit Return Rate} = \frac{\text{Returned Units}}{\text{Gross Units Sold}} \times 100$$
- **Exportable Drill-Down Table**: Searchable catalog table with one-click CSV export.

### C. Customer Segmentation (RFM Analysis)
- **Explicit Reference Date**: Recency is measured relative to an explicit analysis date (defaulting to 1 day after the selected filter window, never system `today()`).
- **Robust Quantile Scoring**: Rank-based binning (`method='first'`) ensures monotonic 1-4 quartile assignments without edge collision errors on single-order buyers.
- **Documented Segments**: Champions, Loyal Customers, Potential Loyalists, New Customers, At Risk, Lost, etc.
- **Methodological Disclaimer**: Transparently identified as a rule-based behavioral taxonomy based on past purchase history, not a probabilistic machine learning churn prediction.

### D. Cohort Retention Analysis
- **Acquisition Definition**: Cohort assigned based on customer's first observed qualifying purchase across the full loaded dataset.
- **Retention Heatmap**: Monthly repeat purchase activity percentage across subsequent calendar months ($M+0, M+1, \dots$).
- **Blank Future Periods**: Future unobserved months are preserved strictly as blank (`NaN`), never false 0% retention.
- **Left-Censoring Note**: Documents that first observed transaction in an extracted dataset does not guarantee true brand inception.

### E. Demand & Sales Forecasting
- **Weekly Aggregation & Coverage**: Distinguishes true zero-revenue weeks from missing calendar dates. Boundary partial weeks are flagged.
- **Rigorous Chronological Holdout**: Evaluates last 4 weeks as test set; zero random splitting.
- **Model Benchmark**: Evaluates Exponential Smoothing (statsmodels) against a Naive 4-Week Moving Average baseline.
- **Error Metrics**: Weighted Absolute Percentage Error (WAPE) and MAE with safe zero-denominator handling.
- **4-Week Point Forecast & 95% Confidence Cone**: Residual-based prediction intervals.

### F. Data Quality & Audit
- **Pipeline Audit**: Real-time counts of ingested rows, deduplicated rows, quarantined rows, and retained analytical rows.
- **Quarantine Reason Breakdown**: Granular reporting of invalid prices, corrupted timestamps, and zero quantities.
- **Guest Checkout Impact**: Reports percentage of rows and revenue originating from missing Customer IDs.
- **Downloadable Audit Report**: One-click Markdown data quality audit download.
- **Unified Data Dictionary**: Complete table of formulas, units, and interpretations.

---

## 🛡️ Analytical Ground Rules & Integrity

1. **No Fabricated Margins**: Cost of Goods Sold (COGS) is not present in the UCI Online Retail dataset. Gross margin and net profit are deliberately excluded to avoid misleading speculation.
2. **Cancellation Accounting**: Cancellation invoices (`'C'`) and negative quantities are captured as returns and subtracted from gross sales; they are never subtracted twice.
3. **Guest Checkout Transparency**: Valid orders without Customer IDs are included in sales and revenue totals, but excluded from customer-level RFM and cohort retention to prevent anonymous cohort distortion.
4. **No Artificial Accuracy Claims**: Time-series models report out-of-sample holdout WAPE/MAE and explicitly acknowledge that statistical projections cannot predict upcoming promotional campaigns or stockouts.

---

## 🧪 Testing & Verification

RetailPulse comes with a comprehensive test suite in `tests/`:
```bash
python -m pytest tests/ -v
```

Tests verify:
- ✅ Customer ID formatting and string preservation (e.g. `'17850.0'` -> `'17850'`).
- ✅ Schema validation and corrupted file rejection.
- ✅ Exact duplicate row logging and deduplication.
- ✅ Hand-checkable financial ledger calculations.
- ✅ Zero-denominator safeguards on empty date filters.
- ✅ RFM quantile scoring with heavy tie values.
- ✅ Retention heatmap blanking for future unobserved months.
- ✅ WAPE zero-handling and forecasting chronological splits.
- ✅ SQL aggregation parity with direct Pandas ground truth.
- ✅ Deterministic 5-part insight card schema adherence.
