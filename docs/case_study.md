# Portfolio Case Study: RetailPulse - E-commerce Sales & Customer Analytics

## 1. Executive Summary
**RetailPulse** is an interactive, reproducible analytics platform designed to solve operational visibility and customer intelligence challenges in transactional e-commerce businesses. Built using Python, Pandas, DuckDB, Streamlit, Plotly, and statsmodels, it transforms raw transactional ledgers (modeled after the UCI Online Retail dataset) into actionable commercial intelligence spanning top-line demand, catalog Pareto concentration, RFM behavioral segmentation, monthly cohort retention, and short-term statistical demand forecasting.

---

## 2. Business Problem & Context
Retail and direct-to-consumer e-commerce businesses frequently face several critical analytical blindspots:
1. **Unreliable Financial Ledgers**: Transaction files frequently conflate positive sales with cancellation entries, credit adjustments, and refunds. Naive sums result in distorted revenue figures, inflated order counts, and incorrect basket calculations.
2. **Catalog Concentration Vulnerability**: Retail operators often fail to identify how reliant their top-line revenue is on a small fraction of catalog items, exposing them to catastrophic supply-chain stockouts.
3. **Customer Retention & Churn Ambiguity**: Businesses struggle to distinguish one-time holiday promotional buyers from repeat loyal brand advocates.
4. **Demand Forecasting Over-Engineering**: Complex machine learning pipelines frequently fail in production due to data leakage, random test splitting, and an inability to account for partial recording weeks.

---

## 3. Engineering & Analytical Methodology

### A. Repeatable Ingestion & Quarantine Pipeline
- **Raw Input Preservation**: All original transaction lines are preserved in raw storage.
- **Deduplication Audit**: Exact duplicate transaction rows across all fields are logged and deduplicated to prevent artificial order count inflation.
- **Quarantine Store**: Strict quarantine validation captures invalid entries (unit prices $\le 0$, corrupted timestamps, zero quantities) with explicit reason logging.
- **Cancellation & Return Segregation**: Invoices prefixed with `'C'` and negative quantities are isolated as return events. Financial metrics strictly define:
  $$\text{Gross Sales} = \sum (\text{Quantity} \times \text{UnitPrice}) \quad [\text{Quantity} > 0, \text{UnitPrice} > 0, \text{not return}]$$
  $$\text{Return Value} = \sum (|\text{Quantity}| \times \text{UnitPrice}) \quad [\text{is\_return} = \text{True}]$$
  $$\text{Net Revenue} = \text{Gross Sales} - \text{Return Value}$$

### B. Dimensional Modeling in DuckDB
- Data is normalized into an in-memory dimensional star schema:
  - `fct_transactions` (line item grain)
  - `fct_orders` (invoice grain)
  - `dim_customers` (known customer accounts)
  - `dim_products` (catalog items)
  - `dim_dates` (calendar reference dates)
- SQL queries leverage CTEs, multi-table joins, and window functions (`SUM() OVER (...)`, `ROW_NUMBER() OVER (...)`) for high-performance sub-second slice-and-dice aggregations.

### C. Customer RFM Segmentation
- **Explicit Analysis Date**: Recency is measured relative to a defined reference date (defaulting to 1 day after the selected analysis window, never system clock `today()`), ensuring reproducible historical backtesting.
- **Tied-Score Handling**: Rank-based quantile scoring (`pd.Series.rank(method='first')`) eliminates edge collision errors caused by the large volume of single-order buyers.
- **Taxonomy**: Maps quartiles (1-4) into 10 documented segments (Champions, Loyal Customers, Potential Loyalists, At Risk, Hibernating, etc.).

### D. Monthly Cohort Retention Heatmap
- Acquisition cohort is determined from the customer's *first observed qualifying purchase* across the full loaded dataset.
- Active return rates are tracked across monthly offsets ($M+0, M+1, \dots$).
- Future unobserved calendar periods are strictly represented as `NaN` (blank), avoiding false zeros.

### E. Chronological Demand Forecasting
- Aggregates net revenue at the weekly boundary (`W-MON`) with boundary partial-week detection.
- Employs strict chronological holdout validation (last 4 weeks as holdout; zero random splitting).
- Benchmarks Exponential Smoothing against a Naive 4-Week Moving Average.
- Uses Weighted Absolute Percentage Error (WAPE) and MAE to evaluate holdout accuracy with zero-denominator stability.
- Generates 4-week point forecasts with residual-based 95% prediction intervals.

---

## 4. Observed Analytical Findings (Demo Dataset)

1. **Extreme Catalog Pareto Concentration**:
   - The top ~15-20% of catalog SKUs account for approximately 78-82% of total gross sales.
   - A handful of home and gift lines (e.g. T-Light holders, retro lunch bags, cake stands) drive the majority of commercial volume, while the bottom 80% represents a long-tail inventory carrying cost.
2. **Q4 Seasonal Peak & Post-Holiday Lull**:
   - Net weekly revenue surges by approximately 2.0x during November and December, followed by an immediate January drop.
3. **Customer Churn Decay Curve**:
   - Month 1 repeat retention typically drops to 18-25% from the initial 100% acquisition baseline, followed by gradual stabilization around 12-15% for loyal recurring cohorts.
4. **Guest Checkout Volume**:
   - Anonymous transactions (missing Customer ID) represent approximately 15-20% of total sales. Retaining these in sales reporting prevents top-line undercounting while excluding them from RFM prevents persona contamination.

---

## 5. Methodological Limitations

1. **Lack of COGS (Cost of Goods Sold)**: The UCI Online Retail dataset does not record supplier purchase prices or operational fulfillment costs. Gross profit and net margins are omitted to avoid false financial modeling.
2. **Left-Censoring in Cohort Analysis**: The first observed transaction in an extracted dataset does not guarantee true initial brand acquisition; long-time pre-existing customers appearing in month 1 are categorized in the initial cohort.
3. **Descriptive Nature of RFM**: RFM assigns descriptive behavioral buckets based on past transaction patterns; it is not a predictive machine learning model estimating future purchase probabilities.
4. **Time-Series Extrapolation Constraints**: The statistical forecast assumes prevailing operational and market conditions remain consistent; it does not model upcoming marketing budget alterations, viral spikes, or supplier outages.

---

## 6. Proposed Commercial Next Steps

1. **Catalog Risk Mitigation**: Establish dual-supplier commitments for the top 20% core revenue-generating SKUs to eliminate stockout risk.
2. **Guest-to-Member Conversion**: Deploy a post-checkout incentive (e.g. instant order tracking or loyalty points) to capture emails and convert anonymous guest checkouts into identified accounts.
3. **Automated RFM Lifecycle Triggers**: Integrate segment tags with marketing automation tools (e.g. Klaviyo) to trigger automated re-engagement emails when a customer drifts into "Need Attention" or "At Risk".
4. **Returns Root-Cause Auditing**: Implement return-reason codes at the warehouse return desk to distinguish shipping damage from description mismatch on high-return SKUs.
