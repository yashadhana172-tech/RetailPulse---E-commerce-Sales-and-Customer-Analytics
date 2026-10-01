"""
RetailPulse - E-commerce Sales and Customer Analytics
Production-grade Portfolio Streamlit Application

Authors: Senior Data Analyst & Analytics Engineering Team
Stack: Python, Pandas, DuckDB, Streamlit, Plotly, statsmodels
"""

from datetime import date, datetime, timedelta
import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ingestion import (
    UCI_METADATA,
    REQUIRED_COLUMNS,
    ensure_sample_data_exists,
    get_demo_dataset_path,
    load_file,
    validate_schema,
)
from src.cleaning import clean_raw_data, CleaningSummary
from src.database import DatabaseManager, get_database_manager
from src.metrics import (
    KPI_DATA_DICTIONARY,
    KPIComparison,
    PeriodKPIs,
    compute_comparison,
    compute_kpis_for_period,
)
from src.segmentation import (
    SEGMENT_DEFINITIONS,
    compute_rfm_segmentation,
)
from src.cohorts import (
    COHORT_METHODOLOGY_NOTES,
    compute_cohort_retention,
)
from src.forecasting import (
    ForecastResult,
    generate_sales_forecast,
)
from src.insights import (
    BusinessInsight,
    generate_business_insights,
)

# ---------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="RetailPulse - E-commerce Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        padding: 1.2rem 1.8rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        border-left: 6px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .demo-banner {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        color: #92400e;
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        font-weight: 500;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .real-banner {
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        color: #065f46;
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        font-weight: 500;
        margin-bottom: 1.2rem;
    }

    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        margin-bottom: 0.75rem;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px -2px rgba(0,0,0,0.08);
    }
    .kpi-title {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        color: #0f172a;
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .kpi-delta-pos {
        color: #059669;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }
    .kpi-delta-neg {
        color: #dc2626;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }
    .kpi-delta-neutral {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 0.35rem;
    }

    .insight-card {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
    }
    .insight-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #0f172a;
        margin-bottom: 0.4rem;
    }
    .insight-obs {
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 0.4rem;
    }
    .insight-sec {
        font-size: 0.9rem;
        color: #334155;
        margin-top: 0.4rem;
    }
    .insight-tag {
        display: inline-block;
        background: #e2e8f0;
        color: #334155;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }

    .caveat-box {
        background: #f1f5f9;
        border-left: 4px solid #64748b;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #475569;
        border-radius: 4px;
        margin-top: 0.8rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Caching & Management
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean_data(source_type: str, file_bytes: Optional[bytes] = None, filename: Optional[str] = None):
    """
    Cached data pipeline: loads either the deterministic synthetic demo sample
    or an uploaded CSV/Excel file.
    """
    if source_type == "demo":
        sample_path = ensure_sample_data_exists()
        raw_df, val = load_file(sample_path)
    else:
        file_ext = filename.split(".")[-1].lower() if filename else "csv"
        raw_df, val = load_file(io.BytesIO(file_bytes), file_type=file_ext)

    if not val.is_valid or raw_df is None:
        return None, None, None, val

    clean_df, quarantine_df, summary = clean_raw_data(raw_df)
    return clean_df, quarantine_df, summary, val


# ---------------------------------------------------------
# Sidebar: Navigation & Controls
# ---------------------------------------------------------
st.sidebar.title("🛍️ RetailPulse")
st.sidebar.caption("E-commerce Sales & Customer Analytics")

# Data Source Switcher
st.sidebar.subheader("1. Data Source")
source_option = st.sidebar.radio(
    "Select Source",
    ["Deterministic Demo Data", "Upload CSV / Excel"],
    index=0,
)

clean_df = None
quarantine_df = None
cleaning_summary = None
is_demo = (source_option == "Deterministic Demo Data")

if is_demo:
    with st.spinner("Loading synthetic demo data..."):
        clean_df, quarantine_df, cleaning_summary, val_res = load_and_clean_data("demo")
else:
    uploaded = st.sidebar.file_uploader(
        "Upload dataset (UCI schema)",
        type=["csv", "xlsx", "xls"],
        help=f"Required columns: {', '.join(REQUIRED_COLUMNS)}",
    )
    if uploaded is not None:
        with st.spinner("Validating and processing uploaded data..."):
            file_bytes = uploaded.getvalue()
            clean_df, quarantine_df, cleaning_summary, val_res = load_and_clean_data(
                "upload", file_bytes=file_bytes, filename=uploaded.name
            )
            if not val_res.is_valid:
                st.sidebar.error(f"Upload Rejected: {val_res.error_message}")
                st.stop()
    else:
        st.info("👈 Please upload a CSV or Excel file, or switch to 'Deterministic Demo Data' in the sidebar.")
        st.stop()

if clean_df is None or clean_df.empty:
    st.error("No valid transaction records available to analyze. Check Data Quality tab.")
    st.stop()

# Initialize DuckDB Manager
db = get_database_manager(clean_df)

# Filters: Date Range and Country
st.sidebar.subheader("2. Analytical Filters")
min_db_date, max_db_date = db.get_date_bounds()

if min_db_date and max_db_date:
    date_selection = st.sidebar.date_input(
        "Analysis Date Window",
        value=(min_db_date, max_db_date),
        min_value=min_db_date,
        max_value=max_db_date,
    )
    if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2:
        start_date, end_date = date_selection
    else:
        start_date, end_date = min_db_date, max_db_date
else:
    start_date, end_date = date(2024, 1, 1), date(2025, 12, 31)

all_countries = ["ALL"] + db.get_countries()
selected_country = st.sidebar.selectbox("Country Filter", all_countries, index=0)

# Sidebar Navigation
st.sidebar.subheader("3. Dashboard Pages")
page = st.sidebar.radio(
    "Navigate",
    [
        "A. Executive Overview",
        "B. Sales and Products",
        "C. Customer Segmentation",
        "D. Cohort Retention",
        "E. Sales Forecast",
        "F. Data Quality & Audit",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Documentation & Integrity**  \n"
    "• Currency: **GBP (£)**  \n"
    "• Database: **DuckDB (SQL Engine)**  \n"
    "• COGS / Profit: *Excluded (not present)*  \n"
    "• Analysis Grain: *Transaction line item*"
)


# ---------------------------------------------------------
# Banner Notification: Demo vs Real Data
# ---------------------------------------------------------
if is_demo:
    st.markdown(
        """
        <div class="demo-banner">
            ⚠️ <strong>Synthetic Demo Data Active:</strong>
            Loaded 24 months of deterministically generated retail transactions (seed=42)
            featuring seasonal patterns, cancellations, repeat cohorts, and intentional anomalies.
            Findings shown are for demonstration and portfolio testing only.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="real-banner">
            ✅ <strong>Custom / Uploaded Data Active:</strong>
            Loaded external dataset. Data has passed schema validation, quarantine filtering,
            and dimensional modeling.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# PAGE A: EXECUTIVE OVERVIEW
# ---------------------------------------------------------
if page == "A. Executive Overview":
    st.title("📊 Executive Overview")
    st.caption(
        f"Commercial performance summary from **{start_date.strftime('%d %b %Y')}** to "
        f"**{end_date.strftime('%d %b %Y')}** | Country: **{selected_country}**"
    )

    # Compute Period KPIs & Comparison
    comparison = compute_comparison(db, start_date, end_date, selected_country)
    curr = comparison.current
    prev = comparison.previous

    # Display KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        delta_str = f"{comparison.net_revenue_pct_change:+.1f}% vs prev period" if comparison.net_revenue_pct_change is not None else "Prior period N/A"
        delta_class = "kpi-delta-pos" if (comparison.net_revenue_pct_change or 0) >= 0 else "kpi-delta-neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Net Revenue (£)</div>
                <div class="kpi-value">£{curr.net_revenue:,.2f}</div>
                <div class="{delta_class}">{delta_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        delta_str = f"{comparison.gross_sales_pct_change:+.1f}% vs prev period" if comparison.gross_sales_pct_change is not None else "Prior period N/A"
        delta_class = "kpi-delta-pos" if (comparison.gross_sales_pct_change or 0) >= 0 else "kpi-delta-neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Gross Sales (£)</div>
                <div class="kpi-value">£{curr.gross_sales:,.2f}</div>
                <div class="{delta_class}">{delta_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        ret_pct = curr.value_return_rate_pct
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Return Value (£)</div>
                <div class="kpi-value">£{curr.return_value:,.2f}</div>
                <div class="kpi-delta-neutral">{ret_pct:.1f}% of gross sales</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        delta_str = f"{comparison.orders_pct_change:+.1f}% vs prev period" if comparison.orders_pct_change is not None else "Prior period N/A"
        delta_class = "kpi-delta-pos" if (comparison.orders_pct_change or 0) >= 0 else "kpi-delta-neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Completed Orders</div>
                <div class="kpi-value">{curr.completed_sales_orders:,}</div>
                <div class="{delta_class}">{delta_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        delta_str = f"{comparison.aov_pct_change:+.1f}% vs prev period" if comparison.aov_pct_change is not None else "Prior period N/A"
        delta_class = "kpi-delta-pos" if (comparison.aov_pct_change or 0) >= 0 else "kpi-delta-neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Average Order Value (AOV)</div>
                <div class="kpi-value">£{curr.average_order_value:.2f}</div>
                <div class="{delta_class}">{delta_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c6:
        delta_str = f"{comparison.customers_pct_change:+.1f}% vs prev period" if comparison.customers_pct_change is not None else "Prior period N/A"
        delta_class = "kpi-delta-pos" if (comparison.customers_pct_change or 0) >= 0 else "kpi-delta-neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Purchasing Customers</div>
                <div class="kpi-value">{curr.purchasing_customers:,}</div>
                <div class="{delta_class}">{delta_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c7:
        diff_str = f"{comparison.repeat_rate_diff_pp:+.1f} pp vs prev period" if comparison.repeat_rate_diff_pp is not None else "Prior period N/A"
        diff_class = "kpi-delta-pos" if (comparison.repeat_rate_diff_pp or 0) >= 0 else "kpi-delta-neg"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Repeat Purchase Rate</div>
                <div class="kpi-value">{curr.repeat_purchase_rate_pct:.1f}%</div>
                <div class="{diff_class}">{diff_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c8:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Units Sold</div>
                <div class="kpi-value">{curr.units_sold:,}</div>
                <div class="kpi-delta-neutral">{curr.returned_units:,} units returned</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Guest Checkout Attribution Notice
    if curr.missing_customer_orders_count > 0:
        st.caption(
            f"ℹ️ **Guest Checkout Attribution**: {curr.missing_customer_orders_count:,} orders "
            f"({curr.missing_customer_revenue_share_pct:.1f}% of gross sales) were placed without a Customer ID. "
            "These are included in total revenue/orders, but excluded from customer-level metrics."
        )

    st.markdown("---")

    # Time-Series Revenue Trends (Weekly & Monthly)
    st.subheader("📈 Revenue Trends Over Time")
    trend_tab1, trend_tab2 = st.tabs(["Weekly Net Revenue", "Monthly Net Revenue"])

    with trend_tab1:
        weekly_sql = """
        SELECT
            CAST(week_start AS DATE) AS week,
            COALESCE(SUM(CASE WHEN is_qualifying_sale THEN gross_amount ELSE 0.0 END), 0.0) AS gross_sales,
            COALESCE(SUM(CASE WHEN is_return THEN return_amount ELSE 0.0 END), 0.0) AS returns,
            (COALESCE(SUM(CASE WHEN is_qualifying_sale THEN gross_amount ELSE 0.0 END), 0.0) -
             COALESCE(SUM(CASE WHEN is_return THEN return_amount ELSE 0.0 END), 0.0)) AS net_revenue
        FROM fct_transactions
        WHERE date >= $start_date AND date <= $end_date
          AND ($country = 'ALL' OR country = $country)
        GROUP BY week_start
        ORDER BY week_start ASC;
        """
        weekly_df = db.query(weekly_sql, {"start_date": start_date, "end_date": end_date, "country": selected_country})
        if not weekly_df.empty:
            fig_w = px.line(
                weekly_df,
                x="week",
                y="net_revenue",
                title="Weekly Net Revenue (£)",
                markers=True,
                labels={"week": "Week Commencing", "net_revenue": "Net Revenue (£)"},
                template="plotly_white",
            )
            fig_w.update_traces(line_color="#2563eb", line_width=2.5)
            fig_w.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.info("No weekly sales data found for the selected filter.")

    with trend_tab2:
        monthly_sql = """
        SELECT
            year_month,
            COALESCE(SUM(CASE WHEN is_qualifying_sale THEN gross_amount ELSE 0.0 END), 0.0) AS gross_sales,
            COALESCE(SUM(CASE WHEN is_return THEN return_amount ELSE 0.0 END), 0.0) AS returns,
            (COALESCE(SUM(CASE WHEN is_qualifying_sale THEN gross_amount ELSE 0.0 END), 0.0) -
             COALESCE(SUM(CASE WHEN is_return THEN return_amount ELSE 0.0 END), 0.0)) AS net_revenue
        FROM fct_transactions
        WHERE date >= $start_date AND date <= $end_date
          AND ($country = 'ALL' OR country = $country)
        GROUP BY year_month
        ORDER BY year_month ASC;
        """
        monthly_df = db.query(monthly_sql, {"start_date": start_date, "end_date": end_date, "country": selected_country})
        if not monthly_df.empty:
            fig_m = px.bar(
                monthly_df,
                x="year_month",
                y="net_revenue",
                title="Monthly Net Revenue (£)",
                labels={"year_month": "Month", "net_revenue": "Net Revenue (£)"},
                template="plotly_white",
                color_discrete_sequence=["#3b82f6"],
            )
            fig_m.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("No monthly sales data found for the selected filter.")

    # Geographic and Top SKU Breakdowns
    st.markdown("---")
    st.subheader("🌐 Top Commercial Drivers")
    col_prod, col_geo = st.columns(2)

    with col_prod:
        st.markdown("**Top 10 Products by Gross Revenue**")
        top_p_sql = """
        SELECT
            p.description,
            t.stock_code,
            COALESCE(SUM(CASE WHEN t.is_qualifying_sale THEN t.gross_amount ELSE 0.0 END), 0.0) AS gross_sales,
            COALESCE(SUM(CASE WHEN t.is_qualifying_sale THEN t.quantity ELSE 0 END), 0) AS units_sold
        FROM fct_transactions t
        LEFT JOIN dim_products p ON t.stock_code = p.stock_code
        WHERE t.date >= $start_date AND t.date <= $end_date
          AND ($country = 'ALL' OR t.country = $country)
        GROUP BY p.description, t.stock_code
        ORDER BY gross_sales DESC
        LIMIT 10;
        """
        top_p_df = db.query(top_p_sql, {"start_date": start_date, "end_date": end_date, "country": selected_country})
        if not top_p_df.empty:
            fig_tp = px.bar(
                top_p_df.sort_values("gross_sales", ascending=True),
                x="gross_sales",
                y="description",
                orientation="h",
                labels={"gross_sales": "Gross Sales (£)", "description": "Product"},
                template="plotly_white",
                color_discrete_sequence=["#1e40af"],
            )
            fig_tp.update_layout(margin=dict(l=10, r=20, t=20, b=20), height=380)
            st.plotly_chart(fig_tp, use_container_width=True)
        else:
            st.info("No products recorded.")

    with col_geo:
        st.markdown("**Top 10 Markets by Net Revenue**")
        top_c_sql = """
        SELECT
            country,
            (COALESCE(SUM(CASE WHEN is_qualifying_sale THEN gross_amount ELSE 0.0 END), 0.0) -
             COALESCE(SUM(CASE WHEN is_return THEN return_amount ELSE 0.0 END), 0.0)) AS net_revenue
        FROM fct_transactions
        WHERE date >= $start_date AND date <= $end_date
        GROUP BY country
        ORDER BY net_revenue DESC
        LIMIT 10;
        """
        top_c_df = db.query(top_c_sql, {"start_date": start_date, "end_date": end_date})
        if not top_c_df.empty:
            fig_tc = px.bar(
                top_c_df.sort_values("net_revenue", ascending=True),
                x="net_revenue",
                y="country",
                orientation="h",
                labels={"net_revenue": "Net Revenue (£)", "country": "Country"},
                template="plotly_white",
                color_discrete_sequence=["#0d9488"],
            )
            fig_tc.update_layout(margin=dict(l=10, r=20, t=20, b=20), height=380)
            st.plotly_chart(fig_tc, use_container_width=True)
        else:
            st.info("No countries recorded.")

    # Business Insights Section
    st.markdown("---")
    st.subheader("💡 Automated Business Insights")
    st.caption("Rule-based, deterministic analytical findings computed directly from the current filter slice.")
    insights = generate_business_insights(db, comparison, selected_country)

    for ins in insights:
        st.markdown(
            f"""
            <div class="insight-card">
                <span class="insight-tag">{ins.category}</span>
                <div class="insight-title">{ins.title}</div>
                <div class="insight-obs">📌 Observation: {ins.numerical_observation}</div>
                <div class="insight-sec"><strong>Comparison Window:</strong> {ins.period_and_comparison}</div>
                <div class="insight-sec"><strong>Cautious Interpretation:</strong> {ins.cautious_interpretation}</div>
                <div class="insight-sec"><strong>Recommended Action:</strong> {ins.recommended_action}</div>
                <div class="insight-sec" style="color: #64748b; font-size: 0.82rem; margin-top: 6px;">
                    <em>⚠️ Limitation: {ins.important_limitation}</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# PAGE B: SALES AND PRODUCTS
# ---------------------------------------------------------
elif page == "B. Sales and Products":
    st.title("📦 Sales & Product Concentration")
    st.caption(
        "Pareto distribution, item-level revenue rankings, return rates, and product drill-downs."
    )

    # Load SQL query from sql/product_pareto.sql
    pareto_sql_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sql", "product_pareto.sql")
    with open(pareto_sql_path, "r", encoding="utf-8") as f:
        pareto_sql = f.read()

    pareto_df = db.query(
        pareto_sql,
        {"start_date": start_date, "end_date": end_date, "country": selected_country},
    )

    if pareto_df.empty:
        st.warning("No sales transactions found for the selected date window and country.")
    else:
        # Pareto Concentration Curve
        st.subheader("📐 Revenue Concentration (80/20 Pareto Curve)")
        st.markdown(
            "Visualizes cumulative share of revenue generated against cumulative share of catalog items. "
            "Identifies core volume drivers vs. long-tail catalog inventory."
        )

        fig_pareto = go.Figure()
        fig_pareto.add_trace(
            go.Scatter(
                x=pareto_df["cumulative_product_pct"],
                y=pareto_df["cumulative_revenue_pct"],
                mode="lines",
                name="Cumulative Revenue %",
                line=dict(color="#2563eb", width=3),
                hovertemplate="Product Top %: %{x:.1f}%<br>Cumulative Rev: %{y:.1f}%<extra></extra>",
            )
        )
        # 80% Benchmark Line
        fig_pareto.add_hline(
            y=80.0,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="80% Revenue Threshold",
            annotation_position="bottom right",
        )
        # Equality 45-degree line
        fig_pareto.add_trace(
            go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode="lines",
                name="Perfect Equality",
                line=dict(color="#94a3b8", dash="dot"),
            )
        )
        fig_pareto.update_layout(
            xaxis_title="Cumulative % of Products (Ranked by Revenue)",
            yaxis_title="Cumulative % of Gross Revenue",
            template="plotly_white",
            hovermode="x unified",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        # Core Products vs Long Tail Breakdown Cards
        core_items = pareto_df[pareto_df["pareto_class"] == "Top 80% (Core)"]
        tail_items = pareto_df[pareto_df["pareto_class"] == "Long Tail (Bottom 20%)"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Catalog SKUs Active", f"{len(pareto_df):,}")
        with col2:
            st.metric(
                "Core SKUs (Top 80% Sales)",
                f"{len(core_items):,} SKUs ({len(core_items) / len(pareto_df) * 100:.1f}%)",
            )
        with col3:
            st.metric(
                "Long Tail SKUs (Bottom 20% Sales)",
                f"{len(tail_items):,} SKUs ({len(tail_items) / len(pareto_df) * 100:.1f}%)",
            )

        st.markdown("---")
        st.subheader("🔍 Product Performance Table & Return Rates")
        st.caption(
            "Return rates state explicit denominators: Unit Return Rate = (Returned Units / Units Sold) * 100. "
            "Value Return Rate = (Return Value / Gross Revenue) * 100."
        )

        # Format display dataframe
        disp_df = pareto_df[[
            "product_rank",
            "stock_code",
            "description",
            "gross_revenue",
            "return_value",
            "net_revenue",
            "units_sold",
            "units_returned",
            "return_rate_units_pct",
            "order_appearances",
            "pareto_class",
        ]].copy()

        disp_df.columns = [
            "Rank",
            "Stock Code",
            "Description",
            "Gross Sales (£)",
            "Return Value (£)",
            "Net Revenue (£)",
            "Units Sold",
            "Units Returned",
            "Unit Return Rate (%)",
            "Order Count",
            "Pareto Category",
        ]

        st.dataframe(
            disp_df.style.format({
                "Gross Sales (£)": "£{:,.2f}",
                "Return Value (£)": "£{:,.2f}",
                "Net Revenue (£)": "£{:,.2f}",
                "Units Sold": "{:,}",
                "Units Returned": "{:,}",
                "Unit Return Rate (%)": "{:.1f}%",
                "Order Count": "{:,}",
            }),
            use_container_width=True,
            height=450,
        )

        # CSV Export
        csv_buffer = io.StringIO()
        disp_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export Product Analytics Table (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"retailpulse_products_{start_date}_{end_date}.csv",
            mime="text/csv",
        )

        st.markdown(
            """
            <div class="caveat-box">
                <strong>Methodological Caveat:</strong> Rankings and Pareto curves show descriptive historical associations.
                High return rates identify SKUs requiring packaging or listing inspection, but do not prove causal defect
                without returns customer survey data. Cost of Goods Sold (COGS) is not available; high revenue SKUs
                may not necessarily be the most profitable.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# PAGE C: CUSTOMER SEGMENTATION
# ---------------------------------------------------------
elif page == "C. Customer Segmentation":
    st.title("👥 Customer Segmentation (RFM Analysis)")
    st.caption(
        "Behavioral taxonomy based on Recency, Frequency, and Monetary Value of qualifying sales orders."
    )

    # Date Control: Explicit Analysis Date (Default: end_date + 1 day, never today's date!)
    col_d1, col_d2 = st.columns([2, 3])
    with col_d1:
        default_analysis_date = end_date + timedelta(days=1)
        analysis_date = st.date_input(
            "Reference Analysis Date",
            value=default_analysis_date,
            help="Recency is measured as days elapsed from customer's latest order to this reference date. Defaults to 1 day after window end.",
        )

    # Compute RFM Segmentation
    with st.spinner("Computing RFM scores and segment aggregations..."):
        cust_rfm_df, segment_summary_df = compute_rfm_segmentation(
            db=db,
            start_date=start_date,
            end_date=end_date,
            country=selected_country,
            analysis_date=analysis_date,
        )

    if cust_rfm_df.empty:
        st.warning("No registered customers with qualifying purchases found in this filter range.")
    else:
        # Segment High-Level Metrics
        st.subheader("📊 Segment Distribution & Revenue Contribution")
        c1, c2 = st.columns(2)
        with c1:
            fig_cust_pie = px.pie(
                segment_summary_df,
                names="Segment",
                values="Customer Count",
                title="Customer Share by Segment (%)",
                hole=0.4,
                template="plotly_white",
            )
            st.plotly_chart(fig_cust_pie, use_container_width=True)

        with c2:
            fig_rev_bar = px.bar(
                segment_summary_df,
                x="Revenue Share (%)",
                y="Segment",
                orientation="h",
                title="Revenue Contribution by Segment (%)",
                template="plotly_white",
                color="Revenue Share (%)",
                color_continuous_scale="Blues",
            )
            fig_rev_bar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_rev_bar, use_container_width=True)

        # 3D / 2D RFM Scatter Visualization
        st.subheader("🎯 RFM Behavioral Scatter")
        fig_scatter = px.scatter(
            cust_rfm_df,
            x="recency_days",
            y="frequency",
            size="monetary_value",
            color="Segment",
            hover_name="customer_id",
            hover_data=["monetary_value", "RFM_Score"],
            labels={
                "recency_days": f"Recency (Days since {analysis_date})",
                "frequency": "Frequency (Distinct Qualifying Orders)",
                "monetary_value": "Monetary Value (£)",
            },
            title="Customer Recency vs Frequency (Bubble Size = Spend)",
            template="plotly_white",
            height=460,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Segment Summary Table with Prescriptive Actions
        st.subheader("📋 Segment Profile & Suggested Commercial Actions")
        st.dataframe(
            segment_summary_df[[
                "Segment",
                "Customer Count",
                "Customer Share (%)",
                "Total Revenue",
                "Revenue Share (%)",
                "Avg Recency (Days)",
                "Avg Frequency",
                "Avg Monetary Value",
                "Recommended Action",
            ]].style.format({
                "Customer Count": "{:,}",
                "Customer Share (%)": "{:.1f}%",
                "Total Revenue": "£{:,.2f}",
                "Revenue Share (%)": "{:.1f}%",
                "Avg Recency (Days)": "{:.1f}",
                "Avg Frequency": "{:.1f}",
                "Avg Monetary Value": "£{:,.2f}",
            }),
            use_container_width=True,
            height=380,
        )

        st.markdown(
            """
            <div class="caveat-box">
                <strong>Methodological Disclaimer:</strong>
                RFM segments represent a rule-based descriptive taxonomy calculated from past transaction history.
                They are NOT probabilistic machine learning churn predictions or lifetime value guarantees.
                Guest checkouts without Customer IDs cannot be attributed to customer accounts and are excluded.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# PAGE D: COHORT RETENTION
# ---------------------------------------------------------
elif page == "D. Cohort Retention":
    st.title("🗓️ Customer Cohort Retention Analysis")
    st.caption(
        "Monthly repeat-purchase retention tracking customer cohorts based on their first observed purchase."
    )

    with st.spinner("Calculating monthly cohort retention matrix..."):
        pct_pivot, counts_pivot, cohort_sizes, notes = compute_cohort_retention(
            db=db,
            country=selected_country,
        )

    if pct_pivot.empty:
        st.warning("Insufficient transaction history to construct cohort retention matrices.")
    else:
        st.subheader("🔥 Monthly Repeat Purchase Retention Rate (%)")
        st.markdown(
            "Tracks the percentage of acquired customers who returned to place at least one qualifying sales order "
            "in subsequent calendar months. Unobserved future periods are kept blank."
        )

        # Heatmap plot
        z_vals = pct_pivot.values
        x_labels = list(pct_pivot.columns)
        y_labels = list(pct_pivot.index)

        fig_heat = px.imshow(
            pct_pivot,
            labels=dict(x="Activity Month Offset", y="Acquisition Cohort", color="Retention %"),
            x=x_labels,
            y=y_labels,
            color_continuous_scale="Blues",
            text_auto=".1f",
            aspect="auto",
            template="plotly_white",
        )
        fig_heat.update_layout(height=480, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Cohort Size Breakdown Table
        st.subheader("👥 Cohort Acquisition Sizes (Month 0 Volume)")
        st.dataframe(
            counts_pivot.style.format(lambda v: "" if pd.isna(v) else f"{int(v):,}", subset=counts_pivot.columns),
            use_container_width=True,
            height=350,
        )

        # Methodology Callout
        st.markdown(
            f"""
            <div class="caveat-box">
                <strong>Cohort Methodology & Analytical Notes:</strong><br>
                • <strong>Acquisition Definition:</strong> {notes['Acquisition Definition']}<br>
                • <strong>Left-Censoring:</strong> {notes['Left-Censoring Limitation']}<br>
                • <strong>Future Periods:</strong> {notes['Future Periods Representation']}<br>
                • <strong>Filter Context:</strong> {notes['Filter Mechanics']}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# PAGE E: SALES FORECAST
# ---------------------------------------------------------
elif page == "E. Sales Forecast":
    st.title("🔮 Demand & Sales Forecasting")
    st.caption(
        "Short-term weekly demand projections using chronological holdout validation and statsmodels."
    )

    with st.spinner("Aggregating weekly series and fitting forecasting models..."):
        fc_result: ForecastResult = generate_sales_forecast(
            db=db,
            country=selected_country,
            horizon_weeks=4,
            min_history_required=12,
        )

    if not fc_result.is_sufficient_history:
        st.warning(f"⚠️ {fc_result.status_message}")
        st.info("Try selecting 'ALL' countries or ensuring the full 24-month dataset is loaded.")
    else:
        # Out-of-sample Evaluation Comparison Table
        st.subheader("🧪 Out-of-Sample Holdout Validation (Last 4 Weeks)")
        st.markdown(
            "Rigorous chronological holdout validation. The model is trained strictly on earlier history "
            "and evaluated against the unobserved last 4 weeks. Zero random splits."
        )

        eval_data = [
            {
                "Model": fc_result.evaluation_stats.model_name,
                "MAE (£)": f"£{fc_result.evaluation_stats.mae:,.2f}",
                "WAPE (%)": f"{fc_result.evaluation_stats.wape_pct:.2f}%",
                "Selected": "✅ Selected Winner" if fc_result.selected_model_name == fc_result.evaluation_stats.model_name else "",
            },
            {
                "Model": fc_result.evaluation_naive.model_name,
                "MAE (£)": f"£{fc_result.evaluation_naive.mae:,.2f}",
                "WAPE (%)": f"{fc_result.evaluation_naive.wape_pct:.2f}%",
                "Selected": "✅ Selected Winner" if fc_result.selected_model_name == fc_result.evaluation_naive.model_name else "",
            },
        ]
        st.table(pd.DataFrame(eval_data))

        # Time-Series Chart: Actuals + Future Forecast Cone
        st.subheader("📈 4-Week Net Revenue Projection with 95% Confidence Intervals")

        hist_df = fc_result.historical_weekly_df
        future_df = fc_result.future_forecast_df

        fig_fc = go.Figure()

        # Historical Net Revenue
        fig_fc.add_trace(
            go.Scatter(
                x=hist_df["week_start"],
                y=hist_df["net_revenue"],
                mode="lines+markers",
                name="Historical Net Revenue",
                line=dict(color="#1e293b", width=2),
            )
        )

        # Forecast Line
        # Bridge last historical point to first forecast point
        last_hist_date = hist_df["week_start"].iloc[-1]
        last_hist_rev = hist_df["net_revenue"].iloc[-1]

        fc_dates = [last_hist_date] + list(future_df["week_start"])
        fc_vals = [last_hist_rev] + list(future_df["forecast_net_revenue"])
        ci_uppers = [last_hist_rev] + list(future_df["ci_upper_95"])
        ci_lowers = [last_hist_rev] + list(future_df["ci_lower_95"])

        fig_fc.add_trace(
            go.Scatter(
                x=fc_dates,
                y=fc_vals,
                mode="lines+markers",
                name=f"Forecast ({fc_result.selected_model_name})",
                line=dict(color="#2563eb", width=2.5, dash="dash"),
            )
        )

        # 95% Prediction Interval Ribbon
        fig_fc.add_trace(
            go.Scatter(
                x=fc_dates + fc_dates[::-1],
                y=ci_uppers + ci_lowers[::-1],
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="95% Prediction Interval",
            )
        )

        fig_fc.update_layout(
            xaxis_title="Week Commencing",
            yaxis_title="Net Revenue (£)",
            template="plotly_white",
            hovermode="x unified",
            height=460,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Future Weekly Point Forecast Table
        st.subheader("📅 Projected Weekly Demand Numbers")
        st.dataframe(
            future_df.style.format({
                "forecast_net_revenue": "£{:,.2f}",
                "ci_lower_95": "£{:,.2f}",
                "ci_upper_95": "£{:,.2f}",
            }),
            use_container_width=True,
        )

        st.markdown(
            f"""
            <div class="caveat-box">
                <strong>Forecasting Assumptions & Limitations:</strong><br>
                • <strong>Validation:</strong> {fc_result.notes['Methodology']}<br>
                • <strong>Baseline Projection:</strong> {fc_result.notes['Interpretation']}<br>
                • <strong>Operational Risks:</strong> {fc_result.notes['Limitation']}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# PAGE F: DATA QUALITY & AUDIT
# ---------------------------------------------------------
elif page == "F. Data Quality & Audit":
    st.title("🛡️ Data Quality, Quarantine & Data Dictionary")
    st.caption(
        "Audit trail of input records, deduplication, quarantine filters, and unified metric definitions."
    )

    if cleaning_summary:
        st.subheader("📋 Ingestion & Transformation Ledger")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("Total Raw Rows Ingested", f"{cleaning_summary.total_input_rows:,}")
        with q2:
            st.metric("Exact Duplicates Deduplicated", f"{cleaning_summary.exact_duplicates_removed:,}")
        with q3:
            st.metric("Quarantined Invalid Rows", f"{cleaning_summary.quarantined_rows:,}")
        with q4:
            st.metric("Retained Analytical Rows", f"{cleaning_summary.retained_clean_rows:,}")

        # Quarantine Reasons Breakdown
        st.markdown("---")
        st.subheader("🚫 Quarantined Records by Reason")
        st.caption(
            "Records quarantined due to missing/negative unit prices, unparseable timestamps, or missing quantities."
        )

        if cleaning_summary.quarantine_reasons:
            reasons_df = pd.DataFrame(
                list(cleaning_summary.quarantine_reasons.items()),
                columns=["Quarantine Reason", "Record Count"],
            )
            st.table(reasons_df)
        else:
            st.success("Zero records were quarantined. All rows satisfied data constraints.")

        if quarantine_df is not None and not quarantine_df.empty:
            with st.expander("🔍 Inspect Quarantined Row Samples"):
                st.dataframe(quarantine_df.head(50), use_container_width=True)

        # Missing Customer IDs & Guest Orders
        st.markdown("---")
        st.subheader("👤 Missing Customer ID Exposure")
        st.write(
            f"• **Missing Customer ID Rows:** {cleaning_summary.missing_customer_id_rows:,} rows "
            f"({cleaning_summary.missing_customer_id_pct_rows:.2f}% of retained transactions)  \n"
            f"• **Associated Gross Sales:** £{cleaning_summary.missing_customer_id_revenue:,.2f} "
            f"({cleaning_summary.missing_customer_id_pct_revenue:.2f}% of total gross revenue)  \n"
            f"• **Analytical Handling:** Retained in Sales, Products, and Financial KPIs. Excluded from "
            "Customer Segmentation (RFM) and Cohort Retention to prevent anonymous aggregation distortion."
        )

        # Downloadable Quality Report
        st.markdown("---")
        st.subheader("📥 Download Data Quality Audit Report")
        audit_md = f"""# RetailPulse Data Quality Audit Report
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Data Source: {'Deterministic Synthetic Demo Data' if is_demo else 'Custom Upload'}
- Total Input Rows: {cleaning_summary.total_input_rows:,}
- Exact Duplicates Removed: {cleaning_summary.exact_duplicates_removed:,}
- Quarantined Records: {cleaning_summary.quarantined_rows:,}
- Retained Clean Analytical Rows: {cleaning_summary.retained_clean_rows:,}
- Missing Customer ID Rows: {cleaning_summary.missing_customer_id_rows:,} ({cleaning_summary.missing_customer_id_pct_rows:.2f}%)
- Missing Customer ID Revenue: £{cleaning_summary.missing_customer_id_revenue:,.2f} ({cleaning_summary.missing_customer_id_pct_revenue:.2f}%)
- Date Range Coverage: {cleaning_summary.min_date} to {cleaning_summary.max_date}

## Quarantine Breakdown
{pd.DataFrame(list(cleaning_summary.quarantine_reasons.items()), columns=['Reason', 'Count']).to_markdown(index=False) if cleaning_summary.quarantine_reasons else 'None'}
"""
        st.download_button(
            label="Download Quality Audit Report (.md)",
            data=audit_md,
            file_name="retailpulse_data_quality_report.md",
            mime="text/markdown",
        )

    # Unified KPI Data Dictionary
    st.markdown("---")
    st.subheader("📖 Unified KPI Data Dictionary")
    st.caption("Standardized definitions and mathematical formulas utilized throughout RetailPulse.")

    dict_rows = []
    for kpi, meta in KPI_DATA_DICTIONARY.items():
        dict_rows.append({
            "KPI Name": kpi,
            "Mathematical Formula": meta["formula"],
            "Unit": meta["unit"],
            "Business Definition": meta["definition"],
            "Interpretation": meta["interpretation"],
        })
    st.dataframe(pd.DataFrame(dict_rows), use_container_width=True, height=420)

    # Official Target Dataset Attribution
    st.markdown("---")
    st.subheader("📚 Target Dataset Citation & Attribution")
    st.markdown(
        f"""
        • **Dataset Name**: {UCI_METADATA['dataset_name']}  \n
        • **Donor**: {UCI_METADATA['donator']}  \n
        • **Official Citation**: {UCI_METADATA['citation']}  \n
        • **Repository Page**: [{UCI_METADATA['source_url']}]({UCI_METADATA['source_url']})  \n
        • **License**: {UCI_METADATA['license']}  \n
        • **Direct Download**: `{UCI_METADATA['download_url']}`  \n
        • **Currency**: British Pound ({UCI_METADATA['currency']})
        """
    )
