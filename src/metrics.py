"""
RetailPulse - Standardized KPI & Metrics Definitions

Provides unified KPI definitions, data dictionaries, and period-over-period
comparison computations with robust zero-denominator handling.

Cost/Margin Rule:
Product costs (COGS) are NOT available in the UCI Online Retail dataset.
Therefore, Gross Margin, Net Profit, and COGS are explicitly excluded to prevent
speculative or manufactured business claims.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from src.database import DatabaseManager

KPI_DATA_DICTIONARY = {
    "Gross Sales": {
        "definition": "Total monetary value of positive-quantity sales lines at positive prices before any returns or adjustments.",
        "formula": "SUM(Quantity * UnitPrice) WHERE Quantity > 0 AND UnitPrice > 0 AND NOT is_return",
        "unit": "GBP (£)",
        "interpretation": "Represents the top-line demand and gross billing generated during the period.",
    },
    "Return Value": {
        "definition": "Total monetary value of merchandise returned or cancelled by customers during the selected period.",
        "formula": "SUM(ABS(Quantity) * UnitPrice) WHERE is_return = TRUE",
        "unit": "GBP (£)",
        "interpretation": "Deducted from gross sales. Reflects returns processed within the period; return timing may lag original purchases.",
    },
    "Net Revenue": {
        "definition": "Gross sales minus return value for the selected period.",
        "formula": "Gross Sales - Return Value",
        "unit": "GBP (£)",
        "interpretation": "Core retained revenue figure for financial and commercial tracking.",
    },
    "Completed Sales Orders": {
        "definition": "Count of distinct sales invoice identifiers with at least one qualifying sales item.",
        "formula": "COUNT(DISTINCT InvoiceNo) WHERE is_qualifying_sale = TRUE",
        "unit": "Orders",
        "interpretation": "Volume of order transactions completed (cancellation invoices excluded).",
    },
    "Average Order Value (AOV)": {
        "definition": "Average gross transaction value per completed sales order.",
        "formula": "Gross Sales / Completed Sales Orders (0.0 if orders == 0)",
        "unit": "GBP (£) / Order",
        "interpretation": "Key indicator of basket size and pricing mix.",
    },
    "Purchasing Customers": {
        "definition": "Distinct known registered customers who made at least one qualifying sales order during the period.",
        "formula": "COUNT(DISTINCT CustomerID) WHERE is_qualifying_sale AND CustomerID IS NOT NULL",
        "unit": "Customers",
        "interpretation": "Measures active customer base. Excludes guest checkouts without CustomerID.",
    },
    "Repeat Purchase Rate": {
        "definition": "Percentage of purchasing customers who placed 2 or more distinct sales orders within the selected window.",
        "formula": "(Purchasing Customers with >= 2 Orders / Total Purchasing Customers) * 100",
        "unit": "%",
        "interpretation": "Evaluates customer loyalty and re-order velocity within the observed time horizon.",
    },
    "Units Sold": {
        "definition": "Total volume of positive merchandise units dispatched.",
        "formula": "SUM(Quantity) WHERE is_qualifying_sale = TRUE",
        "unit": "Units",
        "interpretation": "Operational fulfillment volume.",
    },
    "Returned Units": {
        "definition": "Total volume of returned items.",
        "formula": "SUM(ABS(Quantity)) WHERE is_return = TRUE",
        "unit": "Units",
        "interpretation": "Operational return reverse-logistics volume.",
    },
    "Unit Return Rate": {
        "definition": "Returned units expressed as a percentage of gross units sold.",
        "formula": "(Returned Units / Units Sold) * 100 (0.0 if Units Sold == 0)",
        "unit": "%",
        "interpretation": "Product quality and customer expectation alignment signal.",
    },
}


@dataclass
class PeriodKPIs:
    start_date: date
    end_date: date
    country: str
    gross_sales: float = 0.0
    return_value: float = 0.0
    net_revenue: float = 0.0
    completed_sales_orders: int = 0
    average_order_value: float = 0.0
    purchasing_customers: int = 0
    repeat_purchase_rate_pct: float = 0.0
    repeat_customers_count: int = 0
    units_sold: int = 0
    returned_units: int = 0
    unit_return_rate_pct: float = 0.0
    value_return_rate_pct: float = 0.0
    missing_customer_orders_count: int = 0
    missing_customer_revenue_share_pct: float = 0.0


@dataclass
class KPIComparison:
    current: PeriodKPIs
    previous: Optional[PeriodKPIs]
    # Percent changes (None if previous period was zero or nonexistent)
    gross_sales_pct_change: Optional[float] = None
    net_revenue_pct_change: Optional[float] = None
    orders_pct_change: Optional[float] = None
    aov_pct_change: Optional[float] = None
    customers_pct_change: Optional[float] = None
    repeat_rate_diff_pp: Optional[float] = None  # Percentage point difference
    units_sold_pct_change: Optional[float] = None


def compute_kpis_for_period(
    db: DatabaseManager,
    start_date: date,
    end_date: date,
    country: str = "ALL",
) -> PeriodKPIs:
    """
    Computes all standard KPIs for a specified date range and country filter using DuckDB.
    Handles empty filters and zero denominators safely.
    """
    sql = """
    WITH filtered_tx AS (
        SELECT *
        FROM fct_transactions
        WHERE date >= $start_date
          AND date <= $end_date
          AND ($country = 'ALL' OR country = $country)
    ),
    sales_agg AS (
        SELECT
            COALESCE(SUM(CASE WHEN is_qualifying_sale THEN gross_amount ELSE 0.0 END), 0.0) AS gross_sales,
            COALESCE(SUM(CASE WHEN is_return THEN return_amount ELSE 0.0 END), 0.0) AS return_value,
            COALESCE(SUM(CASE WHEN is_qualifying_sale THEN quantity ELSE 0 END), 0) AS units_sold,
            COALESCE(SUM(CASE WHEN is_return THEN ABS(quantity) ELSE 0 END), 0) AS returned_units,
            COUNT(DISTINCT CASE WHEN is_qualifying_sale THEN invoice_no END) AS completed_orders,
            COUNT(DISTINCT CASE WHEN is_qualifying_sale AND has_customer_id THEN customer_id END) AS purchasing_custs,
            COUNT(DISTINCT CASE WHEN is_qualifying_sale AND NOT has_customer_id THEN invoice_no END) AS guest_orders,
            COALESCE(SUM(CASE WHEN is_qualifying_sale AND NOT has_customer_id THEN gross_amount ELSE 0.0 END), 0.0) AS guest_revenue
        FROM filtered_tx
    ),
    repeat_custs AS (
        SELECT customer_id
        FROM filtered_tx
        WHERE is_qualifying_sale AND has_customer_id
        GROUP BY customer_id
        HAVING COUNT(DISTINCT invoice_no) >= 2
    )
    SELECT
        s.*,
        (SELECT COUNT(*) FROM repeat_custs) AS repeat_count
    FROM sales_agg s;
    """

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "country": country,
    }

    df = db.query(sql, params)

    if df.empty:
        return PeriodKPIs(start_date=start_date, end_date=end_date, country=country)

    row = df.iloc[0]
    gross = float(row["gross_sales"])
    returns = float(row["return_value"])
    net = gross - returns
    orders = int(row["completed_orders"])
    custs = int(row["purchasing_custs"])
    units = int(row["units_sold"])
    ret_units = int(row["returned_units"])
    rep_count = int(row["repeat_count"])
    guest_rev = float(row["guest_revenue"])

    aov = (gross / orders) if orders > 0 else 0.0
    repeat_pct = ((rep_count / custs) * 100.0) if custs > 0 else 0.0
    unit_ret_pct = ((ret_units / units) * 100.0) if units > 0 else 0.0
    val_ret_pct = ((returns / gross) * 100.0) if gross > 0 else 0.0
    guest_rev_pct = ((guest_rev / gross) * 100.0) if gross > 0 else 0.0

    return PeriodKPIs(
        start_date=start_date,
        end_date=end_date,
        country=country,
        gross_sales=gross,
        return_value=returns,
        net_revenue=net,
        completed_sales_orders=orders,
        average_order_value=aov,
        purchasing_customers=custs,
        repeat_purchase_rate_pct=repeat_pct,
        repeat_customers_count=rep_count,
        units_sold=units,
        returned_units=ret_units,
        unit_return_rate_pct=unit_ret_pct,
        value_return_rate_pct=val_ret_pct,
        missing_customer_orders_count=int(row["guest_orders"]),
        missing_customer_revenue_share_pct=guest_rev_pct,
    )


def compute_comparison(
    db: DatabaseManager,
    start_date: date,
    end_date: date,
    country: str = "ALL",
) -> KPIComparison:
    """
    Computes current period KPIs and compares them against the immediately preceding
    equal-length historical period.
    """
    current_kpis = compute_kpis_for_period(db, start_date, end_date, country)

    duration = (end_date - start_date).days + 1
    prev_end_date = start_date - timedelta(days=1)
    prev_start_date = prev_end_date - timedelta(days=duration - 1)

    # Check bounds in db
    min_bound, _ = db.get_date_bounds()

    if min_bound is None or prev_end_date < min_bound:
        # Preceding period out of bounds
        return KPIComparison(current=current_kpis, previous=None)

    prev_kpis = compute_kpis_for_period(db, prev_start_date, prev_end_date, country)

    def calc_pct_change(curr: float, prev: float) -> Optional[float]:
        if prev == 0.0:
            return None if curr == 0.0 else 100.0
        return ((curr - prev) / prev) * 100.0

    gross_chg = calc_pct_change(current_kpis.gross_sales, prev_kpis.gross_sales)
    net_chg = calc_pct_change(current_kpis.net_revenue, prev_kpis.net_revenue)
    orders_chg = calc_pct_change(float(current_kpis.completed_sales_orders), float(prev_kpis.completed_sales_orders))
    aov_chg = calc_pct_change(current_kpis.average_order_value, prev_kpis.average_order_value)
    custs_chg = calc_pct_change(float(current_kpis.purchasing_customers), float(prev_kpis.purchasing_customers))
    rep_diff = current_kpis.repeat_purchase_rate_pct - prev_kpis.repeat_purchase_rate_pct
    units_chg = calc_pct_change(float(current_kpis.units_sold), float(prev_kpis.units_sold))

    return KPIComparison(
        current=current_kpis,
        previous=prev_kpis,
        gross_sales_pct_change=gross_chg,
        net_revenue_pct_change=net_chg,
        orders_pct_change=orders_chg,
        aov_pct_change=aov_chg,
        customers_pct_change=custs_chg,
        repeat_rate_diff_pp=rep_diff,
        units_sold_pct_change=units_chg,
    )
