"""
RetailPulse - Deterministic Business Insights Engine

Generates dynamic, rule-based business insight cards computed from the active
filtered data without external LLM dependencies.

Strict Schema for Every Insight:
1. Numerical Observation.
2. Relevant Period and Comparison.
3. Cautious Interpretation.
4. Recommended Business Action.
5. Important Limitation.
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional
import pandas as pd
from src.database import DatabaseManager
from src.metrics import KPIComparison, compute_comparison


@dataclass
class BusinessInsight:
    title: str
    category: str  # Growth, Concentration, Return Risk, Customer Base
    numerical_observation: str
    period_and_comparison: str
    cautious_interpretation: str
    recommended_action: str
    important_limitation: str


def generate_business_insights(
    db: DatabaseManager,
    comparison: KPIComparison,
    country: str = "ALL",
) -> List[BusinessInsight]:
    """
    Computes deterministic, structured insight cards from active filter state.
    """
    insights: List[BusinessInsight] = []
    curr = comparison.current
    prev = comparison.previous

    if curr.completed_sales_orders == 0:
        return [
            BusinessInsight(
                title="Zero Sales Activity in Selected Window",
                category="System Notice",
                numerical_observation="0 completed orders and £0.00 net revenue recorded.",
                period_and_comparison=f"{curr.start_date} to {curr.end_date} (Country: {country})",
                cautious_interpretation="No qualifying commercial sales took place within this window.",
                recommended_action="Widen the date range or check country filters to examine active transaction windows.",
                important_limitation="Empty records prevent meaningful statistical comparisons.",
            )
        ]

    # 1. Top-line Revenue & Order Momentum Insight
    if prev is not None and prev.net_revenue > 0:
        net_pct = comparison.net_revenue_pct_change or 0.0
        orders_pct = comparison.orders_pct_change or 0.0
        aov_pct = comparison.aov_pct_change or 0.0

        if net_pct >= 0:
            trend_desc = f"Net revenue expanded by +{net_pct:.1f}% (£{curr.net_revenue:,.2f} vs £{prev.net_revenue:,.2f})."
            driver = "order volume expansion" if orders_pct > aov_pct else "basket size expansion (AOV)"
            interp = (
                f"Top-line growth was primarily accompanied by {driver} "
                f"(Orders: {orders_pct:+.1f}%, AOV: {aov_pct:+.1f}%)."
            )
            action = (
                "Protect inventory availability on top-performing lines and assess whether "
                "promotional discounting or seasonal demand drove the volume lift."
            )
        else:
            trend_desc = f"Net revenue contracted by {net_pct:.1f}% (£{curr.net_revenue:,.2f} vs £{prev.net_revenue:,.2f})."
            driver = "fewer order transactions" if orders_pct < aov_pct else "smaller basket values"
            interp = (
                f"Revenue compression was driven by {driver} "
                f"(Orders: {orders_pct:+.1f}%, AOV: {aov_pct:+.1f}%)."
            )
            action = (
                "Review conversion funnels and re-engage lapsed customer segments with "
                "targeted promotional reminders or free delivery incentives."
            )

        insights.append(
            BusinessInsight(
                title="Top-line Period-over-Period Performance",
                category="Growth & Revenue",
                numerical_observation=trend_desc,
                period_and_comparison=(
                    f"Selected period ({curr.start_date} to {curr.end_date}) vs. "
                    f"previous equal-length period ({prev.start_date} to {prev.end_date})"
                ),
                cautious_interpretation=interp,
                recommended_action=action,
                important_limitation=(
                    "Historical comparisons reflect descriptive associations only; external factors "
                    "such as seasonality, marketing budget shifts, and out-of-stock periods are not controlled."
                ),
            )
        )

    # 2. Product Revenue Concentration / Pareto Risk Insight
    pareto_sql = """
    WITH product_rev AS (
        SELECT
            stock_code,
            COALESCE(SUM(gross_amount), 0.0) AS gross_rev
        FROM fct_transactions
        WHERE is_qualifying_sale
          AND date >= $start_date AND date <= $end_date
          AND ($country = 'ALL' OR country = $country)
        GROUP BY stock_code
    ),
    ranked AS (
        SELECT
            stock_code,
            gross_rev,
            ROW_NUMBER() OVER (ORDER BY gross_rev DESC) AS rn,
            COUNT(*) OVER () AS total_skus,
            SUM(gross_rev) OVER () AS total_rev,
            SUM(gross_rev) OVER (ORDER BY gross_rev DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_rev
        FROM product_rev
    )
    SELECT
        COUNT(*) AS total_products,
        COALESCE(MAX(total_rev), 0.0) AS total_revenue,
        COUNT(CASE WHEN cum_rev <= (0.80 * total_rev) THEN 1 END) AS top_80_skus
    FROM ranked;
    """
    pareto_df = db.query(pareto_sql, {
        "start_date": curr.start_date,
        "end_date": curr.end_date,
        "country": country,
    })

    if not pareto_df.empty and pareto_df.iloc[0]["total_products"] > 0:
        row_p = pareto_df.iloc[0]
        tot_skus = int(row_p["total_products"])
        top_skus = max(1, int(row_p["top_80_skus"]))
        sku_pct = (top_skus / tot_skus) * 100.0 if tot_skus > 0 else 0.0

        insights.append(
            BusinessInsight(
                title="Catalog Concentration (Pareto Distribution)",
                category="Catalog Concentration",
                numerical_observation=(
                    f"Top {top_skus} product SKU{'s' if top_skus != 1 else ''} ({sku_pct:.1f}% of catalog) "
                    f"generate approximately 80% of total gross sales."
                ),
                period_and_comparison=f"{curr.start_date} to {curr.end_date} (Active Filter)",
                cautious_interpretation=(
                    "Commercial revenue exhibits high product dependency. A supplier delay or "
                    "stockout on a handful of top SKUs would create immediate top-line vulnerability."
                ),
                recommended_action=(
                    "Establish safety stock buffers and dual-supplier sourcing for top core items, "
                    "while evaluating clearance or bundle strategies for stagnant long-tail SKUs."
                ),
                important_limitation=(
                    "Gross revenue does not measure gross margin or net profitability because product unit "
                    "costs (COGS) are not present in this dataset."
                ),
            )
        )

    # 3. Merchandise Returns & Reverse Logistics Insight
    ret_rate = curr.value_return_rate_pct
    unit_ret_rate = curr.unit_return_rate_pct
    ret_val = curr.return_value

    if ret_val > 0:
        interp_ret = (
            f"Returned items accounted for {ret_rate:.1f}% of gross sales value and "
            f"{unit_ret_rate:.1f}% of dispatched units."
        )
        if ret_rate > 7.0:
            urgency = "Elevated return rates indicate potential product defect, sizing ambiguity, or buyer remorse."
            action_ret = "Audit product listings for high-return items to clarify dimensions, descriptions, and packaging durability."
        else:
            urgency = "Return rates are within normal retail gift benchmark ranges."
            action_ret = "Monitor return reasons on emerging high-volume items to maintain quality control."

        insights.append(
            BusinessInsight(
                title="Merchandise Returns & Quality Signal",
                category="Return Risk",
                numerical_observation=f"£{ret_val:,.2f} returned across {curr.returned_units:,} units ({ret_rate:.1f}% value return rate).",
                period_and_comparison=f"{curr.start_date} to {curr.end_date}",
                cautious_interpretation=f"{interp_ret} {urgency}",
                recommended_action=action_ret,
                important_limitation=(
                    "Cancellation records can lag original purchases by several weeks; returns processed "
                    "in this window may originate from orders placed in prior months."
                ),
            )
        )

    # 4. Customer Retention & Guest Checkout Exposure
    purch_custs = curr.purchasing_customers
    repeat_rate = curr.repeat_purchase_rate_pct
    guest_rev_pct = curr.missing_customer_revenue_share_pct

    insights.append(
        BusinessInsight(
            title="Customer Retention & Guest Account Exposure",
            category="Customer Base",
            numerical_observation=(
                f"{purch_custs:,} known customers active with a {repeat_rate:.1f}% repeat purchase rate; "
                f"{guest_rev_pct:.1f}% of gross revenue came from guest checkouts (missing Customer ID)."
            ),
            period_and_comparison=f"{curr.start_date} to {curr.end_date}",
            cautious_interpretation=(
                "A significant fraction of transaction volume occurs without registered customer profiles, "
                "which reduces customer-level attribution and biases cohort/RFM metrics toward registered users."
            ),
            recommended_action=(
                "Implement post-purchase account creation incentives (e.g. order tracking and loyalty points) "
                "to convert anonymous guest buyers into identified, trackable accounts."
            ),
            important_limitation=(
                "Guest checkout transactions are safely included in total sales and revenue figures, "
                "but are necessarily excluded from individual customer retention and segmentation analyses."
            ),
        )
    )

    return insights
