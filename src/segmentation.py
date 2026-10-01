"""
RetailPulse - Customer Segmentation Module (RFM Analysis)

Implements:
1. RFM calculation strictly using qualifying positive sales orders and known CustomerIDs.
2. Explicit analysis date (defaulting to 1 day after selected period end, NEVER today's date).
3. Robust quantile scoring handling tied values (e.g. many one-time buyers).
4. Deterministic business segmentation mapping (Champions, Loyal, At Risk, etc.).
5. Segment metrics: customer count, sales contribution, and recommended business actions.
6. Clear disclaimer: Rule-based behavioral taxonomy, NOT a probabilistic churn prediction.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.database import DatabaseManager

SEGMENT_DEFINITIONS = {
    "Champions": {
        "description": "Bought recently, buy often, and spend the most.",
        "action": "Reward with VIP perks, early access to new lines, and referral incentives.",
    },
    "Loyal Customers": {
        "description": "Buy on a regular basis with solid lifetime value.",
        "action": "Upsell higher-value products and engage with loyalty reward schemes.",
    },
    "Potential Loyalists": {
        "description": "Recent buyers with average frequency and above-average spend.",
        "action": "Offer membership benefits or personalized product recommendations.",
    },
    "New Customers": {
        "description": "Bought recently for the first time.",
        "action": "Provide onboarding support, satisfaction check-ins, and next-purchase discount vouchers.",
    },
    "Promising": {
        "description": "Recent buyers with lower initial spend.",
        "action": "Create brand awareness and offer cross-sell bundle recommendations.",
    },
    "Need Attention": {
        "description": "Above average recency and frequency, but haven't purchased in a while.",
        "action": "Send limited-time reactivation offers and re-engagement campaigns.",
    },
    "At Risk": {
        "description": "Spent big money and purchased often in the past, but long time since last purchase.",
        "action": "Deploy high-value win-back campaigns and personalized check-ins.",
    },
    "About to Sleep": {
        "description": "Below average recency and frequency. At risk of becoming inactive.",
        "action": "Share popular product highlights and special clearance incentives.",
    },
    "Hibernating": {
        "description": "Low recency and low frequency. Long dormant.",
        "action": "Offer relevant discounts or low-cost automated email reactivation.",
    },
    "Lost": {
        "description": "Lowest recency, frequency, and monetary scores.",
        "action": "Revive with aggressive discount promotions or exclude from paid remarketing.",
    },
}


def score_rfm_quantiles(series: pd.Series, ascending: bool = True, num_bins: int = 4) -> pd.Series:
    """
    Assigns 1-num_bins integer score using rank-based quantiles.
    Handling ties: rank(method='first') resolves tied values (e.g. Frequency=1 for 60% of buyers)
    without raising ValueError on duplicate bin edges.

    - ascending=True (for Frequency and Monetary): higher value -> higher score (4 is highest).
    - ascending=False (for Recency): lower days since last purchase -> higher score (4 is most recent).
    """
    if series.empty:
        return pd.Series(dtype=int)

    n_unique = series.nunique()
    if n_unique == 1:
        return pd.Series(num_bins if ascending else 1, index=series.index)

    # Rank with method='first' to avoid duplicate quantile bin edges
    ranks = series.rank(method="first", ascending=ascending)
    scores = pd.qcut(ranks, q=min(num_bins, len(series)), labels=False, duplicates="drop")
    return scores + 1  # 1-indexed (1 to num_bins)


def assign_rfm_segment(r: int, f: int, m: int) -> str:
    """
    Maps (R, F, M) quartile scores (1-4) to documented business customer segments.
    """
    # 4-tier logic
    if r >= 4 and f >= 4 and m >= 3:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal Customers"
    if r >= 3 and f in [1, 2] and m >= 2:
        return "Potential Loyalists"
    if r >= 4 and f == 1:
        return "New Customers"
    if r == 3 and f == 1:
        return "Promising"
    if r == 2 and f >= 3:
        return "At Risk"
    if r == 2 and f == 2:
        return "Need Attention"
    if r == 2 and f == 1:
        return "About to Sleep"
    if r == 1 and f >= 2:
        return "Lost"
    # r == 1, f == 1
    return "Hibernating"


def compute_rfm_segmentation(
    db: DatabaseManager,
    start_date: date,
    end_date: date,
    country: str = "ALL",
    analysis_date: Optional[date] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes RFM scores and segments for known customers.

    Parameters:
    - analysis_date: defaults to end_date + 1 day (never today's date).

    Returns:
    - customer_rfm_df: Customer-level RFM metrics and segment labels.
    - segment_summary_df: Aggregate segment metrics, revenue share, and action plans.
    """
    if analysis_date is None:
        analysis_date = end_date + timedelta(days=1)

    sql = """
    WITH filtered_orders AS (
        SELECT
            t.customer_id,
            t.invoice_no,
            t.date,
            SUM(t.gross_amount) AS order_gross
        FROM fct_transactions t
        WHERE t.is_qualifying_sale
          AND t.has_customer_id
          AND t.date >= $start_date
          AND t.date <= $end_date
          AND ($country = 'ALL' OR t.country = $country)
        GROUP BY t.customer_id, t.invoice_no, t.date
    )
    SELECT
        customer_id,
        DATE_DIFF('day', MAX(date), $analysis_date) AS recency_days,
        COUNT(DISTINCT invoice_no) AS frequency,
        COALESCE(SUM(order_gross), 0.0) AS monetary_value,
        MIN(date) AS first_order_date,
        MAX(date) AS last_order_date
    FROM filtered_orders
    GROUP BY customer_id
    ORDER BY monetary_value DESC;
    """

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "country": country,
        "analysis_date": analysis_date,
    }

    df = db.query(sql, params)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Recency: lower is better (ascending=False -> lower days gets score 4)
    df["R_score"] = score_rfm_quantiles(df["recency_days"], ascending=False, num_bins=4)
    # Frequency: higher is better
    df["F_score"] = score_rfm_quantiles(df["frequency"], ascending=True, num_bins=4)
    # Monetary: higher is better
    df["M_score"] = score_rfm_quantiles(df["monetary_value"], ascending=True, num_bins=4)

    df["RFM_Score"] = (
        df["R_score"].astype(str) + df["F_score"].astype(str) + df["M_score"].astype(str)
    )

    df["Segment"] = [
        assign_rfm_segment(r, f, m)
        for r, f, m in zip(df["R_score"], df["F_score"], df["M_score"])
    ]

    total_customers = len(df)
    total_revenue = df["monetary_value"].sum()

    # Compute segment summaries
    summary_list = []
    for seg_name, meta in SEGMENT_DEFINITIONS.items():
        sub = df[df["Segment"] == seg_name]
        cnt = len(sub)
        seg_rev = sub["monetary_value"].sum() if cnt > 0 else 0.0
        avg_rec = sub["recency_days"].mean() if cnt > 0 else 0.0
        avg_freq = sub["frequency"].mean() if cnt > 0 else 0.0
        avg_mon = sub["monetary_value"].mean() if cnt > 0 else 0.0

        summary_list.append({
            "Segment": seg_name,
            "Customer Count": cnt,
            "Customer Share (%)": (cnt / total_customers * 100.0) if total_customers > 0 else 0.0,
            "Total Revenue": seg_rev,
            "Revenue Share (%)": (seg_rev / total_revenue * 100.0) if total_revenue > 0 else 0.0,
            "Avg Recency (Days)": round(avg_rec, 1),
            "Avg Frequency": round(avg_freq, 1),
            "Avg Monetary Value": round(avg_mon, 2),
            "Description": meta["description"],
            "Recommended Action": meta["action"],
        })

    summary_df = pd.DataFrame(summary_list).sort_values("Total Revenue", ascending=False).reset_index(drop=True)

    return df, summary_df
