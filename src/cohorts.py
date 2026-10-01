"""
RetailPulse - Cohort Retention Analysis Module

Implements:
1. Acquisition cohort assignment from customer's first observed qualifying purchase
   in the full loaded dataset.
2. Monthly repeat-purchase retention matrix calculation.
3. Strict preservation of unobserved future periods as NaN / blank (never false zeros).
4. Cohort size tracking and retention percentage matrices.
5. Analytical caveats regarding first observed purchase vs. true commercial acquisition.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.database import DatabaseManager


COHORT_METHODOLOGY_NOTES = {
    "Acquisition Definition": (
        "Each customer is assigned to an acquisition cohort based on the month of their "
        "first observed qualifying purchase across the entire dataset."
    ),
    "Left-Censoring Limitation": (
        "The first observed purchase in an extracted dataset does not guarantee true brand acquisition; "
        "customers active in the first month may have been buying for years prior to dataset capture."
    ),
    "Future Periods Representation": (
        "Calendar months beyond the maximum date present in the dataset have not occurred yet. "
        "These cells are intentionally left blank (NaN) rather than reported as 0% retention."
    ),
    "Filter Mechanics": (
        "Country filtering restricts the analysis to transactions originating from the selected country. "
        "Only customers with qualifying purchases in that country are evaluated."
    ),
}


def compute_cohort_retention(
    db: DatabaseManager,
    country: str = "ALL",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Computes monthly cohort retention matrices.

    Returns:
    - retention_matrix_pct: DataFrame of retention rates (%) with future periods as NaN.
    - cohort_counts_matrix: DataFrame of active customer counts with future periods as NaN.
    - cohort_sizes: Series of cohort sizes (total customers acquired in Month 0).
    - notes: Dictionary of methodology caveats and interpretations.
    """
    sql = """
    WITH customer_first_purchase AS (
        SELECT
            customer_id,
            MIN(date) AS first_order_date,
            STRFTIME(MIN(date), '%Y-%m') AS cohort_month
        FROM fct_transactions
        WHERE is_qualifying_sale AND has_customer_id
          AND ($country = 'ALL' OR country = $country)
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
          AND ($country = 'ALL' OR t.country = $country)
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
        r.active_customers
    FROM retention_counts r
    INNER JOIN cohort_sizes s ON r.cohort_month = s.cohort_month
    ORDER BY r.cohort_month ASC, r.month_number ASC;
    """

    df = db.query(sql, {"country": country})

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=int), COHORT_METHODOLOGY_NOTES

    # Find total calendar months in dataset to identify future unobserved periods
    bounds_df = db.query("SELECT MIN(date) AS min_d, MAX(date) AS max_d FROM fct_transactions;")
    max_date = pd.to_datetime(bounds_df["max_d"].iloc[0])
    max_year = max_date.year
    max_month = max_date.month

    # Pivot customer counts
    counts_pivot = df.pivot(index="cohort_month", columns="month_number", values="active_customers")
    cohort_sizes = df.groupby("cohort_month")["cohort_size"].first()

    # Calculate percentage retention
    pct_pivot = counts_pivot.divide(cohort_sizes, axis=0) * 100.0

    # Mask future unobserved periods as NaN
    for cohort_str in pct_pivot.index:
        c_year = int(cohort_str[:4])
        c_month = int(cohort_str[5:7])
        for col_offset in pct_pivot.columns:
            # Calendar month for this cell
            cell_year = c_year + (c_month - 1 + col_offset) // 12
            cell_month = ((c_month - 1 + col_offset) % 12) + 1
            if (cell_year > max_year) or (cell_year == max_year and cell_month > max_month):
                pct_pivot.loc[cohort_str, col_offset] = np.nan
                counts_pivot.loc[cohort_str, col_offset] = np.nan

    # Format column headers as "Month 0", "Month 1", etc.
    col_names = {c: f"M+{c}" for c in pct_pivot.columns}
    pct_pivot.rename(columns=col_names, inplace=True)
    counts_pivot.rename(columns=col_names, inplace=True)

    # Insert Cohort Size as first column in counts_pivot
    counts_pivot.insert(0, "Cohort Size", cohort_sizes)

    return pct_pivot, counts_pivot, cohort_sizes, COHORT_METHODOLOGY_NOTES
