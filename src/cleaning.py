"""
RetailPulse - Cleaning and Transformation Pipeline

Implements:
1. Column validation and type standardization.
2. String customer ID preservation (handles floating numbers like 17850.0).
3. Exact duplicate row detection, documentation, and deduplication.
4. Quarantine logic for invalid prices, invalid timestamps, and zero quantities.
5. Cancellation and return isolation without double-counting.
6. Support for missing CustomerIDs in sales reporting while tracking excluded share.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class CleaningSummary:
    total_input_rows: int = 0
    exact_duplicates_removed: int = 0
    quarantined_rows: int = 0
    retained_clean_rows: int = 0
    quarantine_reasons: Dict[str, int] = field(default_factory=dict)
    cancellation_rows: int = 0
    return_rows: int = 0
    missing_customer_id_rows: int = 0
    missing_customer_id_revenue: float = 0.0
    missing_customer_id_pct_rows: float = 0.0
    missing_customer_id_pct_revenue: float = 0.0
    min_date: Optional[pd.Timestamp] = None
    max_date: Optional[pd.Timestamp] = None


def format_customer_id(val) -> Optional[str]:
    """
    Standardizes CustomerID to clean string, handling float representations like 17850.0.
    Returns None for missing, NaN, empty strings or 'nan'/'none'.
    """
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if val_str.lower() in ("", "nan", "none", "null", "undefined"):
        return None
    if val_str.endswith(".0"):
        val_str = val_str[:-2]
    return val_str if val_str else None


def clean_raw_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, CleaningSummary]:
    """
    Executes the full cleaning and transformation pipeline.

    Returns:
    - clean_df: Cleaned analytical transaction DataFrame.
    - quarantine_df: Quarantined records with explicit reasons.
    - summary: CleaningSummary metrics.
    """
    summary = CleaningSummary()
    summary.total_input_rows = len(df_raw)

    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame(), summary

    df = df_raw.copy()

    # 1. Clean CustomerID strings early
    df["CustomerID"] = df["CustomerID"].apply(format_customer_id)

    # 2. Check for exact duplicate rows across original columns
    orig_cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    avail_orig_cols = [c for c in orig_cols if c in df.columns]

    exact_dupe_mask = df.duplicated(subset=avail_orig_cols, keep="first")
    summary.exact_duplicates_removed = int(exact_dupe_mask.sum())
    # Remove exact duplicate lines and keep first occurrence
    df = df[~exact_dupe_mask].copy().reset_index(drop=True)

    # 3. Standardize string fields
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Country"] = df["Country"].fillna("Unspecified").astype(str).str.strip()
    df["Description"] = df["Description"].fillna("(No description)").astype(str).str.strip()
    # Replace empty description with fallback
    df.loc[df["Description"] == "", "Description"] = "(No description)"

    # 4. Quarantine checks
    quarantine_records = []
    quarantine_indices = set()

    # A. Timestamp validation
    parsed_dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    bad_date_mask = parsed_dates.isna()
    for idx in df[bad_date_mask].index:
        quarantine_indices.add(idx)
        quarantine_records.append({
            **df.loc[idx].to_dict(),
            "quarantine_reason": "Invalid or unparseable timestamp",
        })

    # B. UnitPrice validation: must be numeric and > 0
    numeric_prices = pd.to_numeric(df["UnitPrice"], errors="coerce")
    bad_price_null_mask = numeric_prices.isna()
    bad_price_nonpos_mask = (numeric_prices <= 0) & (~bad_price_null_mask)

    for idx in df[bad_price_null_mask].index:
        if idx not in quarantine_indices:
            quarantine_indices.add(idx)
            quarantine_records.append({
                **df.loc[idx].to_dict(),
                "quarantine_reason": "Missing or non-numeric unit price",
            })

    for idx in df[bad_price_nonpos_mask].index:
        if idx not in quarantine_indices:
            quarantine_indices.add(idx)
            reason = "Zero unit price" if numeric_prices.loc[idx] == 0 else "Negative unit price"
            quarantine_records.append({
                **df.loc[idx].to_dict(),
                "quarantine_reason": f"{reason} (bad debt / adjustment / damaged stock)",
            })

    # C. Quantity validation: must be numeric and != 0
    numeric_quantities = pd.to_numeric(df["Quantity"], errors="coerce")
    bad_qty_mask = numeric_quantities.isna() | (numeric_quantities == 0)
    for idx in df[bad_qty_mask].index:
        if idx not in quarantine_indices:
            quarantine_indices.add(idx)
            quarantine_records.append({
                **df.loc[idx].to_dict(),
                "quarantine_reason": "Zero or missing quantity",
            })

    # Compile quarantined DataFrame
    quarantine_df = pd.DataFrame(quarantine_records)
    summary.quarantined_rows = len(quarantine_indices)

    if not quarantine_df.empty:
        summary.quarantine_reasons = quarantine_df["quarantine_reason"].value_counts().to_dict()

    # Retain clean rows
    clean_mask = ~df.index.isin(quarantine_indices)
    clean_df = df[clean_mask].copy().reset_index(drop=True)
    summary.retained_clean_rows = len(clean_df)

    if clean_df.empty:
        return clean_df, quarantine_df, summary

    # 5. Type standardization on clean rows
    clean_df["InvoiceDate"] = parsed_dates[clean_mask].reset_index(drop=True)
    clean_df["Quantity"] = numeric_quantities[clean_mask].astype(int).reset_index(drop=True)
    clean_df["UnitPrice"] = numeric_prices[clean_mask].astype(float).reset_index(drop=True)

    # 6. Flag cancellations and returns without double counting
    clean_df["is_cancellation"] = clean_df["InvoiceNo"].str.upper().str.startswith("C")
    # A return is either explicitly flagged by invoice 'C' or has a negative quantity
    clean_df["is_return"] = clean_df["is_cancellation"] | (clean_df["Quantity"] < 0)

    # Qualifying sale is strictly a positive transaction: positive quantity and positive price, not a cancellation
    clean_df["is_qualifying_sale"] = (~clean_df["is_return"]) & (clean_df["Quantity"] > 0) & (clean_df["UnitPrice"] > 0)

    # 7. Financial calculation columns
    # Signed line amount: positive for sales, negative for returns
    clean_df["line_amount"] = clean_df["Quantity"] * clean_df["UnitPrice"]
    # Gross amount: sales revenue on qualifying transactions
    clean_df["gross_amount"] = np.where(clean_df["is_qualifying_sale"], clean_df["line_amount"], 0.0)
    # Return amount: positive magnitude of returned value
    clean_df["return_amount"] = np.where(clean_df["is_return"], np.abs(clean_df["Quantity"]) * clean_df["UnitPrice"], 0.0)
    # Net amount: gross - return
    clean_df["net_amount"] = clean_df["gross_amount"] - clean_df["return_amount"]

    # 8. Derived Date Attributes for time-series & cohort analysis
    clean_df["date"] = clean_df["InvoiceDate"].dt.date
    clean_df["year_month"] = clean_df["InvoiceDate"].dt.to_period("M").astype(str)
    # Monday of the week for weekly aggregation
    clean_df["week_start"] = clean_df["InvoiceDate"].dt.to_period("W-SUN").apply(lambda r: r.start_time.date())
    clean_df["day_of_week"] = clean_df["InvoiceDate"].dt.day_name()
    clean_df["hour"] = clean_df["InvoiceDate"].dt.hour

    # Customer ID presence flag
    clean_df["has_customer_id"] = clean_df["CustomerID"].notna()

    # Generate unique surrogate line ID
    clean_df["transaction_id"] = [
        f"TX_{inv}_{code}_{i}"
        for i, (inv, code) in enumerate(zip(clean_df["InvoiceNo"], clean_df["StockCode"]))
    ]

    # Rename columns to snake_case for consistent analytical naming
    rename_cols = {
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "UnitPrice": "unit_price",
        "InvoiceDate": "invoice_date",
        "CustomerID": "customer_id",
        "Country": "country",
    }
    clean_df.rename(columns=rename_cols, inplace=True)

    # Populate summary metadata
    summary.cancellation_rows = int(clean_df["is_cancellation"].sum())
    summary.return_rows = int(clean_df["is_return"].sum())
    missing_cust_mask = ~clean_df["has_customer_id"]
    summary.missing_customer_id_rows = int(missing_cust_mask.sum())
    summary.missing_customer_id_revenue = float(clean_df.loc[missing_cust_mask & clean_df["is_qualifying_sale"], "gross_amount"].sum())

    total_gross = float(clean_df.loc[clean_df["is_qualifying_sale"], "gross_amount"].sum())
    summary.missing_customer_id_pct_rows = (
        (summary.missing_customer_id_rows / len(clean_df) * 100.0) if len(clean_df) > 0 else 0.0
    )
    summary.missing_customer_id_pct_revenue = (
        (summary.missing_customer_id_revenue / total_gross * 100.0) if total_gross > 0 else 0.0
    )

    summary.min_date = clean_df["invoice_date"].min()
    summary.max_date = clean_df["invoice_date"].max()

    return clean_df, quarantine_df, summary
