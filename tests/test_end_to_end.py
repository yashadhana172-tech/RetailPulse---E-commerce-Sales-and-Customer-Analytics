"""
End-to-End & Acceptance Tests for RetailPulse

Covers:
- Ingestion and schema validation of corrupted/invalid files.
- SQL aggregation correctness verified against direct pandas calculation.
- Country and date filtering consistency.
- Forecasting model selection and prediction bounds.
- Deterministic demo data generator integrity.
"""

from datetime import date
import io
import numpy as np
import pandas as pd
import pytest

from src.ingestion import generate_synthetic_demo_data, load_file, validate_schema
from src.cleaning import clean_raw_data
from src.database import DatabaseManager
from src.metrics import compute_comparison, compute_kpis_for_period
from src.segmentation import compute_rfm_segmentation
from src.cohorts import compute_cohort_retention
from src.forecasting import generate_sales_forecast


def test_synthetic_data_generator_properties():
    """Verify deterministic generation properties of demo dataset."""
    df = generate_synthetic_demo_data(seed=42)
    assert len(df) > 10000
    # Must contain both regular and cancellation invoices
    assert df["InvoiceNo"].str.startswith("C").any()
    # Must have negative quantities
    assert (df["Quantity"] < 0).any()
    # Must have missing customer IDs
    assert df["CustomerID"].isna().any()
    # Check date coverage: 24 months
    # Convert dates to check min and max
    valid_dates = pd.to_datetime(df["InvoiceDate"], errors="coerce").dropna()
    assert (valid_dates.max() - valid_dates.min()).days >= 700


def test_invalid_csv_schema_rejected():
    """Verify that an arbitrary CSV missing required columns is rejected."""
    bad_csv = io.StringIO("Name,Age,Score\nAlice,30,95\nBob,25,88")
    df, val = load_file(bad_csv, file_type="csv")
    assert df is None
    assert val.is_valid is False
    assert "InvoiceNo" in val.missing_columns
    assert "StockCode" in val.missing_columns


def test_sql_kpi_aggregation_matches_pandas_ground_truth():
    """
    Verify that DuckDB SQL aggregations exactly match independent pandas aggregations
    across gross sales, returns, and distinct counts.
    """
    raw_df = generate_synthetic_demo_data(seed=42)
    clean_df, _, _ = clean_raw_data(raw_df)
    db = DatabaseManager(":memory:")
    db.populate_database(clean_df)

    start_d = date(2024, 6, 1)
    end_d = date(2024, 8, 31)

    kpis = compute_kpis_for_period(db, start_date=start_d, end_date=end_d, country="ALL")

    # Pandas ground truth on the same slice
    slice_df = clean_df[(clean_df["date"] >= start_d) & (clean_df["date"] <= end_d)]
    pd_gross = slice_df.loc[slice_df["is_qualifying_sale"], "gross_amount"].sum()
    pd_return = slice_df.loc[slice_df["is_return"], "return_amount"].sum()
    pd_orders = slice_df.loc[slice_df["is_qualifying_sale"], "invoice_no"].nunique()
    pd_custs = slice_df.loc[slice_df["is_qualifying_sale"] & slice_df["has_customer_id"], "customer_id"].nunique()

    assert kpis.gross_sales == pytest.approx(pd_gross, rel=1e-5)
    assert kpis.return_value == pytest.approx(pd_return, rel=1e-5)
    assert kpis.completed_sales_orders == pd_orders
    assert kpis.purchasing_customers == pd_custs


def test_country_filter_isolation():
    """Verify country filtering isolates only transactions from that market."""
    raw_df = generate_synthetic_demo_data(seed=42)
    clean_df, _, _ = clean_raw_data(raw_df)
    db = DatabaseManager(":memory:")
    db.populate_database(clean_df)

    start_d = date(2024, 1, 1)
    end_d = date(2024, 12, 31)

    kpis_germany = compute_kpis_for_period(db, start_d, end_d, country="Germany")
    kpis_uk = compute_kpis_for_period(db, start_d, end_d, country="United Kingdom")
    kpis_all = compute_kpis_for_period(db, start_d, end_d, country="ALL")

    assert kpis_germany.gross_sales > 0
    assert kpis_uk.gross_sales > 0
    assert kpis_all.gross_sales >= (kpis_germany.gross_sales + kpis_uk.gross_sales)


def test_forecasting_output_structure_on_full_demo():
    """Verify that forecasting on 24 months of data produces point forecasts and prediction intervals."""
    raw_df = generate_synthetic_demo_data(seed=42)
    clean_df, _, _ = clean_raw_data(raw_df)
    db = DatabaseManager(":memory:")
    db.populate_database(clean_df)

    res = generate_sales_forecast(db, country="ALL", horizon_weeks=4)
    assert res.is_sufficient_history is True
    assert res.future_forecast_df is not None
    assert len(res.future_forecast_df) == 4
    # Predictions must be >= 0
    assert (res.future_forecast_df["forecast_net_revenue"] >= 0).all()
    # Lower bound must be <= upper bound
    assert (res.future_forecast_df["ci_lower_95"] <= res.future_forecast_df["ci_upper_95"]).all()
