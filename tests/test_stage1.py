"""
Stage 1 Tests: Data Ingestion, Cleaning, Deduplication, Quarantine, and KPI Correctness
Using small hand-checkable fixtures with exact mathematical expectations.
"""

from datetime import date
import pandas as pd
import pytest

from src.ingestion import validate_schema, ValidationResult
from src.cleaning import clean_raw_data, format_customer_id
from src.database import DatabaseManager
from src.metrics import compute_kpis_for_period


def test_customer_id_formatting():
    """Verify string preservation and float/null handling for CustomerID."""
    assert format_customer_id(12345.0) == "12345"
    assert format_customer_id("12345") == "12345"
    assert format_customer_id("  12345  ") == "12345"
    assert format_customer_id(None) is None
    assert format_customer_id(float("nan")) is None
    assert format_customer_id("nan") is None
    assert format_customer_id("") is None


def test_schema_validation_success():
    """Verify valid schema detection."""
    df = pd.DataFrame({
        "InvoiceNo": ["536365"],
        "StockCode": ["85123A"],
        "Description": ["WHITE HEART"],
        "Quantity": [6],
        "InvoiceDate": ["2024-01-01 10:00:00"],
        "UnitPrice": [2.55],
        "CustomerID": ["17850"],
        "Country": ["United Kingdom"],
    })
    res = validate_schema(df)
    assert res.is_valid is True
    assert len(res.missing_columns) == 0


def test_schema_validation_failure():
    """Verify detection of missing required columns."""
    df = pd.DataFrame({
        "InvoiceNo": ["536365"],
        "StockCode": ["85123A"],
        "Quantity": [6],
    })
    res = validate_schema(df)
    assert res.is_valid is False
    assert "Description" in res.missing_columns
    assert "UnitPrice" in res.missing_columns


def test_hand_checkable_cleaning_and_kpis():
    """
    Test end-to-end cleaning, deduplication, quarantine, and DuckDB KPIs
    against an exact 9-row hand-crafted ledger.
    """
    raw_fixture = pd.DataFrame([
        # Row 1: Qualifying sale, Inv 101, Customer 1
        {"InvoiceNo": "101", "StockCode": "ITEM_A", "Description": "Item A", "Quantity": 2, "InvoiceDate": "2024-01-01 09:00:00", "UnitPrice": 10.0, "CustomerID": 1.0, "Country": "UK"},
        # Row 2: Qualifying sale, Inv 101, Customer 1
        {"InvoiceNo": "101", "StockCode": "ITEM_B", "Description": "Item B", "Quantity": 1, "InvoiceDate": "2024-01-01 09:00:00", "UnitPrice": 5.0, "CustomerID": "1", "Country": "UK"},
        # Row 3: Qualifying sale, Inv 102, Customer 2
        {"InvoiceNo": "102", "StockCode": "ITEM_A", "Description": "Item A", "Quantity": 3, "InvoiceDate": "2024-01-02 11:00:00", "UnitPrice": 10.0, "CustomerID": "2", "Country": "UK"},
        # Row 4: Qualifying sale, Inv 103, Guest (CustomerID = None)
        {"InvoiceNo": "103", "StockCode": "ITEM_C", "Description": "Item C", "Quantity": 4, "InvoiceDate": "2024-01-03 14:00:00", "UnitPrice": 15.0, "CustomerID": None, "Country": "France"},
        # Row 5: Return / Cancellation, Inv C104, Customer 1
        {"InvoiceNo": "C104", "StockCode": "ITEM_A", "Description": "Item A", "Quantity": -1, "InvoiceDate": "2024-01-04 10:00:00", "UnitPrice": 10.0, "CustomerID": "1", "Country": "UK"},
        # Row 6: Qualifying sale, Inv 105, Customer 1 (2nd distinct order for Customer 1)
        {"InvoiceNo": "105", "StockCode": "ITEM_A", "Description": "Item A", "Quantity": 1, "InvoiceDate": "2024-01-05 16:00:00", "UnitPrice": 10.0, "CustomerID": "1", "Country": "UK"},
        # Row 7: Exact duplicate of Row 1
        {"InvoiceNo": "101", "StockCode": "ITEM_A", "Description": "Item A", "Quantity": 2, "InvoiceDate": "2024-01-01 09:00:00", "UnitPrice": 10.0, "CustomerID": 1.0, "Country": "UK"},
        # Row 8: Invalid price (negative) -> quarantine
        {"InvoiceNo": "106", "StockCode": "ITEM_D", "Description": "Bad debt adjust", "Quantity": 1, "InvoiceDate": "2024-01-06 10:00:00", "UnitPrice": -5.0, "CustomerID": "3", "Country": "UK"},
        # Row 9: Invalid unparseable date -> quarantine
        {"InvoiceNo": "107", "StockCode": "ITEM_E", "Description": "Bad date", "Quantity": 1, "InvoiceDate": "CORRUPT_TIMESTAMP", "UnitPrice": 10.0, "CustomerID": "3", "Country": "UK"},
    ])

    # 1. Clean data
    clean_df, quarantine_df, summary = clean_raw_data(raw_fixture)

    assert summary.total_input_rows == 9
    assert summary.exact_duplicates_removed == 1
    assert summary.quarantined_rows == 2
    assert summary.retained_clean_rows == 6
    assert len(clean_df) == 6
    assert len(quarantine_df) == 2

    # Check quarantine reasons
    reasons = quarantine_df["quarantine_reason"].tolist()
    assert any("timestamp" in r.lower() for r in reasons)
    assert any("price" in r.lower() for r in reasons)

    # 2. Populate DuckDB
    db = DatabaseManager(":memory:")
    db.populate_database(clean_df)

    # 3. Compute KPIs
    kpis = compute_kpis_for_period(
        db=db,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        country="ALL",
    )

    # Hand-checkable calculations:
    # Row 1: 2 * 10 = 20
    # Row 2: 1 * 5 = 5
    # Row 3: 3 * 10 = 30
    # Row 4: 4 * 15 = 60
    # Row 6: 1 * 10 = 10
    # Total Gross = 125.0
    assert kpis.gross_sales == pytest.approx(125.0, rel=1e-5)

    # Row 5: Return 1 * 10 = 10.0
    assert kpis.return_value == pytest.approx(10.0, rel=1e-5)

    # Net revenue = 125 - 10 = 115.0
    assert kpis.net_revenue == pytest.approx(115.0, rel=1e-5)

    # Qualifying orders: 101, 102, 103, 105 -> 4
    assert kpis.completed_sales_orders == 4

    # AOV: 125 / 4 = 31.25
    assert kpis.average_order_value == pytest.approx(31.25, rel=1e-5)

    # Units sold: 2 + 1 + 3 + 4 + 1 = 11
    assert kpis.units_sold == 11
    # Returned units: 1
    assert kpis.returned_units == 1

    # Purchasing customers (known): Customer 1, Customer 2 -> 2
    assert kpis.purchasing_customers == 2

    # Repeat customers: Customer 1 has order 101 and 105 -> 1 repeat customer
    assert kpis.repeat_customers_count == 1
    # Repeat rate: 1 / 2 = 50.0%
    assert kpis.repeat_purchase_rate_pct == pytest.approx(50.0, rel=1e-5)

    # Guest order revenue: Row 4 is 60.0. Share = 60 / 125 = 48%
    assert kpis.missing_customer_orders_count == 1
    assert kpis.missing_customer_revenue_share_pct == pytest.approx((60.0 / 125.0) * 100.0, rel=1e-5)


def test_empty_period_and_zero_denominators():
    """Verify that empty periods don't crash and safely return zero values."""
    db = DatabaseManager(":memory:")
    # Empty DB
    kpis = compute_kpis_for_period(
        db=db,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        country="ALL",
    )
    assert kpis.gross_sales == 0.0
    assert kpis.return_value == 0.0
    assert kpis.net_revenue == 0.0
    assert kpis.completed_sales_orders == 0
    assert kpis.average_order_value == 0.0
    assert kpis.purchasing_customers == 0
    assert kpis.repeat_purchase_rate_pct == 0.0
