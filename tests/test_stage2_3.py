"""
Stage 2 & 3 Tests: Customer Segmentation, Cohort Retention, Demand Forecasting, and Business Insights.
Validates:
- RFM scoring with tied frequency values.
- Explicit analysis date handling.
- Cohort retention matrices and future period blanking.
- Chronological forecasting evaluation and WAPE zero-handling.
- Deterministic insight card structure.
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest

from src.cleaning import clean_raw_data
from src.database import DatabaseManager
from src.metrics import compute_comparison
from src.segmentation import compute_rfm_segmentation, score_rfm_quantiles, assign_rfm_segment
from src.cohorts import compute_cohort_retention
from src.forecasting import calculate_wape, generate_sales_forecast
from src.insights import generate_business_insights


def test_rfm_quantile_scoring_with_heavy_ties():
    """Verify that rank-based quantile scoring handles tied values without error."""
    # 80 out of 100 customers have frequency = 1
    tied_series = pd.Series([1] * 80 + [2] * 10 + [5] * 7 + [20] * 3)
    scores = score_rfm_quantiles(tied_series, ascending=True, num_bins=4)

    assert len(scores) == 100
    assert not scores.isna().any()
    assert scores.min() >= 1
    assert scores.max() <= 4
    # The high spenders (frequency 20) should have score 4
    assert scores.iloc[-1] == 4


def test_rfm_segment_assignment():
    """Verify segment assignment rule mapping."""
    assert assign_rfm_segment(4, 4, 4) == "Champions"
    assert assign_rfm_segment(4, 1, 1) == "New Customers"
    assert assign_rfm_segment(2, 4, 3) == "At Risk"
    assert assign_rfm_segment(1, 1, 1) == "Hibernating"


def test_cohort_unobserved_future_periods_blank():
    """
    Verify that future periods beyond dataset max date are NaN (blank),
    never reported as false 0.0% retention.
    """
    # 2 months dataset: 2024-01 and 2024-02
    raw_df = pd.DataFrame([
        {"InvoiceNo": "1", "StockCode": "P1", "Description": "P1", "Quantity": 1, "InvoiceDate": "2024-01-10 10:00:00", "UnitPrice": 10.0, "CustomerID": "C1", "Country": "UK"},
        {"InvoiceNo": "2", "StockCode": "P1", "Description": "P1", "Quantity": 1, "InvoiceDate": "2024-02-10 10:00:00", "UnitPrice": 10.0, "CustomerID": "C1", "Country": "UK"},
        {"InvoiceNo": "3", "StockCode": "P1", "Description": "P1", "Quantity": 1, "InvoiceDate": "2024-02-15 10:00:00", "UnitPrice": 10.0, "CustomerID": "C2", "Country": "UK"},
    ])
    clean_df, _, _ = clean_raw_data(raw_df)
    db = DatabaseManager(":memory:")
    db.populate_database(clean_df)

    pct_pivot, counts_pivot, cohort_sizes, notes = compute_cohort_retention(db)

    # 2024-01 cohort has M+0 (Jan) and M+1 (Feb) observed
    assert "M+0" in pct_pivot.columns
    assert "M+1" in pct_pivot.columns
    assert pct_pivot.loc["2024-01", "M+0"] == 100.0
    assert pct_pivot.loc["2024-01", "M+1"] == 100.0

    # 2024-02 cohort has M+0 (Feb) observed, but M+1 (March) is future and must be NaN!
    assert pct_pivot.loc["2024-02", "M+0"] == 100.0
    assert np.isnan(pct_pivot.loc["2024-02", "M+1"])
    assert np.isnan(counts_pivot.loc["2024-02", "M+1"])


def test_wape_calculation_and_zero_handling():
    """Verify WAPE calculation accuracy and zero denominator safety."""
    actual = np.array([100.0, 200.0, 300.0])
    pred = np.array([110.0, 190.0, 330.0])
    # abs err = 10 + 10 + 30 = 50. total actual = 600. WAPE = 50 / 600 * 100 = 8.333%
    assert calculate_wape(actual, pred) == pytest.approx(8.3333, rel=1e-3)

    # Zero actuals
    zero_actual = np.array([0.0, 0.0])
    zero_pred = np.array([0.0, 0.0])
    assert calculate_wape(zero_actual, zero_pred) == 0.0


def test_forecasting_insufficient_history():
    """Verify that forecasting returns clear status when history is below threshold."""
    raw_df = pd.DataFrame([
        {"InvoiceNo": "1", "StockCode": "P1", "Description": "P1", "Quantity": 1, "InvoiceDate": "2024-01-10 10:00:00", "UnitPrice": 10.0, "CustomerID": "C1", "Country": "UK"},
    ])
    clean_df, _, _ = clean_raw_data(raw_df)
    db = DatabaseManager(":memory:")
    db.populate_database(clean_df)

    res = generate_sales_forecast(db, min_history_required=12)
    assert res.is_sufficient_history is False
    assert "Insufficient historical data" in res.status_message


def test_deterministic_insights_structure():
    """Verify that all generated insight cards follow the required 5-part schema."""
    raw_df = pd.DataFrame([
        {"InvoiceNo": "1", "StockCode": "P1", "Description": "P1", "Quantity": 10, "InvoiceDate": "2024-01-15 10:00:00", "UnitPrice": 10.0, "CustomerID": "C1", "Country": "UK"},
        {"InvoiceNo": "2", "StockCode": "P2", "Description": "P2", "Quantity": 5, "InvoiceDate": "2024-01-20 10:00:00", "UnitPrice": 20.0, "CustomerID": "C2", "Country": "UK"},
        {"InvoiceNo": "C3", "StockCode": "P1", "Description": "P1", "Quantity": -2, "InvoiceDate": "2024-01-25 10:00:00", "UnitPrice": 10.0, "CustomerID": "C1", "Country": "UK"},
    ])
    clean_df, _, _ = clean_raw_data(raw_df)
    db = DatabaseManager(":memory:")
    db.populate_database(clean_df)

    comp = compute_comparison(db, start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
    insights = generate_business_insights(db, comp)

    assert len(insights) >= 2
    for ins in insights:
        assert ins.title != ""
        assert ins.category != ""
        assert ins.numerical_observation != ""
        assert ins.period_and_comparison != ""
        assert ins.cautious_interpretation != ""
        assert ins.recommended_action != ""
        assert ins.important_limitation != ""
